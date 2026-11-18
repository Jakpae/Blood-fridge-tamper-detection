// 혈액 냉장고 시뮬레이터 + 탐지 엔진 v2 (브라우저/Node 공용)
// 실험 코드(sim.py, detect.py)와 같은 식을 쓰되, 여러 사건을 동시에 켤 수 있게 했다.
const P = { a: 1/300, b: 0.25, bTa: 0.015, ramp: 3, h: 0.07, door: 0.30, pn: 0.01,
  set: 4.0, band: 0.5, defEvery: 480, defOff: 120, defLen: 10, evapDt: 12, pTau: 3,
  wIdle: 35, wComp: 300, wInrush: 120, wHeat: 550, nT: 0.05, nP: 0.02, nTa: 0.1 };
// 모델이 정상 데이터에서 배운 값 (냉장고 0번, step2 결과)
const M = { a: 0.003369, b: 0.2151, h: 0.0670, ramp: 3 };
const SAFE_HI = 6.0, SAFE_LO = 1.0;
const ST = { IDLE: 0, COOL: 1, HEAT: 2, OFF: 3, BOTH: 4 };

function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function makeGauss(r) { return () => { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }; }
const psat = t => Math.exp(10.975 - 2705 / (t + 273.15));
const tsat = p => 2705 / (10.975 - Math.log(p)) - 273.15;

class Fridge {
  constructor(seed) {
    this.r = rng(seed); this.g = makeGauss(this.r);
    this.clock = 0; this.T = P.set; this.Te = P.set; this.uPrev = 0; this.since = 99; this.walk = 0;
    // 실제로 일어나는 일 (여러 개를 동시에 켤 수 있음)
    this.ev = { doorStuck: false, heaterOn: false, compOff: false, compOn: false, powerCut: false };
    this.doorPulse = 0; // 이번 분에 사람이 문을 연 시간 [초]
  }
  ambient() { this.walk += this.g() * 0.02; this.walk *= 0.995; const h = this.clock / 60; return 22 + 3 * Math.sin(2 * Math.PI * (h - 9) / 24) + this.walk; }
  step() {
    const g = this.g, ev = this.ev, hour = (this.clock / 60) % 24;
    const Ta = this.ambient();
    // 문: 현장 자문값. 주간 시간당 3~4회, 야간 1~2회, 한 번 열면 3~10초
    const rate = (hour >= 8 && hour < 20) ? 3.5 / 60 : 1.5 / 60;
    let door = this.r() < rate ? (3 + this.r() * 7) / 60 : 0;
    door += this.doorPulse / 60; this.doorPulse = 0;
    if (ev.doorStuck) door = 1;
    door = Math.min(1, door);
    // 제어기
    const Tc = this.T + g() * P.nT;
    const defr = ((this.clock - P.defOff) % P.defEvery + P.defEvery) % P.defEvery < P.defLen;
    let u, d;
    if (defr) { u = 0; d = 1; } else { d = 0; u = Tc > P.set + P.band ? 1 : Tc < P.set - P.band ? 0 : this.uPrev; }
    if (ev.heaterOn) { d = 1; u = 0; }          // 공격자가 제상 모드로 고정 (설정 조작)
    if (ev.compOff) u = 0;                        // 냉각 정지
    if (ev.compOn) u = 1;                         // 냉각 계속 (온도 조절 무시)
    if (ev.powerCut) { u = 0; d = 0; }            // 정전·플러그 빠짐
    this.since = (u && this.uPrev) ? this.since + 1 : (u ? 0 : 99);
    const ramp = u ? Math.min(1, (this.since + 1) / P.ramp) : 0;
    const bEff = P.b * (1 - P.bTa * (Ta - 22)) * ramp;
    this.T += P.a * (Ta - this.T) - bEff + P.h * d + P.door * door + g() * P.pn;
    const target = u ? this.T - P.evapDt : (d ? this.T + 3 : this.T);
    this.Te += (target - this.Te) / P.pTau;
    const Pr = psat(this.Te);
    let W = u ? P.wComp + 3 * (Ta - 20) + 40 * (Pr - 2) + (this.since < 2 ? P.wInrush : 0) : P.wIdle;
    if (d) W += P.wHeat;
    if (ev.powerCut) W = 0;                       // 감시 모듈은 자체 배터리로 계속 측정
    W = Math.max(0, W + g() * (u ? 6 : ev.powerCut ? 0.5 : 3));
    this.uPrev = u;
    const rec = { t: this.clock, Ttrue: this.T, Trep: this.T + g() * P.nT, Prep: Pr + g() * P.nP, W, Ta: Ta + g() * P.nTa, u, d, door };
    this.clock++;
    return rec;
  }
}

// 전력으로 작동 상태 판별 (학습한 혼합 모델과 같은 경계)
function powerState(W) { return W < 15 ? ST.OFF : W > 760 ? ST.BOTH : W > 520 ? ST.HEAT : W > 150 ? ST.COOL : ST.IDLE; }
const isCool = s => s === ST.COOL || s === ST.BOTH, isHeat = s => s === ST.HEAT || s === ST.BOTH;

function stepModel(T, e, m) {
  const u = isCool(e.st) ? 1 : 0, d = isHeat(e.st) ? 1 : 0;
  const ramp = u ? Math.min(1, ((e.sinceOn ?? 0) + 1) / m.ramp) : 0;
  return T + m.a * (e.Ta - T) - m.b * u * ramp + m.h * d;
}

const WHY = {
  thermal: '전력으로 계산한 온도보다 표시된 온도가 계속 낮습니다.',
  pressure: '압력으로 계산한 온도와 표시된 온도가 계속 어긋납니다.',
  heat: '제상 히터가 평소(10분)보다 오래 켜져 있습니다.',
  idle: '표시된 온도는 높은데 냉각기가 돌지 않습니다.',
  power: '전력이 0입니다. 정전이나 플러그 빠짐이 의심됩니다.',
  cold: '온도가 낮은데 냉각기가 멈추지 않습니다.',
  longcool: '냉각기가 40분 넘게 쉬지 않고 돕니다. 문 열림이나 냉각 고장이 의심됩니다.',
};

class Detector {
  constructor(cfg) { this.cfg = cfg; this.hist = []; this.lastState = -1; this.stateRun = 0; this.since = 99; this.clear(); }
  // 경보 확인 후 누적값과 경보를 비운다 (기록은 유지)
  clear() { this.sTh = 0; this.sVhi = 0; this.sVlo = 0; this.runTh = 0; this.runV = 0; this.heatRun = 0; this.warmIdle = 0; this.offRun = 0; this.coldRun = 0; this.coolRun = 0; this.okRun = 0; this.alarm = null; this.agree = null; this.lastVt = null; }
  virtualT(e) {
    if (e.run < 8 || e.st === ST.BOTH) return null;   // 상태가 막 바뀌었거나 냉각·히터가 같이 돌면 쓰지 않음
    const ts = tsat(e.Prep);
    return e.st === ST.COOL ? ts + P.evapDt : e.st === ST.HEAT ? ts - 3 : ts;
  }
  push(rec) {
    const h = this.hist, c = this.cfg;
    const st = powerState(rec.W);
    this.stateRun = st === this.lastState ? this.stateRun + 1 : 0; this.lastState = st;
    this.since = isCool(st) ? (h.length && isCool(h[h.length - 1].st) ? this.since + 1 : 0) : 99;
    const e = { ...rec, st, run: this.stateRun, sinceOn: this.since };
    h.push(e); if (h.length > 3000) h.shift();
    const n = h.length, out = { st };
    // (1) 전력 기반 30분 굴림 (화면 온도가 계산보다 낮을 때만 누적)
    if (n > 31) {
      let T = h[n - 31].Trep;
      for (let k = n - 31; k < n - 1; k++) T = stepModel(T, h[k], M);
      out.pred = T;
      const z = Math.max(-4, Math.min(4, (rec.Trep - T - c.thMu) / c.thSd));
      this.sTh = Math.max(0, this.sTh - z - 2);
    }
    out.sTh = this.sTh;
    // (2) 압력 기반 가상 온도 센서
    const vt = this.virtualT(e);
    out.virt = vt;
    if (vt !== null) {
      const z = Math.max(-4, Math.min(4, (rec.Trep - vt - c.vMu) / c.vSd));
      this.sVhi = Math.max(0, this.sVhi + z - 1); this.sVlo = Math.max(0, this.sVlo - z - 1);
    }
    out.sV = Math.max(this.sVhi, this.sVlo);
    this.runTh = this.sTh > 0 ? this.runTh + 1 : 0;
    this.runV = out.sV > 0 ? this.runV + 1 : 0;
    // (3) 제어 논리 검사
    this.heatRun = isHeat(st) ? this.heatRun + 1 : 0;
    this.warmIdle = (st === ST.IDLE && rec.Trep > P.set + P.band + 0.3) ? this.warmIdle + 1 : 0;
    this.offRun = st === ST.OFF ? this.offRun + 1 : 0;
    this.coldRun = (isCool(st) && rec.Trep < P.set - P.band - 0.3) ? this.coldRun + 1 : 0;
    this.coolRun = isCool(st) ? this.coolRun + 1 : 0;
    out.logic = this.offRun >= 2 ? 'power' : this.heatRun > P.defLen + 2 ? 'heat' : this.warmIdle >= 4 ? 'idle'
      : this.coldRun >= 4 ? 'cold' : this.coolRun > 40 ? 'longcool' : null;
    if (vt !== null && out.pred !== undefined) this.agree = (this.agree ?? Math.abs((vt + c.vMu) - (out.pred + c.thMu))) * 0.9 + 0.1 * Math.abs((vt + c.vMu) - (out.pred + c.thMu));
    let why = null;
    if (out.sTh > c.thThr) why = 'thermal'; else if (out.sV > c.vThr) why = 'pressure'; else if (out.logic) why = out.logic;
    if (!this.alarm && why) {
      const run = why === 'heat' ? this.heatRun : why === 'idle' ? this.warmIdle + 2 : why === 'power' ? this.offRun : why === 'cold' ? this.coldRun + 2 : why === 'longcool' ? 40 : why === 'pressure' ? this.runV : this.runTh;
      this.alarm = { t: rec.t, why, cause: null, onset: rec.t - Math.min(240, run) };
    }
    if (this.alarm) {
      const a = this.alarm;
      // 복원: 추정 시작 시점의 화면 온도에서 전력으로 굴림
      if (a.restoreT === undefined) {
        const i0 = h.findIndex(x => x.t === a.onset);
        let T = i0 >= 1 ? h[i0 - 1].Trep : rec.Trep;
        for (let k = Math.max(i0, 0); k < n; k++) T = stepModel(T, h[k], M);
        a.restoreT = T;
      } else a.restoreT = stepModel(a.restoreT, e, M);
      if (vt !== null) this.lastVt = vt;
      // 경보 10분 뒤 원인 판별
      //   화면 온도가 전력으로 계산한 온도와 맞음      → 센서는 정상, 냉장고 자체 이상(설정 조작·고장·정전)
      //   압력 온도와 전력 온도가 맞음                → 온도 센서만 조작
      //   그 밖                                      → 온도·압력 모두 조작
      if (a.cause === null && rec.t - a.t >= 10) {
        // 제어 논리로 울린 경보(히터·정전 등)에서 화면 온도가 압력 온도와 맞으면 센서는 정상
        // (문 열림처럼 전력 모델이 모르는 열 때문에 전력 계산과는 어긋날 수 있음)
        const pOk = a.why !== 'thermal' && a.why !== 'pressure' && this.lastVt != null && Math.abs(rec.Trep - (this.lastVt + c.vMu)) < 0.8;
        a.cause = (Math.abs(rec.Trep - a.restoreT) < 0.8 || pOk) ? 'none' : (this.agree ?? 9) < 0.8 ? 'T' : 'TP';
      }
      out.restore = a.cause === 'none' ? rec.Trep : a.cause === 'T' && this.lastVt != null ? this.lastVt + c.vMu : a.restoreT;
    }
    // 우리 시스템이 믿는 온도: 경보 중이면 되살린 온도, 아니면 화면 온도
    out.best = out.restore ?? rec.Trep;
    // 원인 판별 전에는 화면 값과 되살린 값 중 더 위험한 쪽으로 판단
    const cand = this.alarm && this.alarm.cause === null ? [out.best, rec.Trep] : [out.best];
    out.limit = Math.max(...cand) > SAFE_HI ? 'hi' : Math.min(...cand) < SAFE_LO ? 'lo' : null;   // 절대 기준 (평균이 아니라 값 자체로 판단)
    // 기준까지 남은 시간: 지금 상태가 이어진다고 가정
    if (this.alarm || out.limit) {
      let T = out.best, m = 0;
      const dir = stepModel(T, e, M) - T;
      if (out.limit) out.left = 0;
      else { while (T < SAFE_HI && T > SAFE_LO && m < 600) { T = stepModel(T, e, M); m++; } out.left = m < 600 ? m : null; out.leftTo = dir >= 0 ? SAFE_HI : SAFE_LO; }
    }
    // 자동 정상 복귀: 원인 판별이 끝난 경보에서 세 값이 다시 맞고 제어 논리 이상도 없으며
    // 온도가 기준 안에 있는 상태가 일정 시간 이어지면 경보를 내린다.
    // 실제 이상이든 센서 조작이든 10분 동안 이어져야 정상으로 본다.
    if (this.alarm && this.alarm.cause !== null) {
      // 센서 조작 경보는 지난 표시값에서 출발한 계산(조작된 값을 따라감)이 아니라,
      // 조작 시작 전부터 전력·압력으로 따로 계산해 온 실제 온도와 비교한다.
      const tamper = this.alarm.cause !== 'none';
      const okT = tamper ? (out.restore !== undefined && Math.abs(rec.Trep - out.restore) < 1.0)
        : (out.pred === undefined || Math.abs(rec.Trep - (out.pred + c.thMu)) < 1.0);
      const okP = vt === null || Math.abs(rec.Trep - (vt + c.vMu)) < 1.0;
      const ok = !out.logic && okT && okP && rec.Trep >= SAFE_LO && rec.Trep <= SAFE_HI;
      this.okRun = ok ? this.okRun + 1 : 0;
      const need = 10;
      if (this.okRun >= need) {
        out.recovered = { ...this.alarm, minutes: this.okRun };
        this.clear();
        out.restore = undefined; out.best = rec.Trep; out.limit = null; out.left = undefined;
      }
    }
    out.alarm = this.alarm;
    return out;
  }
}
if (typeof module !== 'undefined') module.exports = { P, M, ST, Fridge, Detector, powerState, stepModel, psat, tsat, SAFE_HI, SAFE_LO, WHY };
