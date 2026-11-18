const C = require('./core.js');
const base = { thMu: 0, thSd: 1, thThr: 1e9, vMu: 0, vSd: 1, vThr: 1e9 };
function run(seed, days, cfg, sc) {
  const f = new C.Fridge(seed), d = new C.Detector(cfg); const rows = [], buf = []; let frozen = null;
  for (let i = 0; i < days * 1440; i++) {
    if (sc && i === sc.start) Object.assign(f.ev, sc.ev || {});
    const r = f.step(); buf.push(r); const s = { ...r };
    if (sc && i >= sc.start) {
      if (!frozen) frozen = { T: buf[i - 1].Trep, P: buf[i - 1].Prep };
      if (sc.T === 'offset') s.Trep -= 1.5;
      if (sc.T === 'freeze') s.Trep = frozen.T + (Math.random() - 0.5) * 0.08;
      if (sc.T === 'replay') s.Trep = buf[i - 1440].Trep;
      if (sc.P === 'freeze') s.Prep = frozen.P + (Math.random() - 0.5) * 0.02;
      if (sc.P === 'replay') s.Prep = buf[i - 1440].Prep;
    }
    rows.push({ r, s, o: d.push(s) });
  }
  return rows;
}
const med = a => { const b = [...a].sort((x, y) => x - y); return b[Math.floor(b.length / 2)]; };
const mad = a => { const m = med(a); return 1.4826 * med(a.map(x => Math.abs(x - m))); };
let th = [], v = [];
for (const s of [1, 2, 3, 4]) for (const x of run(s, 5, base).slice(100)) { if (x.o.pred !== undefined) th.push(x.r.Trep - x.o.pred); if (x.o.virt != null) v.push(x.r.Trep - x.o.virt); }
const cfg = { thMu: med(th), thSd: Math.max(mad(th), 0.2), vMu: med(v), vSd: mad(v) };   // 열 모델 표준화 단위 하한 0.2°C (실험과 같음)
cfg.thThr = 1e9; cfg.vThr = 1e9;
let mTh = 0, mV = 0;
for (const s of [11, 12, 13, 14, 15, 16, 17, 18]) for (const x of run(s, 3, cfg).slice(100)) { mTh = Math.max(mTh, x.o.sTh); mV = Math.max(mV, x.o.sV); }
cfg.thThr = Math.max(mTh * 1.2, 3); cfg.vThr = Math.max(mV * 1.2, 5);
console.log(JSON.stringify(cfg));
require('fs').writeFileSync(__dirname + '/cfg.json', JSON.stringify(cfg));
let fa = 0, n = 0; for (let s = 21; s < 41; s++) { const rr = run(s, 4, cfg); n++; if (rr.some(x => x.o.alarm)) fa++; }
console.log(`정상 4일 x ${n}회 중 경보 ${fa}회`);
const cases = [
  ['온도 1.5 낮게', { T: 'offset' }],
  ['히터+온도멈춤+압력멈춤', { ev: { heaterOn: true }, T: 'freeze', P: 'freeze' }],
  ['히터+어제기록(T,P)', { ev: { heaterOn: true }, T: 'replay', P: 'replay' }],
  ['냉각끔+어제기록(T,P)', { ev: { compOff: true }, T: 'replay', P: 'replay' }],
  ['히터+문안닫힘+어제기록', { ev: { heaterOn: true, doorStuck: true }, T: 'replay', P: 'replay' }],
  ['냉각끔+온도만 어제기록', { ev: { compOff: true }, T: 'replay' }],
  ['히터만(위조 없음)', { ev: { heaterOn: true } }],
  ['문 안 닫힘만', { ev: { doorStuck: true } }],
  ['정전만', { ev: { powerCut: true } }],
  ['냉각 계속만', { ev: { compOn: true } }],
];
for (const [name, sc] of cases) {
  const res = [];
  for (const s of [31, 32, 33, 34, 35]) {
    const start = 1440 + 600 + s * 7;
    const rr = run(s, 2, cfg, { ...sc, start, ev: { ...(sc.ev || {}) } });
    const ia = rr.findIndex((x, i) => i >= start && x.o.alarm), il = rr.findIndex((x, i) => i >= start && x.o.limit);
    const a = ia > 0 ? rr[rr.length - 1].o.alarm : null;
    const over = rr.findIndex((x, i) => i >= start && (x.r.Ttrue > 6 || x.r.Ttrue < 1));
    res.push(a ? `${a.t - start}분 ${a.why}/${a.cause}` : '없음' ); res[res.length-1] += ` 기준이탈경보${il > 0 ? il - start : '-'} 실제이탈${over > 0 ? over - start : '-'}`;
    if (ia > 0 && ia < start) res[res.length - 1] = '공격전 오경보';
  }
  console.log(name.padEnd(16), '|', res.join(' ; '));
}
