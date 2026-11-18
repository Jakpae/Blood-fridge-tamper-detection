"""HMAC-SHA256 메시지 인증과 재전송 방지 (전송 구간 담당).

센서 측 장치가 메시지마다 태그를 붙이고, 수신 측이 태그·시퀀스·시각을 확인한다.
HMAC은 '전송 중 변조'와 '메시지 재전송'만 잡는다. 센서가 처음부터 거짓 값을 서명하면 통과하므로,
그 경우는 열 모델·가상 센서(detect.py)가 맡는다.
"""
import hashlib
import hmac
import json
import secrets


def make_key():
    return secrets.token_bytes(32)


def sign(key, sensor_id, seq, ts, values):
    body = json.dumps({"id": sensor_id, "seq": seq, "ts": ts, "v": values},
                      sort_keys=True, separators=(",", ":"))
    tag = hmac.new(key, body.encode(), hashlib.sha256).hexdigest()
    return {"body": body, "tag": tag}


class Receiver:
    def __init__(self, keys, max_age_s=120):
        self.keys = keys                # 센서별로 다른 키
        self.last_seq = {}
        self.max_age_s = max_age_s

    def check(self, msg, now_ts):
        body = json.loads(msg["body"])
        key = self.keys.get(body["id"])
        if key is None:
            return "unknown_sensor"
        good = hmac.new(key, msg["body"].encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(good, msg["tag"]):
            return "auth_fail"
        if body["seq"] <= self.last_seq.get(body["id"], -1):
            return "replay"
        if now_ts - body["ts"] > self.max_age_s:
            return "stale"
        self.last_seq[body["id"]] = body["seq"]
        return "ok"


def self_test(n=2000, seed=0):
    """정상·본문 변조·재전송 메시지를 섞어 판정 정확도를 잰다."""
    import random
    rnd = random.Random(seed)
    key = make_key()
    rx = Receiver({"T1": key})
    sent, results = [], {"normal": [0, 0], "tamper": [0, 0], "replay": [0, 0]}
    for i in range(n):
        ts = 1_000_000 + 60 * i
        msg = sign(key, "T1", i, ts, {"T": round(4 + rnd.gauss(0, 0.5), 2)})
        kind = "normal"
        r = rnd.random()
        if r < 0.03:
            b = json.loads(msg["body"]); b["v"]["T"] = 4.0
            msg = {"body": json.dumps(b, sort_keys=True, separators=(",", ":")), "tag": msg["tag"]}
            kind = "tamper"
        elif r < 0.06 and sent:
            msg = rnd.choice(sent)
            kind = "replay"
        verdict = rx.check(msg, ts)
        caught = verdict != "ok"
        results[kind][0] += 1
        results[kind][1] += caught
        if kind == "normal":
            sent.append(msg)
    return results
