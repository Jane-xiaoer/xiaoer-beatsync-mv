#!/usr/bin/env python3
"""分析准确度测试：合成「答案已知」的曲子，拿 analyze 的结果去对答案。

生成的曲子有段落（前奏稀疏 / 主歌 / 副歌加满 / 间奏骤停 / 副歌），覆盖：
  直拍、摇摆、人手误差（±12ms）、速度漂移（±4%）、只有贝斯和铺底（无鼓）的稀疏段。
答案 = 每一拍、每个强拍、每个段落边界的真实时刻（样本级）。
评分：
  · 拍 F 值（MIREX 标准，±70ms 算对）
  · 对上的拍平均差多少毫秒、多少比例在 ±1 帧（30fps = 16.7ms）以内
  · 强拍 F 值、段落边界 ±1 拍命中率
  beatsync 的 analyze 不知道这些答案，答案只存在测试脚本里。
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import mir_eval

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "beatsync"))
import analyze  # noqa: E402

SR = 44100
rng = np.random.default_rng(2026)


def env(n, a=0.002, d=0.2):
    t = np.arange(n) / SR
    return np.minimum(1, t / a) * np.exp(-t / d)


def kick(n=int(0.35 * SR)):
    t = np.arange(n) / SR
    f = 50 + 90 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.12)


def snare(n=int(0.25 * SR)):
    return (rng.standard_normal(n) * 0.6 + np.sin(2 * np.pi * 190 * np.arange(n) / SR) * 0.4) * env(n, 0.001, 0.07)


def hat(n=int(0.06 * SR)):
    x = rng.standard_normal(n)
    x = np.diff(x, prepend=0)
    return x * env(n, 0.0005, 0.015) * 0.5


def tone(freq, dur, amp=0.3, a=0.01, d=0.6):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return amp * np.sin(2 * np.pi * freq * t) * env(n, a, d)


def add(buf, x, t):
    i = int(round(t * SR))
    if i < 0 or i >= len(buf):
        return
    j = min(len(buf), i + len(x))
    buf[i:j] += x[:j - i]


CHORDS = [[220, 277, 330], [196, 247, 294], [175, 220, 262], [196, 247, 311]]
# 段落：名称、小节数、配器
FORM = [("intro", 4, {"pad", "bass"}), ("verse", 8, {"pad", "bass", "kick", "hat"}),
        ("chorus", 8, {"pad", "bass", "kick", "snare", "hat", "lead"}), ("break", 4, {"pad"}),
        ("chorus", 8, {"pad", "bass", "kick", "snare", "hat", "lead"})]


def make_track(bpm, swing=0.0, jitter_ms=0.0, drift=0.0):
    beats_per_bar = 4
    n_bars = sum(b for _, b, _ in FORM)
    n_beats = n_bars * beats_per_bar
    # 每拍时长：可带缓慢漂移
    base = 60 / bpm
    lens = base * (1 + drift * np.sin(np.linspace(0, 2 * np.pi, n_beats)))
    beat_t = 1.0 + np.r_[0, np.cumsum(lens)[:-1]]
    total = beat_t[-1] + 3
    buf = np.zeros(int(total * SR))
    sec_bounds = []
    k = 0
    for name, bars, inst in FORM:
        sec_bounds.append(beat_t[k])
        for bar in range(bars):
            ch = CHORDS[(k // 4) % 4]
            for b in range(4):
                t = beat_t[k] + rng.normal(0, jitter_ms / 1000) if jitter_ms else beat_t[k]
                L = lens[k]
                if "kick" in inst and b in (0, 2):
                    add(buf, kick() * 0.9, t)
                if "snare" in inst and b in (1, 3):
                    add(buf, snare() * 0.5, t)
                if "hat" in inst:
                    add(buf, hat(), t)
                    add(buf, hat() * 0.6, t + L * (0.5 + swing))
                if "bass" in inst and b in (0, 2):
                    add(buf, tone(ch[0] / 2, L * 1.8, 0.35, 0.005, 0.5), t)
                if "lead" in inst and b == 0:
                    add(buf, tone(ch[2] * 2, L * 3, 0.12, 0.01, 0.4), t)
                if "pad" in inst and b == 0:
                    for f in ch:
                        add(buf, tone(f, L * 4, 0.08, 0.15, 3.0), beat_t[k])
                k += 1
    buf /= np.max(np.abs(buf)) + 1e-9
    gt = {"beats": beat_t.tolist(), "downbeats": beat_t[::4].tolist(), "sections": sec_bounds[1:]}
    return buf * 0.9, gt


CASES = []
for bpm in (72, 90, 100, 112, 124, 140):
    CASES += [dict(name=f"{bpm}bpm 直拍", bpm=bpm),
              dict(name=f"{bpm}bpm 人手±12ms", bpm=bpm, jitter_ms=12),
              dict(name=f"{bpm}bpm 摇摆", bpm=bpm, swing=0.16),
              dict(name=f"{bpm}bpm 速度漂移±4%", bpm=bpm, drift=0.04)]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="beatsync-eval-"))
    rows = []
    for c in CASES:
        y, gt = make_track(c["bpm"], c.get("swing", 0), c.get("jitter_ms", 0), c.get("drift", 0))
        wav = tmp / "t.wav"
        sf.write(wav, y, SR)
        res = analyze.analyze(str(wav), fps=30)
        est_b = np.array([b["t"] for b in res["beats"]])
        ref_b = np.array(gt["beats"])
        f = mir_eval.beat.f_measure(ref_b, est_b)          # ±70ms
        m = mir_eval.util.match_events(ref_b, est_b, 0.07)
        err = np.array([est_b[j] - ref_b[i] for i, j in m]) * 1000
        est_d = np.array([b["t"] for b in res["downbeats"]])
        fd = mir_eval.beat.f_measure(np.array(gt["downbeats"]), est_d)
        beat_len = 60 / c["bpm"]
        est_s = [s["start"] for s in res["sections"][1:]]
        hit = sum(1 for g in gt["sections"] if est_s and min(abs(g - e) for e in est_s) <= beat_len * 1.05)
        rows.append({"case": c["name"], "beat_F": round(f, 3), "mean_abs_ms": round(float(np.mean(np.abs(err))), 1) if len(err) else None,
                     "median_ms": round(float(np.median(err)), 1) if len(err) else None,
                     "within_1f": round(float(np.mean(np.abs(err) <= 1000 / 30)), 3) if len(err) else 0,
                     "downbeat_F": round(fd, 3), "sections_hit": f"{hit}/{len(gt['sections'])}",
                     "tempo_est": res["meta"]["tempo"]})
        r = rows[-1]
        print(f"{r['case']:<18} 拍F {r['beat_F']:.3f}  平均差 {r['mean_abs_ms']}ms  ≤1帧 {r['within_1f']:.0%}  "
              f"强拍F {r['downbeat_F']:.3f}  段落 {r['sections_hit']}  测得 {r['tempo_est']}BPM", flush=True)
    keys = ["beat_F", "within_1f", "downbeat_F"]
    summ = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys}
    summ["mean_abs_ms"] = round(float(np.mean([r["mean_abs_ms"] for r in rows if r["mean_abs_ms"] is not None])), 1)
    hs = [tuple(map(int, r["sections_hit"].split("/"))) for r in rows]
    summ["sections_hit"] = round(sum(a for a, _ in hs) / sum(b for _, b in hs), 3)
    print("\n总计", json.dumps(summ, ensure_ascii=False))
    out = Path(__file__).with_name("eval_synth_result.json")
    out.write_text(json.dumps({"summary": summ, "cases": rows}, ensure_ascii=False, indent=1))
    print("✓", out)


if __name__ == "__main__":
    main()
