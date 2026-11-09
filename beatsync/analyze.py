#!/usr/bin/env python3
"""小耳音乐卡点MV · 第 1 步：听歌 —— 把一首歌拆成「拍 / 小节 / 乐句 / 段落 / 能量 / 气口」，
每个点都带置信度，并对齐到视频帧。

  beatsync analyze song.mp3 [-o cues.json] [--fps 30] [--meter 4]

输出 cues.json（时间单位：秒；frame 按 --fps 取整到最近一帧）：
  meta      时长、速度、fps、拍号
  beats     每一拍：时刻、帧、置信度、属于第几小节、小节里第几拍
  downbeats 每小节第一拍（强拍）
  phrases   乐句起点（默认 4 小节一句，按结构变化挑相位）
  sections  段落：起止、平均能量、能量档（安静/中/高/最高）
  hits      重音（全曲最突出的起音，不一定在拍上）
  breaths   气口（明显的静音或骤停）
  energy    10Hz 的能量曲线（0~1）

做法上的几个关键（来自小耳FM 1082 首歌的实测）：
  · 卡点用整轨的「起音」，不用分轨能量 —— 分轨能量曲线会晚 60~75ms（30fps 下 2 帧）。
  · 每个点在 ±25ms 内回到波形上找真正的起音沿，再取整到帧。
  · 每个点都给置信度；拿不准的点（如自由速度的慢曲）不当硬卡点。
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import librosa

import pdoom_core
import stems as stems_mod

SR = 22050
HOP = 256          # 11.6ms 一帧，比视频帧细 3 倍
FINE_HOP = 64      # 回波形找起音沿时用 2.9ms


def r(x, n=3):
    return round(float(x), n)


def norm(x, pct=95):
    x = np.asarray(x, float)
    s = np.percentile(x[x > 0], pct) if np.any(x > 0) else 1.0
    return np.clip(x / max(s, 1e-9), 0, 1.5)


def smooth(x, k):
    k = int(max(1, k))
    if k < 2:
        return np.asarray(x, float)
    return np.convolve(x, np.ones(k) / k, mode="same")


def refine_attack(t, fine_env, fine_fps, win=0.025):
    """在 t±win 内找能量上升最陡的位置（真正的起音沿）。"""
    i0 = max(0, int((t - win) * fine_fps))
    i1 = min(len(fine_env) - 1, int((t + win) * fine_fps))
    if i1 <= i0 + 1:
        return t
    seg = fine_env[i0:i1]
    return (i0 + int(np.argmax(seg))) / fine_fps


def _wmax(curve, i, r=2):
    return float(curve[max(0, i - r):min(len(curve), i + r + 1)].max())


def track_beats(oenv, sr, oenv_low):
    """拍子追踪，带两道修正：
    1) 速度可能在变：默认用固定速度；只有「变速版」的拍点明显更贴起音时才换它；
    2) 倍速错误：测出 ≥142 BPM 时（MV 常见的歌在 70～140），只看中低频（鼓、贝斯、和弦，不看踩镲）在单数拍和双数拍上的强弱 ——
       一边明显弱，说明把八分音符当成了拍，砍掉弱的那一半，速度减半。
    """
    t0, b0 = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=HOP, trim=False)
    tempo, bfr = float(np.atleast_1d(t0)[0]), np.asarray(b0)

    def align(b):
        # 拍点落在起音峰 ±2 帧（±23ms）内的比例
        if len(b) < 4:
            return 0.0
        pk = librosa.util.peak_pick(oenv, pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=0.2, wait=3)
        if not len(pk):
            return 0.0
        return float(np.mean([np.min(np.abs(pk - i)) <= 2 for i in b]))
    try:
        curve = librosa.feature.tempo(onset_envelope=oenv, sr=sr, hop_length=HOP, aggregate=None, std_bpm=4)
        t1, b1 = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=HOP, bpm=curve, trim=False)
        b1 = np.asarray(b1)
        if 0.8 < len(b1) / max(1, len(bfr)) < 1.25 and align(b1) > align(bfr) + 0.10:
            tempo, bfr = float(np.median(curve)), b1
    except Exception:
        pass
    if tempo >= 142 and len(bfr) >= 8:
        s = np.array([_wmax(oenv_low, i) for i in bfr])
        even, odd = float(np.mean(s[0::2])), float(np.mean(s[1::2]))
        if max(even, odd) / max(min(even, odd), 1e-9) > 1.25:
            bfr = bfr[0::2] if even >= odd else bfr[1::2]
            tempo /= 2
    return tempo, bfr


def checkerboard_novelty(feat, k):
    """结构变化曲线（Foote 1999）：段落/乐句换了，曲线就冒峰。"""
    feat = librosa.util.normalize(feat, axis=0)
    S = feat.T @ feat
    n = S.shape[0]
    g = np.outer(np.hanning(2 * k), np.hanning(2 * k))
    sign = np.ones((2 * k, 2 * k))
    sign[:k, k:] = -1
    sign[k:, :k] = -1
    ker = g * sign
    pad = np.pad(S, k, mode="constant")
    nov = np.array([np.sum(pad[i:i + 2 * k, i:i + 2 * k] * ker) for i in range(n)])
    nov[nov < 0] = 0
    # 两端补零会造成假峰：前后各 k 列按距离压低
    ramp = np.clip(np.arange(n) / k, 0, 1) * np.clip((n - 1 - np.arange(n)) / k, 0, 1)
    nov = nov * ramp
    return norm(nov, 99)


def analyze(path, fps=30.0, meter=4, phrase_bars=4, stems_dir=None):
    y, sr = librosa.load(path, sr=SR, mono=True)
    dur = len(y) / sr
    ffps = sr / HOP
    ST = load_stems(path, stems_dir, len(y)) if stems_dir else None

    # ---- 起音 & 拍 ----
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, aggregate=np.median)
    _mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, n_mels=64, fmax=8000)
    oenv_low = librosa.onset.onset_strength(S=librosa.power_to_db(_mel[:24]), sr=sr, hop_length=HOP)
    grid_info = None
    if ST and ST["has_drums"]:
        # 有鼓轨：拍子跟鼓走（干净得多）；整轨起音只作补充
        od = librosa.onset.onset_strength(y=ST["sr22"]["drums"], sr=sr, hop_length=HOP, aggregate=np.median)
        n_ = min(len(od), len(oenv))
        oenv = od[:n_] / (np.percentile(od, 99) + 1e-9) + 0.5 * oenv[:n_] / (np.percentile(oenv, 99) + 1e-9)
        lowmix = ST["sr22"]["bass"] + ST["sr22"]["drums"]
        oenv_low = librosa.onset.onset_strength(y=lowmix, sr=sr, hop_length=HOP)[:n_]
    tempo, bfr = track_beats(oenv, sr, oenv_low)
    beat_t = librosa.frames_to_time(bfr, sr=sr, hop_length=HOP)
    if ST and ST["has_drums"]:
        # pdoom：恒定速度网格，粗搜 0.02BPM/4ms、细搜 0.001BPM/1ms，再用底鼓起音沿修相位
        bpm, P, off, res = pdoom_core.fit_grid(ST["drums"], ST["mix"], 44100, dur, (tempo - 4, tempo + 4))
        grid_t = off + P * np.arange(int((dur - off) / P) + 1)
        dyn = beat_t
        dev = np.array([np.min(np.abs(dyn - g)) for g in grid_t]) if len(dyn) else np.array([1.0])
        p95 = float(np.percentile(dev, 95)) if len(dev) else 1.0

        # 谁更贴鼓：底鼓 + 军鼓起音落在拍（或半拍）±35ms 内的比例
        drum_t = np.r_[ST["on"]["kick"][:, 0], ST["on"]["snare"][:, 0]] if ST["has_drums"] else np.array([])

        def hit_rate(bt):
            if len(drum_t) < 8 or len(bt) < 4:
                return 0.0
            half = np.sort(np.r_[bt, (bt[:-1] + bt[1:]) / 2])
            return float(np.mean([np.min(np.abs(half - t)) <= 0.035 for t in drum_t]))
        hr_grid, hr_dyn = hit_rate(grid_t), hit_rate(dyn)
        sd = float(np.std(res) * 1000) if len(res) else 99.0
        use_grid = sd <= 20 and hr_grid >= hr_dyn - 0.02
        grid_info = {"mode": "grid" if use_grid else "dynamic", "grid_bpm": round(float(bpm), 3),
                     "grid_vs_dynamic_p95_ms": round(p95 * 1000, 1),
                     "drum_hit_rate_grid": round(hr_grid, 3), "drum_hit_rate_dynamic": round(hr_dyn, 3),
                     "kick_residual_sd_ms": round(float(np.std(res) * 1000), 1) if len(res) else None}
        if use_grid:
            tempo = float(bpm)
            beat_t = grid_t[grid_t < dur - 0.05]
            bfr = librosa.time_to_frames(beat_t, sr=sr, hop_length=HOP)

    on_fr = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=HOP, backtrack=False)
    on_t = librosa.frames_to_time(on_fr, sr=sr, hop_length=HOP)
    on_s = oenv[on_fr] if len(on_fr) else np.array([])

    # 细粒度能量导数：用来把点「吸」到真正的起音沿上
    fine_rms = librosa.feature.rms(y=y, frame_length=256, hop_length=FINE_HOP)[0]
    fine_env = np.maximum(0, np.diff(fine_rms, prepend=fine_rms[0]))
    fine_fps = sr / FINE_HOP

    # 拍的置信度：拍点附近起音强度 / 局部中位数，再看它离最近起音多远
    loc_med = smooth(oenv, int(ffps * 2)) + 1e-9
    beats = []
    for t in beat_t:
        i = int(round(t * ffps))
        w = oenv[max(0, i - 3):i + 4]
        peak = float(w.max()) if len(w) else 0.0
        strength = peak / float(loc_med[min(i, len(loc_med) - 1)])
        near = np.min(np.abs(on_t - t)) if len(on_t) else 1.0
        conf = np.clip((strength - 0.8) / 2.0, 0, 1) * np.clip(1 - near / 0.07, 0, 1)
        t2 = t if (grid_info and grid_info["mode"] == "grid") else (refine_attack(t, fine_env, fine_fps) if near < 0.05 else t)
        beats.append({"t": t2, "conf": float(conf), "raw": float(t)})

    if ST:
        # 分轨证据：拍上真有底鼓 / 军鼓 / 贝斯 / 旋律起音 → 置信度按那个起音的强度抬上去
        pool = [ST["on"][k] for k in ("kick", "snare", "bass", "other") if k in ST["on"] and len(ST["on"][k])]
        allon = np.vstack(pool) if pool else np.zeros((0, 2))
        gridm = bool(grid_info and grid_info["mode"] == "grid")
        for b in beats:
            if len(allon):
                d = np.abs(allon[:, 0] - b["t"])
                m = d <= 0.035
                sv = float(allon[m, 1].max()) if m.any() else 0.0
                b["conf"] = max(b["conf"], (0.35 + 0.65 * sv) if (gridm and sv > 0) else sv)

    # 速度稳不稳：拍间隔的离散度。自由速度的慢曲这里会很大 → 全体降置信度
    ibi = np.diff([b["t"] for b in beats]) if len(beats) > 2 else np.array([1.0])
    steadiness = float(np.clip(1 - np.std(ibi) / max(np.mean(ibi), 1e-6) * 4, 0, 1))
    for b in beats:
        b["conf"] *= 0.4 + 0.6 * steadiness

    # ---- 强拍（小节线）：低频起音 + 和声变化，选 4 个相位里最像的 ----
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, n_mels=64, fmax=8000)
    low = librosa.onset.onset_strength(S=librosa.power_to_db(mel[:8]), sr=sr, hop_length=HOP)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    chroma_nov = np.r_[0, np.sum(np.abs(np.diff(smooth_cols(chroma, 4), axis=1)), axis=0)]
    def wmax(curve, i):
        return float(curve[max(0, i - 2):min(len(curve), i + 3)].max())
    bidx = [int(round(t * ffps)) for t in beat_t]
    lb = np.array([wmax(low, i) for i in bidx])
    cb = np.array([wmax(chroma_nov, i) for i in bidx])
    score_b = lb / (np.mean(lb) + 1e-9) + 0.7 * cb / (np.mean(cb) + 1e-9)
    if ST and ST["has_drums"]:
        def near_strength(ons, t, tol=0.05):
            if not len(ons):
                return 0.0
            d = np.abs(ons[:, 0] - t)
            j = int(np.argmin(d))
            return float(ons[j, 1]) if d[j] <= tol else 0.0
        K, S_, B = ST["on"]["kick"], ST["on"]["snare"], ST["on"]["bass"]
        kb = np.array([near_strength(K, t) for t in beat_t])
        sb = np.array([near_strength(S_, t) for t in beat_t])
        bb = np.array([near_strength(B, t) for t in beat_t])
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
        score_b = z(kb) + z(bb) + z(cb) - z(sb)      # pdoom：底鼓 + 贝斯 + 和声变化 − 军鼓（军鼓在 2、4 拍）
        score_b = score_b - score_b.min()
    phase_scores = [float(np.mean(score_b[p::meter])) if len(score_b[p::meter]) else 0 for p in range(meter)]
    order = np.argsort(phase_scores)[::-1]
    phase = int(order[0])
    db_conf = float(np.clip((phase_scores[order[0]] - phase_scores[order[1]]) /
                            max(phase_scores[order[0]], 1e-6) * 6, 0, 1))
    for k, b in enumerate(beats):
        pos = (k - phase) % meter
        b["beat_in_bar"] = pos + 1
        b["bar"] = (k - phase) // meter + (1 if phase == 0 else 2)
    downbeats = [b for b in beats if b["beat_in_bar"] == 1]

    # ---- 结构：乐句 & 段落 ----
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
    # 降采样到「每拍一列」再算，快且稳
    sync = librosa.util.sync(np.vstack([librosa.util.normalize(chroma), librosa.util.normalize(mfcc)]),
                             bfr, aggregate=np.median)
    nov_s = checkerboard_novelty(sync, k=max(4, meter * 4))   # 大窗：段落
    nov_p = checkerboard_novelty(sync, k=max(2, meter * 2))   # 小窗：乐句

    def beat_nov(nov, k):
        # sync 的第 0 列是第一拍之前那段，所以第 k 拍对应 k+1
        return float(nov[min(k + 1, len(nov) - 1)])

    db_idx = [beats.index(d) for d in downbeats]

    # 段落：在每个强拍上打分（大窗结构变化 + 前后两小节能量台阶），挑最突出的几个
    rms0 = librosa.feature.rms(y=y, hop_length=HOP)[0]
    e_beat = librosa.util.sync(smooth(rms0, ffps * 0.5)[None, :], bfr, aggregate=np.mean)[0]
    e_beat = e_beat / (e_beat.max() + 1e-9)
    w = 2 * meter

    def estep(k):
        i = k + 1
        a = e_beat[max(0, i - w):i]
        b = e_beat[i:i + w]
        return abs(float(np.mean(b) - np.mean(a))) * 3 if len(a) and len(b) else 0.0
    scored = [(beat_nov(nov_s, k) + estep(k), k) for k in db_idx]
    cand = sorted(scored, reverse=True)
    want = int(np.clip(round(dur / 18), 2, 12))
    top = cand[0][0] if cand else 0
    min_gap = phrase_bars * meter          # 段落至少一句长
    chosen = []
    for sc, k in cand:
        if len(chosen) >= want or sc < max(0.15, 0.35 * top):
            break
        if all(abs(k - c) >= min_gap for c in chosen) and k >= meter and len(beats) - k >= meter * 2:
            chosen.append(k)
    chosen.sort()
    bounds = sorted(set([0.0] + [beats[k]["t"] for k in chosen] + [dur]))

    # 乐句：从每个段落的第一个强拍起，每 phrase_bars 小节一句（乐句跟着段落对齐，而不是全曲一把尺子）
    starts = ([db_idx[0]] if db_idx else []) + chosen
    phrase_idx = []
    for a_i, k0 in enumerate(starts):
        k1 = starts[a_i + 1] if a_i + 1 < len(starts) else len(beats)
        j0 = db_idx.index(k0)
        for j in range(j0, len(db_idx), phrase_bars):
            if db_idx[j] >= k1:
                break
            phrase_idx.append(db_idx[j])
    phrase_idx = sorted(set(phrase_idx))
    phrases = [{"t": beats[k]["t"], "bar": beats[k]["bar"],
                "conf": float(min(1.0, 0.3 + beat_nov(nov_p, k)) * (0.4 + 0.6 * db_conf))}
               for k in phrase_idx]

    # ---- 能量 ----
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    e = norm(smooth(rms, ffps * 1.0), 98).clip(0, 1)
    sections = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = e[int(a * ffps):max(int(a * ffps) + 1, int(b * ffps))]
        sections.append({"start": a, "end": b, "energy": float(np.mean(seg)),
                         "trend": float(np.polyfit(np.arange(len(seg)), seg, 1)[0] * ffps * (b - a)) if len(seg) > 3 else 0.0})
    if sections:
        lo = min(s["energy"] for s in sections)
        hi = max(s["energy"] for s in sections)
        for s in sections:
            rel = (s["energy"] - lo) / max(hi - lo, 0.08)
            s["level"] = "高" if rel >= 0.75 else ("中" if rel >= 0.4 else "安静")
        # 「最高」只给一段：能量最高且足够长的那段，作为全片高潮
        long_ = [s for s in sections if s["end"] - s["start"] >= 8] or sections
        max(long_, key=lambda s: s["energy"])["level"] = "最高"

    # ---- 重音：全曲最突出的起音（前 8%），且明显高于周围 ----
    hits = []
    if len(on_s):
        thr = np.percentile(on_s, 92)
        for t, s in zip(on_t, on_s):
            i = int(t * ffps)
            if s >= thr and s > 2.2 * loc_med[min(i, len(loc_med) - 1)]:
                hits.append({"t": refine_attack(t, fine_env, fine_fps),
                             "conf": float(np.clip(s / (np.max(on_s) + 1e-9), 0, 1))})

    # ---- 气口：整轨音量掉到很低并持续 ≥0.25s ----
    db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max(rms))
    quiet = db < -32
    breaths, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            a, b = i / ffps, j / ffps
            if b - a >= 0.25 and a > 0.5 and b < dur - 0.5:
                breaths.append({"start": a, "end": b})
            i = j
        else:
            i += 1

    def F(t):
        return int(round(t * fps))

    out = {
        "meta": {"source": str(Path(path).name), "duration": r(dur), "tempo": r(tempo, 1),
                 "fps": fps, "meter": meter, "phrase_bars": phrase_bars,
                 "steadiness": r(steadiness), "downbeat_conf": r(db_conf),
                 "note": "frame = round(t*fps)；conf 0~1，<0.35 的点不建议当硬卡点"},
        "beats": [{"t": r(b["t"]), "frame": F(b["t"]), "conf": r(b["conf"], 2),
                   "bar": b["bar"], "beat": b["beat_in_bar"]} for b in beats],
        "downbeats": [{"t": r(b["t"]), "frame": F(b["t"]), "bar": b["bar"], "conf": r(b["conf"], 2)}
                      for b in downbeats],
        "phrases": [{"t": r(p["t"]), "frame": F(p["t"]), "bar": p["bar"], "conf": r(p["conf"], 2)}
                    for p in phrases],
        "sections": [{"start": r(s["start"]), "end": r(s["end"]), "frame": F(s["start"]),
                      "energy": r(s["energy"], 2), "trend": r(s["trend"], 2), "level": s["level"]}
                     for s in sections],
        "hits": [{"t": r(h["t"]), "frame": F(h["t"]), "conf": r(h["conf"], 2)} for h in hits],
        "breaths": [{"start": r(b["start"]), "end": r(b["end"])} for b in breaths],
        "energy": {"rate": 10, "values": [r(v, 2) for v in e[::max(1, int(round(ffps / 10)))]]},
    }
    if grid_info:
        out["meta"]["beat_mode"] = grid_info
    if ST:
        out["stems"] = stem_layer(ST, dur, fps)
        out["meta"]["stems"] = ST["names"]
    return out


# ---------------- 分轨层 ----------------
# 凯尔经公式（小耳FM 实验 7 定稿）：画面推进 = 底速 0.06 + 整体响度^1.5 + Σ 各轨起音推力
#   推力 = 权重 × 起音强度 × e^(−Δt/τ)；旋律 1.0/0.22s、鼓 1.4/0.16s、人声 0.6/0.3s、贝斯 0.7/0.4s
#   所有轨统一按全局最大值归一化，免得某一轨的杂音被放大成强推力。
KELLS = {"other": (1.0, 0.22), "drums": (1.4, 0.16), "vocals": (0.6, 0.30), "bass": (0.7, 0.40),
         "piano": (1.0, 0.22), "guitar": (1.0, 0.22)}


def load_stems(path, stems_dir, n22):
    meta_p = Path(stems_dir) / "stems.json"
    if not meta_p.exists():
        return None
    import json as _j
    meta = _j.loads(meta_p.read_text())
    mix44, _ = librosa.load(path, sr=44100, mono=True)
    n44 = len(mix44)
    S = {"names": meta["stems"], "mix": mix44, "sr22": {}, "on": {}}
    for n in meta["stems"]:
        y44 = stems_mod.load(stems_dir, n, n44)
        S[n] = y44
        S["sr22"][n] = librosa.resample(y44, orig_sr=44100, target_sr=SR)[:n22]
    S["has_drums"] = float(np.sqrt(np.mean(S["drums"] ** 2))) > 0.01 * float(np.sqrt(np.mean(mix44 ** 2)) + 1e-9) * 10
    # 起音（pdoom 带通 dB 包络法：最陡上升处为起音沿，强度按 5/95 分位映射到 0.2..1）
    if S["has_drums"]:
        (kt, kdb), (st, sdb), (ht, hdb), _ = pdoom_core.drum_onsets(S["drums"], 44100, 0.5, 0.0)
        S["on"]["kick"] = np.c_[kt, pdoom_core.strength01(kdb)] if len(kt) else np.zeros((0, 2))
        S["on"]["snare"] = np.c_[st, pdoom_core.strength01(sdb)] if len(st) else np.zeros((0, 2))
        S["on"]["hat"] = np.c_[ht, pdoom_core.strength01(hdb)] if len(ht) else np.zeros((0, 2))
    else:
        for k in ("kick", "snare", "hat"):
            S["on"][k] = np.zeros((0, 2))
    spec = {"bass": (None, 300, 0.015, 0.12, 10), "vocals": (200, 4000, 0.010, 0.09, 9),
            "other": (150, 6000, 0.010, 0.08, 9), "piano": (100, 5000, 0.010, 0.08, 9),
            "guitar": (100, 5000, 0.010, 0.08, 9)}
    for n, (lo, hi, w, gap, rdb) in spec.items():
        if n in S:
            t, db = pdoom_core.band_onsets(S[n], 44100, lo, hi, win=w, min_gap=gap, rel_db=rdb)
            # 太安静的轨（如纯音乐的人声轨只剩串音）不算
            env = pdoom_core.frame_rms(S[n], 44100)
            act = env > 0.1 * np.percentile(env, 99)
            keep = [i for i, x in enumerate(t) if act[min(len(act) - 1, int(x * 100))]]
            S["on"][n] = np.c_[t[keep], pdoom_core.strength01(db[keep])] if keep else np.zeros((0, 2))
    return S


def stem_layer(S, dur, fps):
    out = {"names": S["names"], "onsets": {}, "env": {"rate": 100}, "lines": [], "entries": []}
    for k, arr in S["on"].items():
        out["onsets"][k] = [[r(t), r(s, 2), int(round(t * fps))] for t, s in arr]
    rms_all = {}
    for n in S["names"]:
        e = pdoom_core.frame_rms(S[n], 44100)
        rms_all[n] = e
    gmax = max(float(np.percentile(e, 99)) for e in rms_all.values()) + 1e-12
    for n, e in rms_all.items():
        out["env"][n] = [r(v, 2) for v in np.clip(pdoom_core.smooth_env(e) / gmax, 0, 1)]
    # 人声的句子：人声轨持续有声的段（间隙 <0.35s 合并，≥0.6s 才算一句）
    if "vocals" in rms_all:
        v = rms_all["vocals"]
        on = v > 0.12 * np.percentile(v, 99)
        segs, i = [], 0
        while i < len(on):
            if on[i]:
                j = i
                while j < len(on) and on[j]:
                    j += 1
                segs.append([i / 100, j / 100])
                i = j
            else:
                i += 1
        merged = []
        for a, b in segs:
            if merged and a - merged[-1][1] < 0.35:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        out["lines"] = [{"start": r(a), "end": r(b), "frame": int(round(a * fps))} for a, b in merged if b - a >= 0.6]
    # 乐器进出：每轨 静↔响 跨越（带 1.2s 驻留防抖）—— 小耳FM build_show 的做法
    for n, e in rms_all.items():
        x = np.convolve(e / gmax, np.ones(200) / 200, mode="same")     # 2 秒平滑
        active, last = False, -400
        for i, val in enumerate(x):
            if not active and val > 0.20 and i - last > 400:
                active, last = True, i
                out["entries"].append({"t": r(i / 100), "frame": int(round(i / 100 * fps)), "stem": n, "type": "进"})
            elif active and val < 0.05 and i - last > 400:
                active, last = False, i
                out["entries"].append({"t": r(i / 100), "frame": int(round(i / 100 * fps)), "stem": n, "type": "出"})
    out["entries"].sort(key=lambda d: d["t"])
    # 凯尔经推力曲线（每视频帧一个值）
    nF = int(round(dur * fps)) + 1
    tt = np.arange(nF) / fps
    mixe = pdoom_core.frame_rms(S["mix"], 44100)
    loud = np.interp(tt, np.arange(len(mixe)) / 100, np.clip(mixe / (np.percentile(mixe, 99) + 1e-12), 0, 1))
    push = 0.06 + loud ** 1.5
    contrib = {}
    groups = {"drums": ["kick", "snare"], "bass": ["bass"], "vocals": ["vocals"], "other": ["other"],
              "piano": ["piano"], "guitar": ["guitar"]}
    for stem, keys in groups.items():
        if stem not in S["names"]:
            continue
        w, tau = KELLS[stem]
        c = np.zeros(nF)
        for k in keys:
            for t0, st in S["on"].get(k, np.zeros((0, 2))):
                i0 = int(np.ceil(t0 * fps - 1e-6))
                if i0 >= nF:
                    continue
                d = tt[i0:] - t0
                m = d < tau * 5
                c[i0:i0 + int(m.sum())] += w * st * np.exp(-d[m] / tau)
        contrib[stem] = c
        push = push + c
    out["push"] = {"fps": fps, "values": [r(v, 3) for v in push],
                   "note": "凯尔经公式：0.06 + 响度^1.5 + Σ 权重×起音强度×e^(−Δt/τ)"}
    return out


def smooth_cols(m, k):
    return np.vstack([smooth(row, k) for row in m])


def summary(c):
    m = c["meta"]
    lines = [f"♪ {m['source']}  {m['duration']}s  {m['tempo']} BPM  {m['meter']}/4  "
             f"速度稳定度 {m['steadiness']}  强拍把握 {m['downbeat_conf']}",
             f"  拍 {len(c['beats'])} · 小节 {len(c['downbeats'])} · 乐句 {len(c['phrases'])} · "
             f"重音 {len(c['hits'])} · 气口 {len(c['breaths'])}",
             "  段落："]
    for s in c["sections"]:
        lines.append(f"    {s['start']:7.2f}–{s['end']:7.2f}s  能量 {s['energy']:.2f}  {s['level']}")
    bm = c["meta"].get("beat_mode")
    if bm:
        lines.append(f"  拍子：{'pdoom 恒定网格' if bm['mode'] == 'grid' else '逐拍跟踪（速度在变）'} · 网格 {bm['grid_bpm']} BPM · "
                     f"鼓点落拍率 网格 {bm['drum_hit_rate_grid']:.0%} / 逐拍 {bm['drum_hit_rate_dynamic']:.0%} · 底鼓残差 {bm['kick_residual_sd_ms']}ms")
    if "stems" in c:
        o = c["stems"]["onsets"]
        lines.append("  分轨起音：" + " · ".join(f"{ {'kick': '底鼓', 'snare': '军鼓', 'hat': '踩镲', 'bass': '贝斯', 'vocals': '人声', 'other': '旋律', 'piano': '钢琴', 'guitar': '吉他'}.get(k, k)} {len(v)}"
                                             for k, v in o.items()))
        lines.append(f"  人声句子 {len(c['stems']['lines'])} · 乐器进出 {len(c['stems']['entries'])}")
    low = sum(1 for b in c["beats"] if b["conf"] < 0.35)
    if low:
        lines.append(f"  ⚠ {low}/{len(c['beats'])} 拍置信度 < 0.35，这些点不当硬卡点")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="小耳音乐卡点MV · 听歌")
    ap.add_argument("audio")
    ap.add_argument("-o", "--out")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--meter", type=int, default=4, help="每小节几拍（默认 4）")
    ap.add_argument("--phrase-bars", type=int, default=4, help="几小节一句（默认 4）")
    ap.add_argument("--stems", help="分轨目录（默认自动分轨到 <输出>.stems/）")
    ap.add_argument("--six", action="store_true", help="分 6 轨（多拆吉他、钢琴）；默认 4 轨")
    ap.add_argument("--no-stems", action="store_true", help="不分轨，只听整轨（快，但卡点单薄）")
    a = ap.parse_args(argv)
    out = Path(a.out or Path(a.audio).with_suffix(".cues.json"))
    sd = None
    if not a.no_stems:
        sd = a.stems or str(out.with_suffix("")) + ".stems"
        stems_mod.separate(a.audio, sd, a.six)
    c = analyze(a.audio, a.fps, a.meter, a.phrase_bars, sd)
    out.write_text(json.dumps(c, ensure_ascii=False, indent=1))
    print(summary(c))
    print(f"✓ {out}")


if __name__ == "__main__":
    main()
