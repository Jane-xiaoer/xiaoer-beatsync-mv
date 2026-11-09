#!/usr/bin/env python3
"""小耳音乐卡点MV · 生长模式（凯尔经做法）—— 用「锚点 + 片段」素材库，按乐句排路线，按音符推播放头。

  beatsync grow song.cues.json 素材库目录/ [-o mv.mp4] [--seed 7] [--preview] [--jobs 3]

素材库 = 一个文件夹，里面一份 catalog.json + clips/*.mp4（首尾帧图生视频做好的片段）：
  loop   A → A   呼吸：几乎不动，只有风和光            → 放在安静的乐句
  grow   空 → A  生长：从无到有长出来                  → 放在中等能量的乐句（先倒放退回 40%，再正放长满）
  morph  A → B   变形：一个场景变成另一个              → 放在最响的乐句，顺着锚点图走到下一个场景
画面不是被「剪」出来的：每一段片子的播放进度由音乐决定 ——
  播放头速度 = 0.06 + 0.8·响度^1.5 + Σ 各轨起音的衰减脉冲（旋律 1.0/0.22s、鼓 1.4/0.16s、人声 0.6/0.3s、贝斯 0.7/0.4s）
  在每段里积分、归一，保证乐句结束时片子正好播完；没有音符的地方画面停住（气口），音符一来画面往前涌。
段与段之间交叠溶解 0.4s。路线由种子决定，同一首歌每次一样，换种子换走法。

这套做法来自小耳的凯尔经 MV 和小耳FM 生长素材库（groove.growlib.js / route.py / retime.py），此处是离线出片版，
音乐分析换成「Demucs 分轨 + pdoom 卡点」。
"""
import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import cv2

HZ = 240
GAIN = {"mel": (1.0, 0.22), "drums": (1.4, 0.16), "vocals": (0.6, 0.30), "bass": (0.7, 0.40)}
XF = 0.4
TYPE_CN = {"loop": "呼吸", "grow": "生长", "morph": "变形"}


def rng(seed):
    """mulberry32，和 groove.growlib.js 同一个随机数，同种子同路线。"""
    a = [(seed & 0xFFFFFFFF) or 1]

    def nxt():
        a[0] = (a[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = a[0]
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return nxt


def music(cues, audio):
    """① 音乐：100Hz 响度（0..1）、强拍、四轨起音 [t, 强度]。"""
    import librosa
    import pdoom_core
    y, sr = librosa.load(audio, sr=44100, mono=True)
    rms = pdoom_core.frame_rms(y, sr)
    rms = rms / (rms.max() + 1e-12)
    st = cues.get("stems", {}).get("onsets", {})
    notes = {"drums": sorted([o[:2] for o in st.get("kick", [])] + [o[:2] for o in st.get("snare", [])]),
             "bass": [o[:2] for o in st.get("bass", [])],
             "vocals": [o[:2] for o in st.get("vocals", [])]}
    mel = sorted([o[:2] for k in ("other", "piano", "guitar") for o in st.get(k, [])])
    merged = []
    for t, s in mel:                     # 旋律：50ms 内的起音去重（六轨并四轨）
        if not merged or t - merged[-1][0] > 0.05:
            merged.append([t, s])
    notes["mel"] = merged
    return {"dur": cues["meta"]["duration"], "rms": rms, "downbeats": [d["t"] for d in cues["downbeats"]],
            "notes": notes}


def energy_integral(A):
    """③ 推力曲线 E 和它的积分 C（240Hz）。"""
    n = int(np.ceil(A["dur"] * HZ)) + 1
    t = np.arange(n) / HZ
    gmax = max([1e-6] + [s for k in A["notes"] for _, s in A["notes"][k]])
    ri = np.minimum(len(A["rms"]) - 1, (t * 100).astype(int))
    E = 0.06 + 0.8 * A["rms"][ri] ** 1.5
    for k, (g, hl) in GAIN.items():
        imp = np.zeros(n)
        for t0, s in A["notes"].get(k, []):
            i = int(np.ceil(t0 * HZ))
            if i < n:
                imp[i] += s / gmax * g
        # 衰减脉冲：指数滤波递推（= Σ g·exp(−(t−ts)/hl)）
        dec = np.exp(-1 / HZ / hl)
        v, out = 0.0, np.zeros(n)
        for i in range(n):
            v = v * dec + imp[i]
            out[i] = v
        E = E + out
    C = np.concatenate([[0.0], np.cumsum(E[1:] / HZ)])
    return C, E


def cum_at(C, t):
    x = np.clip(t * HZ, 0, len(C) - 1)
    i = int(np.floor(x))
    return C[-1] if i >= len(C) - 1 else C[i] + (C[i + 1] - C[i]) * (x - i)


def route(A, lib, seed):
    """② 路线：≈7 秒一句（整数小节），按句子在歌里的相对能量排 呼吸 / 生长 / 变形，在锚点图上走。"""
    R = rng(seed)
    dur, clips = A["dur"], lib["clips"]
    db = list(A["downbeats"])
    if len(db) < 2:
        db = list(np.arange(0, dur, 2.0))
    d = sorted(np.diff(db))
    bar = d[len(d) // 2] if d else 2.0
    k = max(1, round(7 / bar))
    b = db[::k]
    b = ([0.0] + b) if b[0] > 0.5 else ([0.0] + b[1:])
    b = [x for x in b if x < dur - 2] + [dur]
    o = b
    b = [o[0]] + [x for x in o[1:] if x - o[0] >= 2 or x == dur]          # 弱起碎段并进第一句
    o = b
    b = [x for i, x in enumerate(o) if i == 0 or x - o[i - 1] >= 2 or x == dur]
    ph = []
    for a, c in zip(b[:-1], b[1:]):
        seg = A["rms"][int(a * 100):int(c * 100)]
        ph.append((a, c, float(seg.mean()) if len(seg) else 0.0))
    order = sorted(range(len(ph)), key=lambda i: ph[i][2])
    rank = [0.0] * len(ph)
    for r_, pi in enumerate(order):
        rank[pi] = r_ / max(1, len(ph) - 1)
    anchors = sorted({c["to"] for c in clips if c["type"] == "loop"})
    grows = {c["to"]: c for c in clips if c["type"] == "grow"}
    with_grow = [a for a in anchors if a in grows]

    def pick(arr):
        return arr[int(R() * len(arr))]
    cur = pick(with_grow or anchors)
    recent, visited, segs = [], [cur], []

    def seg(x, a, c, why, **kw):
        s = {"clip": x, "start": a, "end": c, "xf": XF, "u0": 0.0, "u1": 1.0, "why": why}
        s.update(kw)
        segs.append(s)
    for i, (a, c, e) in enumerate(ph):
        if i == 0 and cur in grows:
            x = grows[cur]
            seg(x, a, c, "开场：从空底图长出来")
            recent.append(x["id"])
            continue
        want = "loop" if rank[i] < 0.34 else ("grow" if rank[i] < 0.67 else "morph")
        lvl = f"这句能量在全曲排第 {round(rank[i] * 100)}%"
        if segs and segs[-1]["clip"]["type"] == "loop" and want == "loop":
            want = "morph"
            lvl += "（上一句已经停过，不连着停）"
        if want == "grow" and (cur not in grows or grows[cur]["id"] in recent[-3:]):
            want = "morph"
        if want == "grow":                     # 收回再生长：倒放退回 40%，再正放长满
            x, m = grows[cur], a + (c - a) * 0.35
            seg(x, a, m, f"{lvl} → 生长（先收回）", u0=1.0, u1=0.6)
            seg(x, m, c, f"{lvl} → 生长（再长满）", u0=0.6, u1=1.0, xf=0.0)
            recent.append(x["id"])
            continue
        r4 = recent[-4:]
        pool = [x for x in clips if x["from"] == cur and x["type"] == want and x["id"] not in r4]
        if not pool:
            pool = [x for x in clips if x["from"] == cur and x["type"] in ("loop", "morph") and x["id"] not in r4]
        if not pool:
            pool = [x for x in clips if x["from"] == cur and x["type"] in ("loop", "morph")]
        if not pool:
            pool = [x for x in clips if x["type"] == "loop"]
        v3 = visited[-3:]
        fresh = [x for x in pool if x["to"] not in v3 or x["type"] == "loop"]   # 优先去最近没去过的锚点
        x = pick(fresh or pool)
        recent.append(x["id"])
        seg(x, a, c, f"{lvl} → {TYPE_CN[x['type']]}" + (f"（{x['from']} → {x['to']}）" if x["type"] == "morph" else ""))
        cur = x["to"]
        visited.append(cur)
    return segs


def frame_layers(segs, C, t):
    L = []
    for si, s in enumerate(segs):
        if t < s["start"] - s["xf"] or t > s["end"] + s["xf"]:
            continue
        c0, c1 = cum_at(C, s["start"]), cum_at(C, s["end"])
        u = (cum_at(C, min(s["end"], max(s["start"], t))) - c0) / (c1 - c0) if c1 > c0 else 0.0
        u = s["u0"] + (s["u1"] - s["u0"]) * u
        w = 1.0
        if t < s["start"]:
            w = (t - (s["start"] - s["xf"])) / s["xf"]
        if t > s["end"]:
            w = 1 - (t - s["end"]) / s["xf"]
        L.append((si, u, float(np.clip(w, 0, 1))))
    return L


def decode_needed(path, needed, W, H):
    """顺序解码一个片段，只留需要的帧号。"""
    need = set(needed)
    out = {}
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf",
                          f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    n, last = 0, None
    mx = max(need) if need else -1
    while n <= mx:
        raw = p.stdout.read(W * H * 3)
        if len(raw) < W * H * 3:
            break
        last = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
        if n in need:
            out[n] = last
        n += 1
    p.stdout.close()
    p.kill()
    for f in need:                      # 片段比目录写的短：用最后一帧
        if f not in out and last is not None:
            out[f] = last
    return out


def render_chunk(job):
    f0, f1, frames_plan, segs, lib_dir, W, H, fps, out = job
    need = {}
    for g in range(f0, f1):
        for si, fidx, w in frames_plan[g]:
            need.setdefault(si, set()).add(fidx)
    cache = {si: decode_needed(str(Path(lib_dir) / segs[si]["clip"]["file"]), idx, W, H) for si, idx in need.items()}
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                            "-pix_fmt", "yuv420p", "-r", str(fps), "-video_track_timescale", str(int(round(fps * 1000))), out],
                           stdin=subprocess.PIPE)
    blank = np.zeros((H, W, 3), np.uint8)
    for g in range(f0, f1):
        L = frames_plan[g]
        if not L:
            fr = blank
        elif len(L) == 1:
            fr = cache[L[0][0]][L[0][1]]
        else:
            acc = np.zeros((H, W, 3), np.float32)
            ws = sum(w for _, _, w in L) or 1.0
            for si, fidx, w in L:
                acc += cache[si][fidx].astype(np.float32) * (w / ws)
            fr = acc.astype(np.uint8)
        enc.stdin.write(fr.tobytes())
    enc.stdin.close()
    enc.wait()
    return out


def grow(cues_path, lib_dir, out, audio=None, seed=7, preview=False, jobs=3, size="1920x1080"):
    cues = json.loads(Path(cues_path).read_text())
    lib = json.loads((Path(lib_dir) / "catalog.json").read_text())
    audio = audio or str(Path(cues_path).parent / cues["meta"]["source"])
    fps = cues["meta"]["fps"]
    W, H = (int(x) for x in size.split("x"))
    if preview:
        W, H = W // 2, H // 2
    A = music(cues, audio)
    C, E = energy_integral(A)
    segs = route(A, lib, seed)
    nF = int(round(A["dur"] * fps))
    frames_plan = []
    for g in range(nF):
        t = g / fps
        L = frame_layers(segs, C, t)
        frames_plan.append([(si, min(segs[si]["clip"]["frames"] - 1, int(round(u * (segs[si]["clip"]["frames"] - 1)))), w)
                            for si, u, w in L if w > 0])
    # 方案（可读）
    plan = {"meta": {"mode": "grow", "audio": str(Path(audio).resolve()), "cues": str(Path(cues_path).resolve()),
                     "library": str(Path(lib_dir).resolve()), "library_name": lib.get("name"), "seed": seed,
                     "fps": fps, "size": size, "frames": nF, "duration": A["dur"]},
            "segments": [{"i": i + 1, "start": round(s["start"], 3), "end": round(s["end"], 3),
                          "frame_in": int(round(s["start"] * fps)), "frame_out": int(round(s["end"] * fps)),
                          "clip": s["clip"]["id"], "type": s["clip"]["type"], "from": s["clip"]["from"], "to": s["clip"]["to"],
                          "u0": s["u0"], "u1": s["u1"], "why": s["why"]} for i, s in enumerate(segs)],
            "push": {"fps": fps, "values": [round(float(E[min(len(E) - 1, int(g / fps * HZ))]), 3) for g in range(nF)]}}
    pj = Path(out).with_suffix(".plan.json")
    pj.write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    md = [f"# 生长方案 · {Path(audio).name} · 素材库「{lib.get('name')}」· 种子 {seed}", "",
          "| # | 时间 | 片段 | 类型 | 进度 | 为什么 |", "|---|---|---|---|---|---|"]
    for s in plan["segments"]:
        md.append(f"| {s['i']} | {s['start']:.2f}–{s['end']:.2f}s | {s['clip']} | {TYPE_CN[s['type']]} | "
                  f"{s['u0']:.0%}→{s['u1']:.0%} | {s['why']} |")
    pj.with_suffix(".md").write_text("\n".join(md) + "\n")
    # 按段落边界切块并行渲染
    tmp = Path(tempfile.mkdtemp(prefix="beatsync-grow-"))
    bounds = sorted({0, nF} | {int(round(s["start"] * fps)) for s in segs})
    groups = np.array_split(np.arange(len(bounds) - 1), max(1, min(jobs * 2, len(bounds) - 1)))
    chunks = [(bounds[g[0]], bounds[g[-1] + 1]) for g in groups if len(g)]
    jobs_l = [(a, b, frames_plan, segs, lib_dir, W, H, fps, str(tmp / f"c{i:03d}.mp4")) for i, (a, b) in enumerate(chunks)]
    print(f"  {len(segs)} 段 · {nF} 帧 · 分 {len(chunks)} 块渲染", flush=True)
    with ThreadPoolExecutor(jobs) as ex:
        files = list(ex.map(render_chunk, jobs_l))
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{f}'\n" for f in files))
    silent = tmp / "silent.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(silent)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", audio, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-t", f"{nF / fps:.4f}", "-movflags", "+faststart", str(out)],
                   check=True)
    from collections import Counter
    cnt = Counter(s["type"] for s in plan["segments"])
    print(f"✓ {out}  {nF} 帧 · {nF / fps:.2f}s · 呼吸 {cnt['loop']} / 生长 {cnt['grow']} / 变形 {cnt['morph']} 段")
    print(f"  方案：{pj.with_suffix('.md')}")
    return plan


def main(argv=None):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description="小耳音乐卡点MV · 生长模式（凯尔经做法）")
    ap.add_argument("cues")
    ap.add_argument("library", help="素材库目录（含 catalog.json）")
    ap.add_argument("-o", "--out")
    ap.add_argument("--audio")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--size", default="1920x1080")
    a = ap.parse_args(argv)
    out = a.out or str(Path(a.cues).with_name("grow.mp4"))
    grow(a.cues, a.library, out, a.audio, a.seed, a.preview, a.jobs, a.size)


if __name__ == "__main__":
    main()
