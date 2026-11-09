#!/usr/bin/env python3
"""小耳音乐卡点MV · 第 2 步：编排 —— 按「拍 / 乐句 / 段落」三层规则，把素材排成一张剪辑方案。

  beatsync plan cues.json clips/ [-o plan.json] [--density standard] [--size 1920x1080]

三层规则：
  · 段落（结构层）：每段的能量档决定剪得多密；进新段落必切；全片最高那段用最有劲的素材。
  · 乐句（句子层）：安静段只在乐句交界处换镜头；镜头运动跟能量走——往上推近，回落拉远。
  · 拍（点层）：高能段切在强拍、半小节、重音上；置信度低的点不硬卡；气口里不切、画面放慢停住，
    下一刀落在音乐回来的那一下。
素材分配：按素材自身的运动量排序，安静段配静的、高潮配动的；最动的那一批留给高潮，不在前面用光；
同一素材再次出现时接着上次往后放，不重复同一段画面。

每个镜头都写了「为什么切在这里」，创作者可以逐条改、删、锁定后再渲染或导出。
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# 每档能量用什么粒度切：phrase / 2bar / bar / half / beat
GRAIN = {
    "calm":      {"安静": "phrase", "中": "phrase", "高": "2bar", "最高": "bar"},
    "standard":  {"安静": "phrase", "中": "2bar", "高": "bar", "最高": "half"},
    "energetic": {"安静": "2bar", "中": "bar", "高": "half", "最高": "beat"},
}
GRAIN_ORDER = ["phrase", "2bar", "bar", "half", "beat"]


def probe(path):
    p = Path(path)
    if p.suffix.lower() in IMAGE_EXT:
        return {"path": str(p), "kind": "image", "duration": 1e9, "motion": 0.0}
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
    dur = float(out or 0)
    # 运动量：每秒抽 2 帧、缩到 64x36 灰度，算平均帧差
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vf", "fps=2,scale=64:36,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8)
    n = len(fr) // (64 * 36)
    motion = 0.0
    if n > 2:
        f = fr[:n * 64 * 36].reshape(n, -1).astype(np.float32)
        motion = float(np.mean(np.abs(np.diff(f, axis=0))))
    return {"path": str(p), "kind": "video", "duration": dur, "motion": round(motion, 3)}


def load_clips(src, cache_dir):
    paths = []
    for s in src:
        s = Path(s)
        if s.is_dir():
            paths += sorted(x for x in s.iterdir() if x.suffix.lower() in VIDEO_EXT | IMAGE_EXT
                            and not x.name.startswith("."))
        elif s.exists():
            paths.append(s)
    cache = Path(cache_dir) / ".beatsync-clips.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    clips = []
    for p in paths:
        key = f"{p.resolve()}|{p.stat().st_mtime_ns}"
        if key not in known:
            known[key] = probe(p)
        clips.append(known[key])
    cache.write_text(json.dumps(known, ensure_ascii=False))
    return [c for c in clips if c["duration"] > 0.3]


def energy_at(cues, t, smooth_s=0.0):
    e = cues["energy"]["values"]
    if smooth_s > 0:
        key = f"_sm{smooth_s}"
        if key not in cues["energy"]:
            k = max(1, int(smooth_s * cues["energy"]["rate"]))
            cues["energy"][key] = np.convolve(np.pad(e, k, mode="edge"), np.ones(k) / k, mode="same")[k:-k].tolist()
        e = cues["energy"][key]
    i = int(t * cues["energy"]["rate"])
    return float(e[min(max(i, 0), len(e) - 1)])


def section_of(cues, t):
    for s in cues["sections"]:
        if s["start"] - 1e-6 <= t < s["end"]:
            return s
    return cues["sections"][-1]


def candidate_points(cues, grain, t0, t1):
    """某段 [t0,t1) 里，按粒度给出候选切点（带说明）。"""
    pts = []
    beats = [b for b in cues["beats"] if t0 - 1e-6 <= b["t"] < t1 - 1e-6]
    meter = cues["meta"]["meter"]
    for b in beats:
        ok, why = False, ""
        if b["beat"] == 1:
            if grain == "2bar":
                ok, why = (b["bar"] % 2 == 1), f"第{b['bar']}小节强拍（两小节一切）"
            elif grain in ("bar", "half", "beat"):
                ok, why = True, f"第{b['bar']}小节强拍"
        elif grain == "half" and b["beat"] == meter // 2 + 1:
            ok, why = True, f"第{b['bar']}小节第{b['beat']}拍（半小节）"
        elif grain == "beat":
            ok, why = True, f"第{b['bar']}小节第{b['beat']}拍"
        if ok and b["conf"] >= 0.35:
            pts.append({"t": b["t"], "prio": 1 if b["beat"] == 1 else 0.5, "conf": b["conf"], "why": why})
    for p in cues["phrases"]:
        if t0 - 1e-6 <= p["t"] < t1 - 1e-6:
            pts.append({"t": p["t"], "prio": 2, "conf": p["conf"], "why": f"乐句起点（第{p['bar']}小节）"})
    if grain in ("half", "beat"):
        for h in cues["hits"]:
            if t0 <= h["t"] < t1 and h["conf"] >= 0.5:
                pts.append({"t": h["t"], "prio": 1.5, "conf": h["conf"], "why": f"重音（强度 {h['conf']:.2f}）"})
    return pts


def build_cuts(cues, density, min_shot, max_shot):
    dur = cues["meta"]["duration"]
    pts = [{"t": 0.0, "prio": 9, "conf": 1, "why": "开场"}]
    for s in cues["sections"]:
        if s["start"] > 0:
            pts.append({"t": s["start"], "prio": 3, "conf": 1,
                        "why": f"进入新段落（{s['level']}，能量 {s['energy']:.2f}）"})
        grain = GRAIN[density][s["level"]]
        pts += candidate_points(cues, grain, s["start"], s["end"])
    # 分轨切点：人声起句、乐器进来；高能段里的军鼓
    st = cues.get("stems")
    if st:
        CN = {"drums": "鼓", "bass": "贝斯", "vocals": "人声", "other": "旋律", "piano": "钢琴", "guitar": "吉他"}
        for ln in st.get("lines", []):
            pts.append({"t": ln["start"], "prio": 2.2, "conf": 0.8, "why": "人声起句"})
        for en in st.get("entries", []):
            if en["type"] == "进" and en["t"] > 0.5:
                # 乐器进来的时刻吸附到最近的强拍（±半拍），没有就用原时刻
                db = [b for b in cues["downbeats"] if abs(b["t"] - en["t"]) <= 60 / cues["meta"]["tempo"] / 2]
                t = db[0]["t"] if db else en["t"]
                pts.append({"t": t, "prio": 2.5, "conf": 0.8, "why": f"{CN.get(en['stem'], en['stem'])}进来"})
        for s_ in cues["sections"]:
            if GRAIN[density][s_["level"]] in ("half", "beat"):
                for t, sv, _f in st["onsets"].get("snare", []):
                    if s_["start"] <= t < s_["end"] and sv >= 0.5:
                        pts.append({"t": t, "prio": 1.2, "conf": sv, "why": f"军鼓（强度 {sv:.2f}）"})

    # 气口：气口里不切；气口结束、音乐回来的那一下必切
    for b in cues["breaths"]:
        if b["end"] - b["start"] >= 0.4:
            pts = [p for p in pts if not (b["start"] + 0.05 < p["t"] < b["end"] - 0.05)]
            pts.append({"t": b["end"], "prio": 3, "conf": 0.8, "why": "气口结束，音乐回来"})

    # 同一时刻合并（重音和拍可能差几毫秒）：保留优先级高的
    pts.sort(key=lambda p: (p["t"], -p["prio"]))
    merged = []
    for p in pts:
        if merged and p["t"] - merged[-1]["t"] < 0.06:
            if p["prio"] > merged[-1]["prio"]:
                merged[-1] = p
            continue
        merged.append(p)

    # 最短镜头：太近的两刀，删优先级低的
    changed = True
    while changed:
        changed = False
        for i in range(1, len(merged)):
            if merged[i]["t"] - merged[i - 1]["t"] < min_shot:
                drop = i if merged[i]["prio"] <= merged[i - 1]["prio"] else i - 1
                if drop == 0:
                    drop = i
                merged.pop(drop)
                changed = True
                break

    # 最长镜头：太长的空档，从更细的拍里补一刀（选离中点最近的强拍）
    out = [merged[0]]
    for p in merged[1:] + [{"t": dur, "prio": 9, "conf": 1, "why": "结束"}]:
        while p["t"] - out[-1]["t"] > max_shot:
            a = out[-1]["t"]
            mid = (a + p["t"]) / 2
            fill = [b for b in cues["beats"] if a + min_shot <= b["t"] <= p["t"] - min_shot and b["conf"] >= 0.35]
            if not fill:
                break
            b = min(fill, key=lambda b: (abs(b["t"] - mid) - (0.3 if b["beat"] == 1 else 0)))
            out.append({"t": b["t"], "prio": 0.8, "conf": b["conf"],
                        "why": f"镜头太长，补在第{b['bar']}小节第{b['beat']}拍"})
        out.append(p)
    return out  # 最后一个是「结束」


def assign(cues, cuts, clips, fps, seed):
    rng = np.random.default_rng(seed)
    order = sorted(range(len(clips)), key=lambda i: clips[i]["motion"])
    rank = {ci: k / max(1, len(order) - 1) for k, ci in enumerate(order)}   # 0=最静 1=最动
    top_q = {ci for ci, q in rank.items() if q >= 0.75} if len(clips) >= 4 else set()
    cursor = {i: 0.0 for i in range(len(clips))}
    recent = []
    used_in_peak = False
    shots = []
    for k in range(len(cuts) - 1):
        a, b = cuts[k]["t"], cuts[k + 1]["t"]
        fa, fb = int(round(a * fps)), int(round(b * fps))
        if fb <= fa:
            continue
        sec = section_of(cues, (a + b) / 2)
        # 运动看的是「这一句」的走向，不是瞬时起伏：用 3 秒平滑的能量，并把镜头前后各延 1 秒
        ea, eb = energy_at(cues, max(0, a - 1.0), 3.0), energy_at(cues, b + 1.0, 3.0)
        em = float(np.mean([energy_at(cues, t) for t in np.linspace(a, b, 6)]))
        in_breath = any(br["start"] <= a + 0.02 and b <= br["end"] + 0.3 for br in cues["breaths"])

        # 镜头运动：跟能量走
        d = eb - ea
        if in_breath:
            move, why_m = {"type": "hold", "from": 1.0, "to": 1.0}, "气口：画面几乎停住"
        elif d > 0.08:
            move, why_m = {"type": "push", "from": 1.0, "to": round(1.0 + min(0.18, 0.06 + d * 0.4), 3)}, "能量往上走 → 推近"
        elif d < -0.08:
            move, why_m = {"type": "pull", "from": round(1.0 + min(0.18, 0.06 - d * 0.4), 3), "to": 1.0}, "能量回落 → 拉远"
        elif sec["level"] in ("最高", "高"):
            move, why_m = {"type": "push", "from": 1.0, "to": 1.05}, "高能段 → 轻推"
        else:
            move, why_m = {"type": "drift", "from": 1.04, "to": 1.0}, "平稳段 → 缓慢拉开"
        speed = 0.15 if in_breath else round(float(np.clip(0.7 + 0.6 * em, 0.6, 1.3)), 2)

        # 选素材：目标运动量 = 当前能量；最动的那批留给高潮
        target = float(np.clip(em, 0, 1))
        pool = list(range(len(clips)))
        if sec["level"] not in ("最高", "高") and not used_in_peak:
            pool = [i for i in pool if i not in top_q] or pool
        pool = [i for i in pool if i not in recent[-3:]] or pool
        need = (b - a) * speed
        def cost(i):
            c = abs(rank[i] - target)
            if clips[i]["kind"] == "video" and cursor[i] + need > clips[i]["duration"]:
                c += 0.5
            return c + rng.uniform(0, 0.08)
        ci = min(pool, key=cost)
        clip = clips[ci]
        src_in = cursor[ci]
        if clip["kind"] == "video" and src_in + need > clip["duration"]:
            src_in = 0.0
        cursor[ci] = src_in + need
        recent.append(ci)
        if sec["level"] == "最高":
            used_in_peak = True

        shots.append({
            "i": len(shots) + 1, "start": round(a, 3), "end": round(b, 3),
            "frame_in": fa, "frame_out": fb, "frames": fb - fa,
            "section": sec["level"], "energy": round(em, 2),
            "cut_why": cuts[k]["why"], "cut_conf": round(float(cuts[k]["conf"]), 2),
            "clip": clip["path"], "src_in": round(src_in, 3), "speed": speed,
            "move": move, "move_why": why_m, "locked": False,
        })
    return shots


def to_md(plan):
    m = plan["meta"]
    L = [f"# 剪辑方案 · {Path(m['audio']).name}", "",
         f"{m['duration']}s · {m['tempo']} BPM · {m['fps']}fps · 节奏密度 {m['density']} · 共 {len(plan['shots'])} 个镜头", "",
         "| # | 时间 | 帧 | 段落 | 为什么切在这里 | 素材 | 运动 |", "|---|---|---|---|---|---|---|"]
    for s in plan["shots"]:
        L.append(f"| {s['i']} | {s['start']:.2f}–{s['end']:.2f}s | {s['frame_in']}–{s['frame_out']} | {s['section']} | "
                 f"{s['cut_why']} | {Path(s['clip']).name} @{s['src_in']:.1f}s ×{s['speed']} | {s['move_why']} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="小耳音乐卡点MV · 编排")
    ap.add_argument("cues")
    ap.add_argument("clips", nargs="+", help="素材文件夹或文件（视频/图片）")
    ap.add_argument("-o", "--out")
    ap.add_argument("--audio", help="音乐文件（默认按 cues 的 source 在 cues 同目录找）")
    ap.add_argument("--density", choices=list(GRAIN), default="standard")
    ap.add_argument("--min-shot", type=float, default=0.45)
    ap.add_argument("--max-shot", type=float, default=6.0)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    cues = json.loads(Path(a.cues).read_text())
    fps = cues["meta"]["fps"]
    clips = load_clips(a.clips, Path(a.cues).parent)
    if not clips:
        raise SystemExit("✗ 没找到可用素材")
    cuts = build_cuts(cues, a.density, a.min_shot, a.max_shot)
    shots = assign(cues, cuts, clips, fps, a.seed)
    audio = a.audio or str(Path(a.cues).parent / cues["meta"]["source"])
    plan = {"meta": {"audio": str(Path(audio).resolve()), "cues": str(Path(a.cues).resolve()),
                     "duration": cues["meta"]["duration"], "tempo": cues["meta"]["tempo"],
                     "fps": fps, "size": a.size, "density": a.density,
                     "frames": int(round(cues["meta"]["duration"] * fps))},
            "shots": shots}
    out = Path(a.out or Path(a.cues).with_name("plan.json"))
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    out.with_suffix(".md").write_text(to_md(plan))
    lens = [s["end"] - s["start"] for s in shots]
    print(f"✓ {out}  {len(shots)} 个镜头 · 平均 {np.mean(lens):.2f}s · 最短 {min(lens):.2f}s · 最长 {max(lens):.2f}s")
    by = {}
    for s in shots:
        by.setdefault(s["section"], []).append(s["end"] - s["start"])
    for k in ("安静", "中", "高", "最高"):
        if k in by:
            print(f"  {k:>2}：{len(by[k])} 个镜头，平均 {np.mean(by[k]):.2f}s")
    print(f"  可读版：{out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
