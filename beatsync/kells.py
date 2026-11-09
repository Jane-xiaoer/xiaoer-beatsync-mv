#!/usr/bin/env python3
"""小耳音乐卡点MV · 凯尔经模式 —— 一组「生长 / 变形」片段 + 一首歌 → 音乐推着画面长的 MV。

  beatsync kells song.cues.json 片段1 片段2 … [-o mv.mp4] [--preview] [--compare plan_v7.json]

这是小耳《凯尔经 MV》（2026-09-27，v7 定稿）的做法，做成可复用的流程：
  1. 片段：首尾帧图生视频做好的「生长」「变形过渡」，按故事顺序给（mp4，或补帧后的帧序列文件夹）
  2. 编排（原来手排 plan_v7.json，这里自动）：
     每段片子应该「吃掉」的音乐能量 ∝ 它的长度 —— 安静处能量少，同一段片子就会铺得更长（气口自然变长）；
     切点只落在半小节上（强拍 / 第 3 拍），段落起点、乐器进来、气口结束、重音处加分，切在气口中间重罚。
     动态规划挑出整体最好的一组切点。
  3. 推播放头（原 retime.py 原样逻辑）：
     速度 = 0.06 + 0.8·响度^1.5 + Σ 各轨起音推力（旋律 1.0/0.22s、鼓 1.4/0.16s、人声 0.6/0.3s、贝斯 0.7/0.4s，
     每轨按自身最大强度归一）；段内积分归一 → 音符来画面往前长，没音画面停住；段间交叠溶解 0.4s。
  4. --compare：和手排的方案逐个比切点，量自动编排离导演的判断有多远。
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
XF = 0.4
FRAMES_MAP = {}
PUSH = {"other": (1.0, 0.22), "drums": (1.4, 0.16), "vocals": (0.6, 0.30), "bass": (0.7, 0.40)}


# ---------------- 片段 ----------------
class Clip:
    def __init__(self, path):
        self.path = str(path)
        self.name = Path(path).name
        p = Path(path)
        if not p.exists() and p.name in FRAMES_MAP:        # 只排方案时：帧序列还没拷到本机，用已知帧数
            self.n, self.fps, self.frames = FRAMES_MAP[p.name], 96.0, []
            self.dur = self.n / self.fps
            return
        if not p.exists():
            raise SystemExit(f"✗ 找不到片段：{path}")
        if p.is_dir():
            self.frames = sorted(x.name for x in p.iterdir() if x.suffix.lower() in (".jpg", ".png"))
            self.n = len(self.frames)
            self.fps = 96.0 if p.name.endswith("_96") else 30.0
        else:
            out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                                  "-show_entries", "stream=nb_read_packets,r_frame_rate", "-of", "json", self.path],
                                 capture_output=True, text=True).stdout
            st = json.loads(out)["streams"][0]
            a, b = st["r_frame_rate"].split("/")
            self.fps = float(a) / float(b)
            self.n = int(st["nb_read_packets"])
        self.dur = self.n / self.fps

    def load(self, idx, W, H):
        """取出需要的帧（帧号 → BGR 图）。"""
        need = sorted(set(idx))
        out = {}
        if Path(self.path).is_dir():
            for i in need:
                im = cv2.imread(str(Path(self.path) / self.frames[i]))
                out[i] = cover(im, W, H)
            return out
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", self.path, "-vf",
                              f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
                              "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        s, n, last, mx = set(need), 0, None, max(need)
        while n <= mx:
            raw = p.stdout.read(W * H * 3)
            if len(raw) < W * H * 3:
                break
            last = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
            if n in s:
                out[n] = last
            n += 1
        p.stdout.close()
        p.kill()
        for i in need:
            out.setdefault(i, last)
        return out


def cover(im, W, H):
    h, w = im.shape[:2]
    if (w, h) == (W, H):
        return im
    s = max(W / w, H / h)
    im = cv2.resize(im, (int(np.ceil(w * s)), int(np.ceil(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4)
    y0, x0 = (im.shape[0] - H) // 2, (im.shape[1] - W) // 2
    return im[y0:y0 + H, x0:x0 + W]


# ---------------- 音乐能量（retime.py 原样公式） ----------------
def energy_curve(cues, audio):
    import librosa
    y, sr = librosa.load(audio, sr=22050, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    rms = rms / (rms.max() + 1e-12)
    rfps = sr / 512
    dur = cues["meta"]["duration"]
    n = int(np.ceil(dur * HZ)) + 1
    t = np.arange(n) / HZ
    E = 0.06 + 0.8 * rms[np.minimum(len(rms) - 1, (t * rfps).astype(int))] ** 1.5
    on = cues.get("stems", {}).get("onsets", {})
    src = {"other": on.get("other", []) + on.get("piano", []) + on.get("guitar", []),
           "drums": on.get("kick", []) + on.get("snare", []),
           "vocals": on.get("vocals", []), "bass": on.get("bass", [])}
    for k, (g, hl) in PUSH.items():
        arr = np.array([o[:2] for o in src[k]]) if src[k] else np.zeros((0, 2))
        if not len(arr):
            continue
        s = arr[:, 1] / arr[:, 1].max()
        imp = np.zeros(n)
        np.add.at(imp, np.minimum(n - 1, np.ceil(arr[:, 0] * HZ).astype(int)), s * g)
        dec = np.exp(-1 / HZ / hl)
        v, acc = 0.0, np.zeros(n)
        for i in range(n):
            v = v * dec + imp[i]
            acc[i] = v
        E = E + acc
    C = np.concatenate([[0.0], np.cumsum(E[1:] / HZ)])
    return E, C


def energy_from_legacy(audio_json, notes_json, dur):
    """用凯尔经当时的分析数据算能量（retime.py 逐字逻辑），只用于复现检验。"""
    A = json.loads(Path(audio_json).read_text())
    Nn = json.loads(Path(notes_json).read_text())
    rms, fa = np.array(A["rms"]), A["fps"]
    P = []
    for stem, (g, hl) in PUSH.items():
        on = np.array(Nn[stem]["onsets"]) if Nn[stem]["onsets"] else np.zeros((0, 2))
        sv = on[:, 1] / (on[:, 1].max() if len(on) else 1)
        P.append((on[:, 0], sv * g, hl))
    n = int(np.ceil(dur * HZ)) + 1
    t = np.arange(n) / HZ
    E = 0.06 + 0.8 * rms[np.clip((t * fa).astype(int), 0, len(rms) - 1)] ** 1.5
    for ts, g, hl in P:
        for t0, gg in zip(ts, g):
            i0 = int(np.ceil(t0 * HZ))
            i1 = min(n, int(np.ceil((t0 + 6 * hl) * HZ)))
            if i0 < n:
                E[i0:i1] += gg * np.exp(-(t[i0:i1] - t0) / hl)
    C = np.concatenate([[0.0], np.cumsum(E[1:] / HZ)])
    return E, C


def cum(C, t):
    return float(np.interp(t * HZ, np.arange(len(C)), C))


# ---------------- 自动编排 ----------------
def candidates(cues):
    """切点候选：半小节（第 1、3 拍）。强拍把握不足（<0.6，前后半小节分不清）时两者一视同仁。"""
    meter = cues["meta"]["meter"]
    sure = cues["meta"].get("downbeat_conf", 0) >= 0.6
    st = cues.get("stems", {})
    out = []
    for b in cues["beats"]:
        if b["beat"] not in (1, meter // 2 + 1):
            continue
        t, sc, why = b["t"], 0.0, []
        why.append(f"第{b['bar']}小节" + ("强拍" if b["beat"] == 1 else f"第{b['beat']}拍"))
        if sure and b["beat"] == 1:
            sc += 0.5
        secs = cues["sections"]
        for i_, s in enumerate(secs[1:], 1):
            if abs(s["start"] - t) < 0.25 and not in_quiet_zone(cues, t):
                jump = abs(s["energy"] - secs[i_ - 1]["energy"])      # 能量跳得越大，越该在这里换
                sc += 2.0 * min(1.0, jump / 0.25); why.append(f"进入新段落（{s['level']}，能量跳 {jump:.2f}）")
        for e in st.get("entries", []):
            if e["type"] == "进" and abs(e["t"] - t) < 0.4:
                sc += 1.0; why.append(f"{ {'drums': '鼓', 'bass': '贝斯', 'vocals': '人声', 'other': '旋律'}.get(e['stem'], e['stem'])}进来")
                break
        for h in cues["hits"]:
            if abs(h["t"] - t) < 0.08:
                sc += 0.6 * h["conf"]; why.append("重音")
                break
        if in_quiet_zone(cues, t):
            sc -= 8.0; why.append("（气口前后，不该切）")
        out.append({"t": t, "score": sc, "why": "、".join(why)})
    # 音乐回来、站稳之后的第一个候选点：换下一段（凯尔经 v7：气口 33.8–37.0 → 39.77 换）
    for br in cues["breaths"]:
        if br["end"] - br["start"] >= 0.6:
            after = [c for c in out if c["t"] >= br["end"] + 2.0]
            if after:
                after[0]["score"] += 3.0
                after[0]["why"] += "、气口过后音乐站稳"
    return out


def in_quiet_zone(cues, t):
    """气口前 1.5 秒到音乐回来后 2 秒：不切 —— 让上一段带着走进静音，音乐回来、站稳了再换。"""
    for br in cues["breaths"]:
        if br["end"] - br["start"] >= 0.6 and br["start"] - 1.5 < t < br["end"] + 2.0:
            return True
    return False


def desired_len(cues, a, b):
    """这一段想要多长（秒）：一句 = 2 小节；安静段铺两句，高潮一句半（再密一点），含气口的段再加上气口长度。"""
    beat = 60 / cues["meta"]["tempo"]
    unit = 2 * cues["meta"]["meter"] * beat
    mid = (a + b) / 2
    lv = next((s["level"] for s in cues["sections"] if s["start"] <= mid < s["end"]), "中")
    L = {"安静": 2.0, "中": 1.0, "高": 1.0, "最高": 0.75}[lv] * unit
    for br in cues["breaths"]:
        if a < br["start"] and br["end"] < b:
            L += br["end"] - br["start"]
    return L


def clip_type(clip):
    n = clip.name.lower()
    return "morph" if "morph" in n or "变形" in n else ("grow" if "grow" in n or "生长" in n else "")


def straddle_bonus(cues, clip, a, b):
    """变形片段跨过段落起点（能量往上的那种）：段落起点落在这段的 15%～60% 处 → 变过去的那一下正好迎上新段落。
    （凯尔经 v7：t2 变形 59.90–66.29，高潮 61.94 落在 32% 处。）"""
    if clip_type(clip) != "morph":
        return 0.0
    for i, s in enumerate(cues["sections"][1:], 1):
        prev = cues["sections"][i - 1]
        if s["energy"] > prev["energy"] and a < s["start"] < b:
            r = (s["start"] - a) / (b - a)
            if 0.15 <= r <= 0.6:
                return 4.0
    return 0.0


def auto_plan(cues, clips, C, dur, w_len=6.0):
    cand = [{"t": 0.0, "score": 0.0, "why": "开场"}] + [c for c in candidates(cues) if 1.0 < c["t"] < dur - 1.0]
    T = [c["t"] for c in cand] + [dur]
    M, N = len(T), len(clips)
    tot = cum(C, dur)
    INF = -1e18
    f = np.full((N + 1, M), INF)
    back = np.zeros((N + 1, M), int)
    f[0, 0] = 0.0
    for i in range(1, N + 1):
        last = (i == N)
        js = [M - 1] if last else range(1, M - 1)
        for j in js:
            best, arg = INF, -1
            for k in range(0, j):
                if f[i - 1, k] == INF or T[j] - T[k] < 1.2:
                    continue
                L = desired_len(cues, T[k], T[j])
                if last:
                    L = max(L, T[j] - T[k])          # 最后一段：结尾铺到底，不罚长
                cost = w_len * np.log((T[j] - T[k]) / L) ** 2
                v = f[i - 1, k] - cost + (0 if last else cand[j]["score"]) + straddle_bonus(cues, clips[i - 1], T[k], T[j])
                if v > best:
                    best, arg = v, k
            f[i, j], back[i, j] = best, arg
    js, j = [], M - 1
    for i in range(N, 0, -1):
        js.append(j)
        j = back[i, j]
    js.append(0)
    js = js[::-1]
    segs = []
    for i in range(N):
        a, b = T[js[i]], T[js[i + 1]]
        segs.append({"id": Path(clips[i].name).stem.replace("_96", ""), "clip": clips[i].path, "start": round(a, 3),
                     "end": round(b, 3), "xf": XF, "why": cand[js[i]]["why"] if js[i] < len(cand) else "",
                     "energy_share": round((cum(C, b) - cum(C, a)) / tot, 3),
                     "target_share": round(desired_len(cues, a, b), 2)})
    return segs


# ---------------- 出片（retime.py 的合成） ----------------
def render(segs, clips, C, dur, fps, W, H, audio, out, jobs=4):
    nF = int(round(dur * fps))
    by_id = {c.path: c for c in clips}
    plan_f = []
    for g in range(nF):
        t = g / fps
        L = []
        for si, s in enumerate(segs):
            if s["start"] - s["xf"] <= t <= s["end"] + s["xf"]:
                c0, c1 = cum(C, s["start"]), cum(C, s["end"])
                u = (cum(C, min(s["end"], max(s["start"], t))) - c0) / (c1 - c0) if c1 > c0 else 0.0
                u = s.get("u0", 0.0) + (s.get("u1", 1.0) - s.get("u0", 0.0)) * u
                n = by_id[s["clip"]].n
                idx = min(n - 1, int(round(u * (n - 1))))
                w = 1.0
                if t < s["start"]:
                    w = (t - (s["start"] - s["xf"])) / s["xf"]
                if t > s["end"]:
                    w = 1 - (t - s["end"]) / s["xf"]
                if w > 0:
                    L.append((si, idx, min(1.0, w)))
        plan_f.append(L)
    tmp = Path(tempfile.mkdtemp(prefix="beatsync-kells-"))
    bounds = sorted({0, nF} | {int(round(s["start"] * fps)) for s in segs})
    groups = [g for g in np.array_split(np.arange(len(bounds) - 1), max(1, min(jobs * 2, len(bounds) - 1))) if len(g)]
    chunks = [(bounds[g[0]], bounds[g[-1] + 1]) for g in groups]

    def one(ch):
        a, b = ch
        need = {}
        for g in range(a, b):
            for si, idx, _ in plan_f[g]:
                need.setdefault(si, set()).add(idx)
        cache = {si: by_id[segs[si]["clip"]].load(idx, W, H) for si, idx in need.items()}
        f = tmp / f"c{a:06d}.mp4"
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                                "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                                "-pix_fmt", "yuv420p", "-r", str(fps), "-video_track_timescale", str(int(round(fps * 1000))), str(f)],
                               stdin=subprocess.PIPE)
        for g in range(a, b):
            L = plan_f[g]
            L = [(si, idx, w) for si, idx, w in L]
            if len(L) == 1:
                fr = zoomed(cache[L[0][0]][L[0][1]], segs[L[0][0]].get("zoom", 1.0))
            elif not L:
                fr = np.zeros((H, W, 3), np.uint8)
            else:
                tw = sum(w for *_, w in L)
                acc = np.zeros((H, W, 3), np.float32)
                for si, idx, w in L:
                    acc += zoomed(cache[si][idx], segs[si].get("zoom", 1.0)).astype(np.float32) * (w / tw)
                fr = acc.clip(0, 255).astype(np.uint8)
            enc.stdin.write(fr.tobytes())
        enc.stdin.close()
        enc.wait()
        return f
    with ThreadPoolExecutor(jobs) as ex:
        files = list(ex.map(one, chunks))
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{f}'\n" for f in files))
    silent = tmp / "silent.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(silent)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "256k", "-t", f"{nF / fps:.4f}", "-movflags", "+faststart", str(out)], check=True)
    return nF


def zoomed(im, z):
    if abs(z - 1.0) < 1e-6:
        return im
    H, W = im.shape[:2]
    cw, ch = W / z, H / z
    x0, y0 = int(round((W - cw) / 2)), int(round((H - ch) / 2))
    return cv2.resize(im[y0:y0 + int(round(ch)), x0:x0 + int(round(cw))], (W, H), interpolation=cv2.INTER_LANCZOS4)


def compare(segs, ref_path, fps):
    ref = json.loads(Path(ref_path).read_text())["segments"]
    rb = [s["start"] for s in ref][1:]
    ab = [s["start"] for s in segs][1:]
    rows = []
    for x in rb:
        j = int(np.argmin([abs(y - x) for y in ab]))
        rows.append((x, ab[j], round((ab[j] - x) * fps)))
    return rows


def main(argv=None):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description="小耳音乐卡点MV · 凯尔经模式")
    ap.add_argument("cues")
    ap.add_argument("clips", nargs="*", help="按故事顺序排好的片段（mp4 或帧序列文件夹）")
    ap.add_argument("-o", "--out")
    ap.add_argument("--audio")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--compare", help="手排的方案（plan_v7.json 这类），逐个比切点")
    ap.add_argument("--plan-only", action="store_true", help="只排方案不出片")
    ap.add_argument("--use-plan", help="不自动编排，直接用手排的方案（plan_v7.json 格式，dir 相对方案所在目录）")
    ap.add_argument("--legacy", nargs=2, metavar=("AUDIO_JSON", "NOTES_JSON"), help="（复现检验）用凯尔经当时的分析数据算能量")
    ap.add_argument("--frames-map", help="（只排方案用）帧序列文件夹名 → 帧数 的清单文件，每行「名字 帧数」")
    a = ap.parse_args(argv)
    if a.frames_map:
        for ln in Path(a.frames_map).read_text().split("\n"):
            if ln.strip():
                k, v = ln.split()
                FRAMES_MAP[k] = int(v)
    cues = json.loads(Path(a.cues).read_text())
    audio = a.audio or str(Path(a.cues).parent / cues["meta"]["source"])
    fps = cues["meta"]["fps"]
    dur = cues["meta"]["duration"]
    if a.use_plan:
        ref = json.loads(Path(a.use_plan).read_text())
        base = Path(a.use_plan).parent
        clips, segs = [], []
        for sg in ref["segments"]:
            c = Clip(base / sg["dir"])
            clips.append(c)
            segs.append({"id": sg["id"], "clip": c.path, "start": sg["start"], "end": sg["end"], "xf": sg.get("xf", XF),
                         "u0": sg.get("u0", 0.0), "u1": sg.get("u1", 1.0), "zoom": sg.get("zoom", 1.0),
                         "why": "手排", "energy_share": 0, "target_share": 0})
    else:
        clips = [Clip(p) for p in a.clips]
    E, C = energy_from_legacy(a.legacy[0], a.legacy[1], dur) if a.legacy else energy_curve(cues, audio)
    if not a.use_plan:
        segs = auto_plan(cues, clips, C, dur)
    out = Path(a.out or Path(a.cues).with_name("kells.mp4"))
    pj = out.with_suffix(".plan.json")
    pj.write_text(json.dumps({"meta": {"mode": "kells", "audio": str(Path(audio).resolve()), "fps": fps, "duration": dur},
                              "segments": segs}, ensure_ascii=False, indent=1))
    print(f"✓ 方案 {pj}")
    for s in segs:
        print(f"  {s['start']:7.2f}–{s['end']:7.2f}s  {s['id']:<10} {s['end'] - s['start']:5.2f}s（想要 {s['target_share']:.1f}s）  {s['why']}")
    if a.compare:
        rows = compare(segs, a.compare, fps)
        d = np.array([abs(r[2]) for r in rows])
        print(f"\n和手排方案比（{len(rows)} 个切点）：一致（≤2 帧）{int(np.sum(d <= 2))} · ≤1 拍内 {int(np.sum(d <= round(fps * 60 / cues['meta']['tempo'])))} · 中位差 {np.median(d):.0f} 帧")
        for x, y, df in rows:
            print(f"  手排 {x:7.2f}s  自动 {y:7.2f}s  差 {df:+d} 帧")
    if a.plan_only:
        return
    W, H = (int(v) for v in a.size.split("x"))
    if a.preview:
        W, H = W // 2, H // 2
    nF = render(segs, clips, C, dur, fps, W, H, audio, out, a.jobs)
    print(f"✓ {out}  {nF} 帧 · {nF / fps:.2f}s")


if __name__ == "__main__":
    main()
