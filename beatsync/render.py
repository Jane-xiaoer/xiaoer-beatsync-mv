#!/usr/bin/env python3
"""小耳音乐卡点MV · 第 3 步（A）：出片 —— 按剪辑方案渲染 MV，分轨驱动画面，一个起音 = 同一帧一个看得见的动作。

  beatsync render plan.json [-o mv.mp4] [--jobs 4] [--preview] [--layers push,kick,snare,bass,vocals]

切点：每个镜头刚好 N 帧（N = 方案里的 frames），第 k 刀永远落在 round(t_k × fps) 那一帧，误差不累积。
镜头内部（有分轨时）：
  push    旋律/鼓/人声/贝斯的起音推着画面往前走 —— 凯尔经公式：
          播放头每帧前进 ∝ 0.06 + 响度^1.5 + Σ 权重×起音强度×e^(−Δt/τ)（旋律 1.0/0.22s、鼓 1.4/0.16s、
          人声 0.6/0.3s、贝斯 0.7/0.4s）。安静处画面几乎停住，音符一来画面就往前涌。
  kick    底鼓 → 镜头往前顶一下（+3%×强度，0.12s 回落）
  snare   军鼓 → 画面亮一下（+10%×强度，0.08s 回落）
  bass    贝斯 → 光的呼吸（亮度随贝斯包络起伏）
  vocals  人声 → 唱的时候镜头慢慢贴近（随人声包络 +2%）
没有分轨时退回：恒速播放 + 方案里的推拉。
"""
import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import cv2

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
ALL_LAYERS = ["push", "kick", "snare", "bass", "vocals"]


class Source:
    """顺序读取素材帧（已缩放裁切到 W×H），支持小数播放头 + 相邻帧混合。"""

    def __init__(self, path, src_in, W, H, fps):
        self.W, self.H = W, H
        self.is_img = Path(path).suffix.lower() in IMAGE_EXT
        if self.is_img:
            im = cv2.imread(path)
            self.still = cover(im, W, H)
            return
        vf = f"fps={fps},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{src_in:.3f}", "-stream_loop", "-1", "-i", path,
                                   "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.buf = []          # [(index, frame)]
        self.next_idx = 0

    def _read(self):
        raw = self.p.stdout.read(self.W * self.H * 3)
        if len(raw) < self.W * self.H * 3:
            fr = self.buf[-1][1] if self.buf else np.zeros((self.H, self.W, 3), np.uint8)
        else:
            fr = np.frombuffer(raw, np.uint8).reshape(self.H, self.W, 3)
        self.buf.append((self.next_idx, fr))
        self.next_idx += 1
        if len(self.buf) > 3:
            self.buf.pop(0)

    def at(self, pos):
        if self.is_img:
            return self.still
        i = int(np.floor(pos))
        while self.next_idx <= i + 1:
            self._read()
        a = next(f for k, f in self.buf if k == i) if any(k == i for k, _ in self.buf) else self.buf[0][1]
        b = next(f for k, f in self.buf if k == i + 1)
        w = pos - i
        if w < 0.02:
            return a
        return cv2.addWeighted(a, 1 - w, b, w, 0)

    def close(self):
        if not self.is_img:
            self.p.stdout.close()
            self.p.kill()


def cover(im, W, H):
    h, w = im.shape[:2]
    s = max(W / w, H / h)
    im = cv2.resize(im, (int(np.ceil(w * s)), int(np.ceil(h * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (im.shape[0] - H) // 2, (im.shape[1] - W) // 2
    return im[y0:y0 + H, x0:x0 + W]


def zoom(fr, z):
    if z <= 1.0005:
        return fr
    H, W = fr.shape[:2]
    M = np.float32([[z, 0, W / 2 * (1 - z)], [0, z, H / 2 * (1 - z)]])
    return cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def pulses(onsets, nF, fps, tau, amp=1.0):
    """起音 → 每帧的脉冲曲线（起音那一帧满值，按 τ 指数回落）。"""
    out = np.zeros(nF)
    tt = np.arange(nF) / fps
    for t0, s, *_ in onsets:
        i0 = int(round(t0 * fps))          # 和切点同一规则：离起音最近的那一帧满值
        if i0 >= nF:
            continue
        n = min(nF - i0, int(tau * 6 * fps) + 1)
        out[i0:i0 + n] = np.maximum(out[i0:i0 + n], amp * s * np.exp(-(tt[i0:i0 + n] - tt[i0]) / tau))
    return out


def env_at_frames(env, rate, nF, fps):
    x = np.arange(len(env)) / rate
    return np.interp(np.arange(nF) / fps, x, env)


def drive_curves(cues, nF, fps, layers):
    st = cues.get("stems")
    C = {"push": np.ones(nF), "zoom": np.zeros(nF), "gain": np.ones(nF)}
    if not st:
        return C, False
    if "push" in layers:
        p = np.array(st["push"]["values"][:nF] + [st["push"]["values"][-1]] * max(0, nF - len(st["push"]["values"])))
        C["push"] = p / (np.mean(p) + 1e-9)
    on = st["onsets"]
    if "kick" in layers and on.get("kick"):
        C["zoom"] += pulses(on["kick"], nF, fps, 0.12, 0.03)
    if "vocals" in layers and "vocals" in st["env"]:
        C["zoom"] += 0.02 * env_at_frames(st["env"]["vocals"], st["env"]["rate"], nF, fps)
    if "snare" in layers and on.get("snare"):
        C["gain"] += pulses(on["snare"], nF, fps, 0.08, 0.10)
    if "bass" in layers and "bass" in st["env"]:
        b = env_at_frames(st["env"]["bass"], st["env"]["rate"], nF, fps)
        C["gain"] *= 0.94 + 0.10 * b
    return C, True


def render_group(shots, C, fps, W, H, out, use_push):
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
                            "-pix_fmt", "yuv420p", "-r", str(fps), "-video_track_timescale", str(int(round(fps * 1000))), out],
                           stdin=subprocess.PIPE)
    for s in shots:
        src = Source(s["clip"], s["src_in"], W, H, fps)
        n = s["frames"]
        z0, z1 = s["move"]["from"], s["move"]["to"]
        pos = 0.0
        for j in range(n):
            g = s["frame_in"] + j
            p = j / max(1, n - 1)
            e = p * p * (3 - 2 * p)
            fr = src.at(pos)
            z = (z0 + (z1 - z0) * e) * (1 + C["zoom"][g])
            fr = zoom(fr, z)
            gain = C["gain"][g]
            if abs(gain - 1) > 0.004:
                fr = cv2.convertScaleAbs(fr, alpha=gain, beta=0)
            enc.stdin.write(fr.tobytes())
            pos += float(np.clip(s["speed"] * (C["push"][g] if use_push else 1.0), 0.0, 3.0))
        src.close()
    enc.stdin.close()
    enc.wait()
    if enc.returncode != 0:
        raise RuntimeError("编码失败")


def render(plan_path, out, jobs=4, preview=False, layers=ALL_LAYERS):
    plan = json.loads(Path(plan_path).read_text())
    cues = json.loads(Path(plan["meta"]["cues"]).read_text())
    m = plan["meta"]
    fps = m["fps"]
    W, H = (int(x) for x in m["size"].split("x"))
    if preview:
        W, H = W // 2, H // 2
    shots = plan["shots"]
    total = shots[-1]["frame_out"]
    C, has = drive_curves(cues, total + 2, fps, layers)
    print(f"  画面驱动：{'分轨 ' + '/'.join(layers) if has else '无分轨（恒速）'}", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="beatsync-"))
    k = max(1, min(jobs, len(shots)))
    groups = [list(g) for g in np.array_split(np.arange(len(shots)), k) if len(g)]
    files = [tmp / f"g{i:02d}.mp4" for i in range(len(groups))]
    with ThreadPoolExecutor(len(groups)) as ex:
        futs = [ex.submit(render_group, [shots[i] for i in g], C, fps, W, H, str(f), "push" in layers and has)
                for g, f in zip(groups, files)]
        for fu in futs:
            fu.result()
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{f}'\n" for f in files))
    silent = tmp / "silent.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(silent)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", m["audio"], "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-t", f"{total / fps:.4f}",
                    "-movflags", "+faststart", str(out)], check=True)
    print(f"✓ {out}  {len(shots)} 个镜头 · {total} 帧 · {total / fps:.2f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description="小耳音乐卡点MV · 出片")
    ap.add_argument("plan")
    ap.add_argument("-o", "--out")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--preview", action="store_true", help="半分辨率快速预览")
    ap.add_argument("--layers", default=",".join(ALL_LAYERS), help="分轨驱动哪些画面层，逗号分隔；空字符串 = 全关")
    a = ap.parse_args(argv)
    out = a.out or str(Path(a.plan).with_name("mv.mp4"))
    render(a.plan, out, a.jobs, a.preview, [x for x in a.layers.split(",") if x])


if __name__ == "__main__":
    main()
