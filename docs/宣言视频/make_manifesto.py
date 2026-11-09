#!/usr/bin/env python3
"""小耳音乐卡点MV · 宣言视频（约 110 秒）
底片 = 本工具全自动出的凯尔经 MV；高潮段左右分屏对比手排 v7；右下角四轨实时跳动（来自 cues.json）；
字幕卡落在乐句交界处。"""
import json
import subprocess
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

K = Path.home() / "projects/凯尔经MV-源素材"
AUTO = K / "beatsync/自动版-skill全自动.mp4"
V7 = Path("/Users/jane/Desktop/works/代码片场-英文歌MV/成片-v7-全曲/凯尔经-What Are You Doing In My Forest-v7.mp4")
SONG = K / "kells/public/audio/forest.wav"
CUES = json.loads((K / "beatsync/forest.cues.json").read_text())
CHECK = Path.home() / "projects/xiaoer-beatsync/docs/说明书/assets/checkup.png"
OUT = Path(__file__).with_name("小耳音乐卡点MV-宣言视频.mp4")
W, H, FPS, DUR = 1920, 1080, 30, 110.0
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"


def font(sz):
    return ImageFont.truetype(FONT, sz)


def text_layer(lines, sizes, pos="bl", pad=96, gap=14, shade=True):
    """白字 + 柔和阴影，返回 (RGB, alpha) 两个 float32 数组。"""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    fs = [font(s) for s in sizes]
    hs = [d.textbbox((0, 0), ln, font=f)[3] for ln, f in zip(lines, fs)]
    total = sum(hs) + gap * (len(lines) - 1)
    y = H - pad - total if pos == "bl" else (H - total) // 2
    for ln, f, h in zip(lines, fs, hs):
        w = d.textbbox((0, 0), ln, font=f)[2]
        x = pad if pos == "bl" else (W - w) // 2
        if shade:
            for dx, dy in ((0, 3), (2, 2), (0, 0)):
                pass
            sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(sh).text((x + 2, y + 3), ln, font=f, fill=(0, 0, 0, 170))
            im = Image.alpha_composite(sh, im)
            d = ImageDraw.Draw(im)
        d.text((x, y), ln, font=f, fill=(255, 250, 238, 255))
        y += h + gap
    a = np.asarray(im).astype(np.float32) / 255.0
    sh_blur = cv2.GaussianBlur(a[..., 3], (0, 0), 6) * 0.55     # 文字底下一点暗晕，保证可读
    return a[..., :3][..., ::-1] * 255.0, a[..., 3:4], sh_blur[..., None]


# 字幕卡：(起, 止, 行, 字号, 位置)
CARDS = [
    (0.6, 6.4, ["小耳音乐卡点MV", "让每一帧，都成为音乐的函数"], [88, 40], "c"),
    (6.9, 12.8, ["传统的卡点，", "只是一个节拍换一个画面"], [52, 52], "bl"),
    (13.2, 19.1, ["我们把歌拆成四轨", "鼓 · 贝斯 · 人声 · 旋律"], [52, 40], "bl"),
    (19.5, 25.4, ["每一个起音，", "在同一帧推动画面生长"], [52, 52], "bl"),
    (25.9, 33.0, ["AI 管画面怎么长", "音乐管什么时候长、长多快"], [52, 52], "bl"),
    (33.9, 37.6, ["音乐停下，画面也停下", "等它回来"], [52, 40], "bl"),
    (40.0, 48.8, ["编排规则来自导演的判断", "气口不切 · 变形迎上新段落 · 高潮加密"], [52, 36], "bl"),
    (49.5, 51.8, ["全自动 vs 导演手排定稿"], [56], "bl"),
    (74.2, 81.8, ["16 个切点，11 个在 1 拍以内", "开头 5 个与手排相差 0–2 帧"], [48, 36], "bl"),
    (82.5, 91.3, ["成片逐帧体检", "切点 100% 落在计划的那一帧 · 起音即动作 98%"], [52, 34], "bl"),
    (91.8, 100.8, ["一首 3 分钟的歌", "分轨 30 秒 · 1080p 出片约 1 分钟 · 不需要付费 API"], [52, 34], "bl"),
    (101.5, 110.0, ["小耳音乐卡点MV", "github.com/Jane-xiaoer/xiaoer-beatsync-mv", "小耳 · AI 艺术家 / AI Builder"], [80, 34, 30], "c"),
]
CREDIT = (0.6, 6.4, ["示意案例：《凯尔经的秘密》主题曲 What Are You Doing In My Forest · 画面 MJ + 首尾帧图生视频"], [22])
SPLIT = (52.0, 74.0)
CHECK_SPAN = (82.3, 91.5)
METER_SPAN = (13.0, 49.0)


def fade(t, a, b, f=0.5):
    if t < a or t > b:
        return 0.0
    return float(min(1.0, (t - a) / f, (b - t) / f))


def reader(path, start=0.0):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(path), "-vf",
                          f"fps={FPS},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE)
    while True:
        raw = p.stdout.read(W * H * 3)
        if len(raw) < W * H * 3:
            yield None
            continue
        yield np.frombuffer(raw, np.uint8).reshape(H, W, 3)


def meter_data():
    st = CUES["stems"]
    rate = st["env"]["rate"]
    env = {k: np.array(st["env"][k]) for k in ("drums", "bass", "vocals", "other")}
    on = {"drums": [o[:2] for o in st["onsets"]["kick"] + st["onsets"]["snare"]],
          "bass": [o[:2] for o in st["onsets"]["bass"]], "vocals": [o[:2] for o in st["onsets"]["vocals"]],
          "other": [o[:2] for o in st["onsets"]["other"]]}
    return rate, env, on


def draw_meter(img, t, rate, env, on, alpha):
    """右下角四轨：每轨一条能量条 + 起音那一帧亮点（0.15s 回落）。"""
    x0, y0, w, rh = W - 560, H - 250, 470, 38
    panel = img.copy()
    cv2.rectangle(panel, (x0 - 24, y0 - 34), (x0 + w + 24, y0 + 4 * rh + 20), (20, 24, 22), -1)
    img[:] = cv2.addWeighted(panel, 0.55 * alpha, img, 1 - 0.55 * alpha, 0)
    names = [("drums", "鼓", (90, 170, 255)), ("bass", "贝斯", (120, 210, 150)),
             ("vocals", "人声", (240, 190, 120)), ("other", "旋律", (230, 140, 220))]
    lbl = Image.fromarray(np.zeros((4 * rh, 90, 4), np.uint8))
    d = ImageDraw.Draw(lbl)
    for i, (_, cn, _) in enumerate(names):
        d.text((0, i * rh + 6), cn, font=font(22), fill=(255, 250, 238, int(255 * alpha)))
    la = np.asarray(lbl).astype(np.float32) / 255
    sub = img[y0:y0 + 4 * rh, x0 - 10:x0 + 80].astype(np.float32)
    img[y0:y0 + 4 * rh, x0 - 10:x0 + 80] = (sub * (1 - la[..., 3:]) + la[..., :3][..., ::-1] * 255 * la[..., 3:]).astype(np.uint8)
    for i, (k, _, col) in enumerate(names):
        y = y0 + i * rh + 18
        v = float(np.interp(t, np.arange(len(env[k])) / rate, env[k]))
        pulse = 0.0
        for t0, s in on[k]:
            if 0 <= t - t0 < 0.6:
                pulse = max(pulse, s * np.exp(-(t - t0) / 0.15))
        bw = int((w - 110) * min(1.0, v * 1.6))
        c = tuple(int(cc * alpha + 40 * (1 - alpha)) for cc in col)
        cv2.rectangle(img, (x0 + 90, y - 6), (x0 + 90 + max(2, bw), y + 6), c, -1)
        r = int(6 + 10 * pulse)
        cv2.circle(img, (x0 + w, y), r, tuple(int(min(255, cc + 80 * pulse)) for cc in c), -1)


def main():
    cards = [(a, b, text_layer(l, s, p)) for a, b, l, s, p in CARDS]
    credit = (CREDIT[0], CREDIT[1], text_layer(CREDIT[2], CREDIT[3], "bl", pad=40))
    chk = cv2.imread(str(CHECK))
    ch_w = 1200
    chk = cv2.resize(chk, (ch_w, int(chk.shape[0] * ch_w / chk.shape[1])), interpolation=cv2.INTER_AREA)
    rate, env, on = meter_data()
    A, B = reader(AUTO), reader(V7)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-ss", "0", "-t", str(DUR), "-i", str(SONG),
                            "-af", f"afade=t=out:st={DUR - 3.5}:d=3.5", "-map", "0:v", "-map", "1:a",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(OUT)],
                           stdin=subprocess.PIPE)
    lab_a = text_layer(["skill 全自动"], [34], "bl", pad=40)
    lab_b = text_layer(["导演手排 v7"], [34], "bl", pad=40)
    n = int(DUR * FPS)
    for f in range(n):
        t = f / FPS
        a, b = next(A), next(B)
        img = a.copy().astype(np.float32)
        # 片头片尾压暗
        dim = max(fade(t, 0.0, 6.6, 0.6) * 0.45, fade(t, 101.2, DUR + 1, 1.2) * 0.6)
        if dim:
            img *= (1 - dim)
        # 高潮分屏：左自动、右手排
        s = fade(t, *SPLIT, f=0.6)
        if s and b is not None:
            half = W // 2
            la = cv2.resize(a[:, W // 4: W // 4 + half], (half, H)).astype(np.float32)
            rb = cv2.resize(b[:, W // 4: W // 4 + half], (half, H)).astype(np.float32)
            sp = np.concatenate([la, rb], axis=1)
            cv2.line(sp, (half, 0), (half, H), (240, 235, 220), 2)
            img = img * (1 - s) + sp * s
            for lay, xoff in ((lab_a, 0), (lab_b, half)):
                rgb, al, sh = lay
                rgb2 = np.roll(rgb, xoff, axis=1); al2 = np.roll(al, xoff, axis=1)
                img = img * (1 - al2 * s) + rgb2 * al2 * s
        # 体检报告卡片
        c = fade(t, *CHECK_SPAN, f=0.5)
        if c:
            y0, x0 = 120, (W - ch_w) // 2
            hh = min(chk.shape[0], H - 380)
            region = img[y0:y0 + hh, x0:x0 + ch_w]
            img[y0:y0 + hh, x0:x0 + ch_w] = region * (1 - c) + chk[:hh].astype(np.float32) * c
        # 字幕
        for a0, b0, (rgb, al, sh) in cards + [credit]:
            k = fade(t, a0, b0)
            if k:
                img = img * (1 - sh * k) + 0 * sh
                img = img * (1 - al * k) + rgb * al * k
        out = img.clip(0, 255).astype(np.uint8)
        m = fade(t, *METER_SPAN, f=0.8)
        if m:
            draw_meter(out, t, rate, env, on, m)
        enc.stdin.write(out.tobytes())
        if f % 300 == 0:
            print(f"  {t:5.1f}s / {DUR}s", flush=True)
    enc.stdin.close()
    enc.wait()
    print("✓", OUT)


if __name__ == "__main__":
    main()
