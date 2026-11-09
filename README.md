# 小耳音乐卡点MV · Xiaoer BeatSync MV

给一段音乐 + 一组画面素材，做一支**画面跟着音符长**的卡点 MV —— 卡点精确到帧，并从成片逐帧体检。

> AI 管「画面怎么长」，音乐管「什么时候长、长多快」。

```bash
./install.sh
./beatsync.sh analyze 歌.mp3 -o work/歌.cues.json          # Demucs 分轨 + pdoom 卡点
./beatsync.sh kells work/歌.cues.json 片段*.mp4 --audio 歌.mp3 -o work/mv.mp4   # 凯尔经模式出片
./beatsync.sh make 歌.mp3 素材文件夹/ -o work/              # 剪辑模式：出片 + 导出时间线 + 体检
```

- **凯尔经模式**：一组首尾帧图生视频做的生长/变形片段 → 自动编排 + 音乐推播放头（一个起音 = 同一帧一个动作）
- **剪辑模式**：普通素材按「段落 / 乐句 / 拍」三层规则剪，每一刀写明为什么；导出 Premiere / DaVinci（XML）、Final Cut（FCPXML）
- **体检**：从成片 mp4 逐帧量每一刀差几帧、起音是否同帧出动作、音画偏移

给 AI agent（Claude Code / Codex）用：读 `SKILL.md`。准确度测试：`tests/eval_synth.py`。
来源与许可：`SKILL.md` 末尾、`LICENSES/`。
