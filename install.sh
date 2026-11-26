#!/usr/bin/env bash
# 小耳音乐卡点MV · 一键安装：建 Python 环境、装依赖（Demucs / librosa / OpenCV），检查 ffmpeg
# 可以重复跑：已有 .venv 就沿用，只补装依赖。想彻底重建：先删掉 .venv 再跑。
set -e
D="$(cd "$(dirname "$0")" && pwd)"
command -v ffmpeg >/dev/null || { echo "✗ 需要 ffmpeg：macOS 上 brew install ffmpeg，Ubuntu 上 sudo apt install ffmpeg"; exit 1; }
if command -v uv >/dev/null; then
  [ -x "$D/.venv/bin/python" ] || uv venv -q "$D/.venv" --python 3.11
  uv pip install -q --python "$D/.venv/bin/python" -r "$D/requirements.txt"
else
  [ -x "$D/.venv/bin/python" ] || python3 -m venv "$D/.venv"
  "$D/.venv/bin/pip" install -q -r "$D/requirements.txt"
fi
"$D/.venv/bin/python" -c "
import librosa, demucs, cv2, torch
gpu = 'CUDA（' + torch.cuda.get_device_name(0) + '）' if torch.cuda.is_available() else ('MPS（Apple）' if torch.backends.mps.is_available() else '无，分轨走 CPU（慢）')
print('✓ 环境就绪 · torch', torch.__version__, '· GPU:', gpu)"
echo "试一下：$D/beatsync.sh --help"
