# tts-cli

tts-cli 是一个本地离线使用的文字转语音命令行工具，基于 ChatTTS 和 Whisper。

## 特性

- 完全本地运行，无需网络，保障隐私
- 支持中文和英文语音合成
- 自动生成 SRT 字幕文件
- 智能文本分割，长文本自动分段处理
- 支持保存和复用说话人音色

## 系统要求

- Python 3.10+
- CUDA GPU（推荐，CPU 也可运行但较慢）
- 显存 4GB+（推荐 8GB+）

## 安装

### 1. 安装 tts-cli

```bash
# 从 PyPI 安装
pip install tts-cli

# 或从源码安装
git clone https://github.com/yahaa/tts-cli.git
cd tts-cli
pip install -e .
```

### 2. 安装 ChatTTS 模型

ChatTTS 模型会在首次运行时自动从 HuggingFace 下载（约 2GB）。

**自动下载**（推荐）：
```bash
# 首次运行会自动下载模型到 ~/.cache/huggingface/hub/models--2Noise--ChatTTS/
tts-cli --text "测试" --output test.wav --skip-subtitles
```

**手动下载**（如果自动下载失败）：
```bash
# 方法 1: 使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download 2Noise/ChatTTS --local-dir ~/.cache/huggingface/hub/models--2Noise--ChatTTS

# 方法 2: 使用 Git LFS
git lfs install
git clone https://huggingface.co/2Noise/ChatTTS ~/.cache/huggingface/hub/models--2Noise--ChatTTS
```

**国内镜像**（如果 HuggingFace 访问慢）：
```bash
# 设置镜像环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 然后运行 tts-cli，会从镜像下载
tts-cli --text "测试" --output test.wav --skip-subtitles
```

### 3. 安装 Whisper（可选，用于字幕生成）

```bash
pip install openai-whisper
```

## 使用

### 基本用法

```bash
# 文本转语音
tts-cli --text "你好，欢迎使用 tts-cli。" --output output.wav

# 从文件读取文本
tts-cli --file article.txt --output output.wav

# 只生成音频，跳过字幕
tts-cli --text "Hello world" --output output.wav --skip-subtitles
```

### 高级用法

```bash
# 保存说话人音色（便于复用）
tts-cli --text "测试音色" --output test.wav --save-speaker my_voice.pt

# 使用已保存的音色
tts-cli --file novel.txt --output novel.wav --speaker my_voice.pt

# 调整语速（0-9，默认 3）
tts-cli --text "快速播放" --output fast.wav --speed 5

# 指定语言（en/zh）
tts-cli --file english.txt --output en.wav --language en

# 静默模式（只输出文件路径）
tts-cli --file text.txt --output out.wav --quiet
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--text` | 要转换的文本内容 | - |
| `--file` / `--input` | 输入文本文件路径 | - |
| `--output` | 输出音频文件路径 | output.wav |
| `--subtitle` | 输出字幕文件路径 | 自动派生 |
| `--speed` | 语速 0-9 | 3 |
| `--language` | 语言 en/zh | en |
| `--speaker` | 说话人音色文件 (.pt) | 随机 |
| `--save-speaker` | 保存当前音色到文件 | - |
| `--max-length` | 文本分块最大长度 | 500 |
| `--max-batch` | 批处理大小 | 1 |
| `--skip-subtitles` | 跳过字幕生成 | false |
| `--no-normalize` | 禁用文本规范化 | false |
| `--whisper-model` | Whisper 模型大小 | base |
| `--quiet` | 静默模式 | false |

## 常见问题

### Q: 模型下载很慢怎么办？

设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 出现 "found invalid characters" 错误？

tts-cli 会自动规范化文本，但如果仍有问题，检查文本是否包含特殊字符。ChatTTS 只支持：
- 英文字母 a-z, A-Z
- 中文字符
- 标点符号：。，！
- 空格

### Q: 音频时长很短，不完整？

这可能是因为某些文本块生成失败。尝试：
1. 使用 `--max-length 300` 减小分块大小
2. 检查文本是否有特殊字符

### Q: GPU 显存不足？

- 减小 `--max-length` 参数
- 使用 `--whisper-model tiny` 减小 Whisper 模型

## License

AGPLv3+
