# tts-cli
tts-cli 是一个本地离线使用的文字转语音的工具

## 目标
- 提供一个简单易用的命令行工具，用于将文本转换为语音，并支持输出对应的字幕文件
- 语音合成过程完全在本地进行，无需依赖网络服务，保障用户隐私
- 语音合成符合人类自然发音习惯，停顿自然流畅，段落分明，适合长文本阅读，如小说、文章等
- 支持中文和英文的文字转语音合成

## 技术调研
- ChatTTS: 基于大型语言模型的文本到语音合成系统，能够生成高质量的语音输出
- 参考 demo: https://github.com/yahaa/ChatTTS/blob/feat/new-tts-cli/tts_cli.py

## 安装
```bash
pip install tts-cli
```
## 使用
```bash
tts-cli --text "你好，欢迎使用 tts-cli。" --output output.wav --subtitle output.srt --speaker my_speaker.bt

tts-cli --file input-text.file --output output_en.wav --subtitle output_en.srt --speaker my_speaker.bt
``` 
## 参数说明
- `--text`: 要转换为语音的文本内容
- `--output`: 输出的音频文件路径
- `--subtitle`: 输出的字幕文件路径
- `--file`: 包含要转换文本的文件路径
- `--speaker`: 语音合成所使用的说话人模型文件
