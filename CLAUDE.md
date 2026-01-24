# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tts-cli is a local offline text-to-speech CLI tool that converts text to speech with subtitle generation. It uses Qwen3-TTS for speech synthesis and Whisper for subtitle generation. Supports voice cloning, 9 premium preset voices, and natural language voice design. Supports 10 languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian).

## Build and Development Commands

```bash
# Install development dependencies
make install

# Format code (Ruff)
make format

# Check code (lint)
make lint

# Run tests
make test

# Run the CLI
tts-cli --text "Hello" --speaker Ryan --skip-subtitles
python -m tts_cli --text "Hello" --speaker Ryan --skip-subtitles
```

## Code Style

This project uses **Ruff** for code formatting and linting:
- Line length: 88 characters
- Auto-fix unused imports on save
- Auto-sort imports (isort style)

Configuration is in `pyproject.toml` under `[tool.ruff]`.

### VSCode Setup

Recommended extensions (configured in `.vscode/settings.json`):
- `ms-python.python` - Python support
- `ms-python.vscode-pylance` - IntelliSense (type checking disabled)
- `charliermarsh.ruff` - Formatting and linting

Do NOT use: Pylint, autopep8, Black, isort, flake8 (redundant with Ruff)

## Project Structure

```
src/tts_cli/
├── __init__.py         # Package exports and version
├── __main__.py         # python -m tts_cli entry point
├── cli.py              # CLI argument parsing
├── core.py             # Main workflow orchestration
├── tts.py              # Qwen3-TTS wrapper with GPU batch processing
├── voice.py            # Voice cloning/design/custom management
├── subtitle.py         # Whisper subtitle generation
├── text_processor.py   # Text splitting and normalization
├── audio.py            # Audio processing utilities
└── utils.py            # Common utilities
```

## Key Architecture Decisions

### Voice Modes (voice.py)
Three voice modes supported:
1. **Custom Voice**: Use 9 premium speakers (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee)
2. **Voice Design**: Natural language descriptions (e.g., "warm elderly male voice")
3. **Voice Clone**: Clone from 3-second reference audio + transcript

Voice prompts can be saved to `.qwen-voice` files for reuse across sessions.

### Long Text Splitting (text_processor.py)
Qwen3-TTS handles longer texts than ChatTTS but still needs splitting for GPU memory management:
- Default max_length: 1000 characters per chunk (increased from 500)
- Priority: paragraphs → sentence boundaries → force split
- Supports both English (. ! ?) and Chinese (。！？) punctuation
- Short tail chunks (< 300 chars) merged with previous to avoid quality issues
- Simplified normalization: Unicode NFKC normalization only (no character filtering)

### GPU Parallel Processing (tts.py)
For long texts split into multiple chunks:
- `calculate_optimal_batch_size()` dynamically determines batch size based on GPU memory
- Chunks processed in parallel batches for maximum throughput (default batch_size: 4)
- Model memory: 1.7B ~3GB, 0.6B ~1.5GB
- Reserved 1GB for model and system, rest for inference

### Audio Merging (audio.py)
- Sentence pause: 0.5 seconds
- Paragraph pause: 1.0 seconds
- Final normalization to maximize dynamic range

## CLI Interface

```bash
# Basic usage with preset voice
tts-cli --text "Hello, world" --speaker Ryan --output output.wav

# Voice cloning from reference audio
tts-cli --mode clone --text "New content" \
  --reference-audio voice_sample.wav \
  --reference-text "Reference transcript" \
  --save-speaker my_voice.qwen-voice

# Reuse saved voice
tts-cli --file chapter2.txt --speaker my_voice.qwen-voice --output ch2.wav

# Voice design with natural language
tts-cli --mode design \
  --voice-description "warm elderly male voice" \
  --text "Good evening" --output speech.wav

# Long text with auto-splitting (recommended for quality)
tts-cli --file novel.txt --speaker Vivian --output novel.wav --max-length 1000

# Multi-language support
tts-cli --text "你好，世界" --speaker Vivian --language zh --output hello_zh.wav
tts-cli --text "Bonjour" --speaker Ryan --language fr --output hello_fr.wav

# Audio only (skip subtitle generation)
tts-cli --text "Quick test" --speaker Ryan --skip-subtitles
```

## Dependencies

Required:
- qwen-tts, torch, numpy, scipy

Optional:
- openai-whisper (for subtitle generation)
- numba (for audio processing optimization)
- flash-attn (for FlashAttention 2 performance optimization)

## Model Variants

- **1.7B-CustomVoice**: 9 premium timbres with style control
- **1.7B-VoiceDesign**: Natural-language voice descriptions
- **1.7B-Base**: 3-second rapid voice cloning (default)
- **0.6B-CustomVoice**: Lightweight version with presets
- **0.6B-Base**: Minimal resource voice cloning

Default: 1.7B-Base (supports voice cloning while being the most versatile)
