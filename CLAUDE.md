# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tts-cli is a local offline text-to-speech CLI tool that converts text to speech with subtitle generation. It uses ChatTTS for speech synthesis and Whisper for subtitle generation. Supports Chinese and English.

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
tts-cli --text "Hello" --skip-subtitles
python -m tts_cli --text "Hello" --skip-subtitles
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
├── tts.py              # ChatTTS wrapper with GPU batch processing
├── subtitle.py         # Whisper subtitle generation
├── text_processor.py   # Text splitting and normalization
├── audio.py            # Audio processing utilities
└── utils.py            # Common utilities
```

## Key Architecture Decisions

### Long Text Splitting (text_processor.py)
ChatTTS produces lower quality audio for long texts. The system intelligently splits text:
- Default max_length: 800 characters per chunk
- Priority: paragraphs → sentence boundaries → force split
- Supports both English (. ! ?) and Chinese (。！？) punctuation
- Short tail chunks (< 300 chars) merged with previous to avoid quality issues

### GPU Parallel Processing (tts.py)
For long texts split into multiple chunks:
- `calculate_optimal_batch_size()` dynamically determines batch size based on GPU memory
- Sentences processed in parallel batches for maximum throughput
- Reserved 1GB for model and system, rest for inference

### Audio Merging (audio.py)
- Sentence pause: 0.5 seconds
- Paragraph pause: 1.0 seconds
- Final normalization to maximize dynamic range

## CLI Interface

```bash
# Basic usage
tts-cli --text "你好，世界" --output output.wav --subtitle output.srt

# From file with speaker consistency
tts-cli --file article.txt --output story.wav --save-speaker voice.pt
tts-cli --file chapter2.txt --output ch2.wav --speaker voice.pt

# Long text with auto-splitting (recommended for quality)
tts-cli --file novel.txt --output novel.wav --max-length 800

# Audio only (skip subtitle generation)
tts-cli --text "Quick test" --skip-subtitles
```

## Dependencies

Required:
- ChatTTS, torch, numpy, scipy

Optional:
- openai-whisper (for subtitle generation)
- numba (for audio processing optimization)
