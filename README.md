# tts-cli

tts-cli 是一个本地离线使用的文字转语音命令行工具，基于 Qwen3-TTS 和 Whisper。

## ⚠️ v0.2.0 重大更新

**底层引擎已从 ChatTTS 升级到 Qwen3-TTS**，带来诸多新特性但包含破坏性变更。

- 🔴 **重要**: `.pt` 音色文件不再兼容，请查看 [迁移指南](MIGRATION.md)
- ✅ 新增声音克隆功能（3 秒音频即可克隆）
- ✅ 9 种高级预设音色
- ✅ 支持 10 种语言
- ✅ 更长文本处理能力（1000 字符/块）

详细迁移说明请查看: **[MIGRATION.md](MIGRATION.md)**

## 特性

- 🎯 **声音克隆**: 从 3 秒参考音频克隆任意声音
- 🎵 **9 种预设音色**: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
- 🎨 **自然语言声音设计**: 用自然语言描述想要的声音（如"温暖的老年男声"）
- 🌍 **多语言支持**: 支持中文、英文、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语
- 📝 **自动字幕生成**: 使用 Whisper 生成 SRT 字幕文件
- 📄 **智能文本分割**: 长文本自动分段处理（最大 1000 字符/块）
- 🔒 **完全本地运行**: 无需网络，保障隐私
- 🚀 **HTTP API Server 模式**: 支持异步任务处理，适合服务化部署

## 快速开始

```bash
# 安装
pip install tts-cli

# 使用预设音色生成语音
tts-cli --text "你好，世界" --speaker Vivian --output hello.wav

# 克隆你自己的声音
tts-cli --mode clone \
  --reference-audio my_voice.wav \
  --reference-text "这是我的声音样本" \
  --text "测试克隆的声音" \
  --save-speaker my_voice.qwen-voice

# 使用克隆的声音
tts-cli --text "新的内容" --speaker my_voice.qwen-voice --output new.wav
```

## 系统要求

- Python 3.10+
- CUDA GPU（推荐，CPU 也可运行但较慢）
- 显存 4GB+（推荐 8GB+ 用于 1.7B 模型）

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

### 2. 安装 Qwen3-TTS 模型

Qwen3-TTS 模型会在首次运行时自动从 HuggingFace 下载（约 3-5GB）。

**自动下载**（推荐）：
```bash
# 首次运行会自动下载模型
tts-cli --text "测试" --speaker Ryan --output test.wav --skip-subtitles
```

**手动下载**（如果自动下载失败）：
```bash
# 方法 1: 使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base

# 方法 2: 使用 Git LFS
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

**国内镜像**（如果 HuggingFace 访问慢）：
```bash
# 设置镜像环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 然后运行 tts-cli，会从镜像下载
tts-cli --text "测试" --speaker Ryan --output test.wav --skip-subtitles
```

**可用模型变体**:
- `1.7B-Base`: 默认，支持声音克隆（推荐）
- `1.7B-CustomVoice`: 9 种预设音色 + 风格控制
- `1.7B-VoiceDesign`: 自然语言声音设计
- `0.6B-Base`: 轻量版声音克隆
- `0.6B-CustomVoice`: 轻量版预设音色

### 3. 安装可选依赖

```bash
# 字幕生成（推荐）
pip install openai-whisper

# 性能优化：numba（推荐）
pip install numba

# 高级性能优化：FlashAttention 2（可选，需要 CUDA）
# 注意：flash-attn 安装可能失败，需要特定的 CUDA 版本和编译环境
pip install flash-attn --no-build-isolation
```

## 使用

### 基本用法

```bash
# 使用预设音色（推荐新手）
tts-cli --text "你好，欢迎使用 tts-cli" --speaker Vivian --output output.wav

# 从文件读取文本
tts-cli --file article.txt --speaker Ryan --output output.wav

# 只生成音频，跳过字幕
tts-cli --text "Hello world" --speaker Ryan --output output.wav --skip-subtitles

# 多语言支持
tts-cli --text "Bonjour le monde" --speaker Ryan --language fr --output french.wav
```

### 声音克隆

```bash
# 从参考音频克隆声音并保存
tts-cli --mode clone \
  --reference-audio voice_sample.wav \
  --reference-text "这是我的声音样本" \
  --text "测试克隆的声音" \
  --save-speaker my_voice.qwen-voice \
  --output test.wav

# 使用已保存的克隆声音
tts-cli --file novel.txt --speaker my_voice.qwen-voice --output novel.wav
```

### 声音设计

```bash
# 用自然语言描述想要的声音
tts-cli --mode design \
  --voice-description "温暖的老年男性声音，带有轻微的英国口音" \
  --text "晚上好，女士们先生们" \
  --output speech.wav
```

### 高级用法

```bash
# 调整语速（0-9，默认 3）
tts-cli --text "快速播放" --speaker Ryan --output fast.wav --speed 7

# 添加说话风格指令
tts-cli --text "我太兴奋了！" --speaker Aiden \
  --instruct "speak with enthusiasm" --output excited.wav

# 处理长文本（自动分块）
tts-cli --file long_article.txt --speaker Vivian \
  --max-length 1000 --max-batch 4 --output long.wav

# 静默模式（只输出文件路径）
tts-cli --file text.txt --speaker Ryan --output out.wav --quiet
```

## 参数说明

### 基础参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--text` | 要转换的文本内容 | - |
| `--file` / `--input` | 输入文本文件路径 | - |
| `--output` | 输出音频文件路径 | output.wav |
| `--subtitle` | 输出字幕文件路径 | 自动派生 |
| `--skip-subtitles` | 跳过字幕生成 | false |
| `--quiet` | 静默模式（只输出文件路径） | false |

### 声音参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 声音模式：`custom`/`design`/`clone` | custom |
| `--speaker` | 预设音色名称 或 音色文件 (.qwen-voice) | Ryan |
| `--voice-description` | 自然语言声音描述（design 模式） | - |
| `--reference-audio` | 参考音频文件（clone 模式） | - |
| `--reference-text` | 参考音频转录文本（clone 模式） | - |
| `--save-speaker` | 保存克隆声音到文件 | - |
| `--instruct` | 说话风格指令 | - |

### 处理参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--speed` | 语速 0-9（低速用 0-2，高速用 7-9） | 3 |
| `--language` | 语言代码（zh/en/ja/ko/de/fr/ru/pt/es/it/auto） | auto |
| `--max-length` | 文本分块最大长度 | 1000 |
| `--max-batch` | 批处理大小 | 4 |
| `--whisper-model` | Whisper 模型大小（tiny/base/small/medium/large） | base |

### 可用预设音色

| 音色名称 | 语言 | 性别 | 特点 |
|---------|------|------|------|
| Vivian | 中文 | 女 | 清晰、专业 |
| Serena | 中文 | 女 | 温暖、友好 |
| Uncle_Fu | 中文 | 男 | 成熟、权威 |
| Dylan | 中文（北京） | 男 | 年轻、有活力 |
| Eric | 中文（四川） | 男 | 独特方言 |
| Ryan | 英语 | 男 | 清晰、中性 |
| Aiden | 英语 | 男 | 专业、温暖 |
| Ono_Anna | 日语 | 女 | 清晰、专业 |
| Sohee | 韩语 | 女 | 清晰、友好 |

## Server 模式

tts-cli 支持 HTTP API Server 模式，提供异步任务处理能力，适合服务化部署。

### 安装 Server 依赖

```bash
# 安装 server 相关依赖
pip install tts-cli[server]

# 或从源码安装
pip install -e ".[server]"
```

### 启动 Server

```bash
# 使用独立命令启动（推荐）
tts-server --port 8000 --mongodb-uri mongodb://localhost:27017

# 或使用子命令启动
tts-cli serve --port 8000 --mongodb-uri mongodb://localhost:27017
```

### API 文档

启动服务后，可以通过以下地址访问 API 文档：

| 地址 | 说明 |
|------|------|
| http://localhost:8000/docs | Swagger UI 交互式文档 |
| http://localhost:8000/redoc | ReDoc 文档 |
| http://localhost:8000/openapi.json | OpenAPI JSON Schema |

### Server 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 绑定主机地址 | 0.0.0.0 |
| `--port` | 绑定端口 | 8000 |
| `--mongodb-uri` | MongoDB 连接 URI | mongodb://localhost:27017 |
| `--db-name` | MongoDB 数据库名称 | tts-server |
| `--output-dir` | 生成文件存储目录 | ./tts_output |
| `--workers` | 后台工作线程数 | 1 |
| `--cleanup-hours` | 任务保留时长（小时） | 24 |
| `--log-level` | 日志级别 | info |

### API 接口

#### 1. 创建 TTS 任务

**POST** `/api/v1/create_tts_task`

创建一个异步 TTS 任务，立即返回任务 ID。

**请求示例：**
```bash
# 使用预设音色
curl -X POST http://localhost:8000/api/v1/create_tts_task \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，欢迎使用 tts-cli 服务。",
    "mode": "custom",
    "speaker": "Vivian",
    "language": "zh",
    "speed": 3,
    "skip_subtitles": false
  }'

# 使用声音克隆
curl -X POST http://localhost:8000/api/v1/create_tts_task \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是克隆声音的测试",
    "mode": "clone",
    "reference_audio_url": "https://example.com/voice.wav",
    "reference_text": "参考音频的转录文本",
    "skip_subtitles": true
  }'
```

**请求参数：**

| 参数 | 类型 | 必填 | 说明 | 默认值 |
|------|------|------|------|--------|
| `text` | string | 是 | 要转换的文本（最大 100,000 字符） | - |
| `mode` | string | 否 | 声音模式：`custom`/`design`/`clone` | custom |
| `speaker` | string | 否 | 预设音色名称（custom 模式） | Ryan |
| `voice_description` | string | 否 | 声音描述（design 模式） | null |
| `reference_audio_url` | string | 否 | 参考音频 URL（clone 模式） | null |
| `reference_text` | string | 否 | 参考音频转录（clone 模式） | null |
| `language` | string | 否 | 语言代码（zh/en/ja/ko/de/fr/ru/pt/es/it/auto） | auto |
| `speed` | int | 否 | 语速 0-9 | 3 |
| `instruct` | string | 否 | 说话风格指令 | null |
| `max_length` | int | 否 | 文本分块最大长度 | 1000 |
| `max_batch` | int | 否 | 批处理大小 | 4 |
| `skip_subtitles` | bool | 否 | 是否跳过字幕生成 | false |
| `whisper_model` | string | 否 | Whisper 模型大小 | base |
| `callback_url` | string | 否 | 任务完成回调 URL | null |

**响应示例：**
```json
{
  "request_id": "req_abc123",
  "task_id": "task_xyz789",
  "status": "waiting"
}
```

#### 2. 查询任务状态

**GET** `/api/v1/describe_tts_task?task_id={task_id}`

查询任务的处理状态和结果。

**请求示例：**
```bash
curl "http://localhost:8000/api/v1/describe_tts_task?task_id=task_xyz789"
```

**响应示例：**
```json
{
  "request_id": "req_def456",
  "task": {
    "task_id": "task_xyz789",
    "status": "success",
    "audio_url": "/api/v1/files/task_xyz789/output.wav",
    "subtitle_url": "/api/v1/files/task_xyz789/output.srt",
    "duration": 12.5,
    "character_count": 256,
    "error_code": null,
    "error_message": null,
    "create_time": "2024-01-10T10:00:00Z",
    "start_time": "2024-01-10T10:00:01Z",
    "finish_time": "2024-01-10T10:00:15Z"
  }
}
```

**任务状态说明：**

| 状态 | 说明 |
|------|------|
| `waiting` | 任务已创建，等待处理 |
| `processing` | 任务正在处理中 |
| `success` | 任务处理成功 |
| `failed` | 任务处理失败 |

#### 3. 下载文件

**GET** `/api/v1/files/{task_id}/{filename}`

下载生成的音频或字幕文件。

**请求示例：**
```bash
# 下载音频文件
curl -O http://localhost:8000/api/v1/files/task_xyz789/output.wav

# 下载字幕文件
curl -O http://localhost:8000/api/v1/files/task_xyz789/output.srt
```

#### 4. 健康检查

**GET** `/api/v1/health`

检查服务健康状态。

**请求示例：**
```bash
curl http://localhost:8000/api/v1/health
```

**响应示例：**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "0.1.0",
  "mongodb_connected": true
}
```

### 回调通知

如果创建任务时指定了 `callback_url`，任务完成后会向该 URL 发送 POST 请求：

```json
{
  "task_id": "task_xyz789",
  "status": "success",
  "audio_url": "/api/v1/files/task_xyz789/output.wav",
  "subtitle_url": "/api/v1/files/task_xyz789/output.srt",
  "duration": 12.5,
  "error_code": null,
  "error_message": null,
  "finish_time": "2024-01-10T10:00:15Z"
}
```

### 完整使用示例

```bash
# 1. 启动 MongoDB（如果还没有）
docker run -d -p 27017:27017 --name mongodb mongo:latest

# 2. 启动 TTS Server
tts-server --port 8000

# 3. 创建任务
RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/create_tts_task \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test.", "language": "en", "skip_subtitles": true}')

TASK_ID=$(echo $RESPONSE | python -c "import sys, json; print(json.load(sys.stdin)['task_id'])")
echo "Task ID: $TASK_ID"

# 4. 轮询任务状态
while true; do
  STATUS=$(curl -s "http://localhost:8000/api/v1/describe_tts_task?task_id=$TASK_ID" \
    | python -c "import sys, json; print(json.load(sys.stdin)['task']['status'])")
  echo "Status: $STATUS"
  if [ "$STATUS" = "success" ] || [ "$STATUS" = "failed" ]; then
    break
  fi
  sleep 2
done

# 5. 下载音频文件
curl -O "http://localhost:8000/api/v1/files/$TASK_ID/output.wav"
```

### Docker 部署

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
RUN pip install tts-cli[server]

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["tts-server", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  tts-server:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./tts_output:/app/tts_output
    environment:
      - HF_ENDPOINT=https://hf-mirror.com  # 国内镜像（可选）
    command: >
      tts-server
      --host 0.0.0.0
      --port 8000
      --mongodb-uri mongodb://mongodb:27017
      --output-dir /app/tts_output
    depends_on:
      - mongodb

volumes:
  mongodb_data:
```

```bash
# 启动服务
docker-compose up -d
```

## 常见问题

### Q: 我的旧 .pt 音色文件无法使用？

**v0.2.0 版本不再兼容 ChatTTS 的 .pt 文件**。请参考 [迁移指南](MIGRATION.md) 使用声音克隆功能重新创建音色文件。

### Q: 模型下载很慢怎么办？

设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 如何克隆自己的声音？

需要准备：
1. 3-10 秒的清晰参考音频（WAV/MP3/FLAC）
2. 参考音频的准确转录文本

```bash
tts-cli --mode clone \
  --reference-audio my_voice.wav \
  --reference-text "这是我的声音样本" \
  --text "测试内容" \
  --save-speaker my_voice.qwen-voice \
  --output test.wav
```

克隆质量提示：
- ✅ 音频清晰、背景噪音少
- ✅ 转录文本完全匹配音频
- ✅ 单人说话
- ✅ 3-10 秒长度最佳

### Q: 哪种预设音色最适合我？

根据语言和需求选择：
- **中文内容**: Vivian（女声，专业）、Dylan（男声，年轻）、Uncle_Fu（男声，成熟）
- **英文内容**: Ryan（男声，中性）、Aiden（男声，温暖）
- **日语内容**: Ono_Anna（女声）
- **韩语内容**: Sohee（女声）

### Q: 音频时长很短，不完整？

这可能是因为某些文本块生成失败。尝试：
1. 使用 `--max-length 800` 调整分块大小
2. 检查 GPU 显存是否足够

### Q: GPU 显存不足？

解决方法：
- 减小 `--max-batch` 参数（如改为 2 或 1）
- 减小 `--max-length` 参数
- 使用 `--whisper-model tiny` 减小 Whisper 模型
- 使用轻量版模型（0.6B）

### Q: 支持哪些语言？

支持 10 种语言：中文（zh）、英语（en）、日语（ja）、韩语（ko）、德语（de）、法语（fr）、俄语（ru）、葡萄牙语（pt）、西班牙语（es）、意大利语（it）

使用 `--language auto` 可自动检测语言。

### Q: 如何安装 FlashAttention 2 以获得更好的性能？

FlashAttention 2 可以减少 GPU 显存使用并提高推理速度，但**不是必需的**。安装方法：

```bash
# 方法 1：先安装 tts-cli，再单独安装 flash-attn
pip install tts-cli
pip install flash-attn --no-build-isolation

# 方法 2：从源码安装
pip install -e .
pip install flash-attn --no-build-isolation
```

**注意**：
- flash-attn 需要 CUDA 环境和特定的编译工具链
- 如果安装失败，**不影响** tts-cli 的正常使用
- 只有在有 NVIDIA GPU 且需要极致性能时才需要安装

## 开发

### VSCode 推荐插件

本项目推荐使用以下 VSCode 插件：

| 插件 | ID | 用途 |
|------|-----|------|
| Python | `ms-python.python` | Python 基础支持 |
| Pylance | `ms-python.vscode-pylance` | 智能补全、跳转定义 |
| Ruff | `charliermarsh.ruff` | 代码检查 + 格式化 |

项目已配置 `.vscode/settings.json`，打开项目后会自动应用格式化规则。

### 常用命令

```bash
# 安装开发依赖
make install

# 格式化代码
make format

# 检查代码
make lint

# 运行测试
make test
```

## License

AGPLv3+
