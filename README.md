# tts-cli

tts-cli 是一个本地离线使用的文字转语音命令行工具，基于 ChatTTS 和 Whisper。

## 特性

- 完全本地运行，无需网络，保障隐私
- 支持中文和英文语音合成
- 自动生成 SRT 字幕文件
- 智能文本分割，长文本自动分段处理
- 支持保存和复用说话人音色
- **HTTP API Server 模式**：支持异步任务处理，适合服务化部署

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
curl -X POST http://localhost:8000/api/v1/create_tts_task \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，欢迎使用 tts-cli 服务。",
    "language": "zh",
    "speed": 3,
    "skip_subtitles": false
  }'
```

**请求参数：**

| 参数 | 类型 | 必填 | 说明 | 默认值 |
|------|------|------|------|--------|
| `text` | string | 是 | 要转换的文本（最大 100,000 字符） | - |
| `language` | string | 否 | 语言：`en` 或 `zh` | en |
| `speed` | int | 否 | 语速 0-9 | 3 |
| `break_level` | int | 否 | 标点停顿强度 0-7 | 5 |
| `speaker_id` | string | 否 | 本地说话人音色文件路径 (.pt) | null |
| `max_length` | int | 否 | 文本分块最大长度 | 500 |
| `max_batch` | int | 否 | 批处理大小 | 1 |
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
