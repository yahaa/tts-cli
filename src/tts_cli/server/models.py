"""Pydantic request/response models for the TTS API."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class CreateTtsTaskRequest(BaseModel):
    """创建 TTS 任务的请求模型"""

    text: str = Field(
        ...,
        description="要转换的文本内容，最大支持 100,000 字符",
        max_length=100000,
        examples=["你好，欢迎使用 tts-cli 服务。"],
    )
    language: str = Field(
        default="en",
        description="语言代码：'en'（英文）或 'zh'（中文）",
        examples=["zh", "en"],
    )
    speed: int = Field(
        default=3,
        ge=0,
        le=9,
        description="语速，范围 0-9，数字越大语速越快",
        examples=[3, 5],
    )
    break_level: int = Field(
        default=5,
        ge=0,
        le=7,
        description="标点停顿强度，范围 0-7，数字越大停顿越长",
        examples=[5],
    )
    speaker_id: Optional[str] = Field(
        default=None,
        description="本地说话人音色文件路径 (.pt 格式)",
        examples=["/path/to/speaker.pt"],
    )
    max_length: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="文本自动分块的最大长度",
        examples=[500, 800],
    )
    max_batch: int = Field(
        default=1,
        ge=1,
        le=10,
        description="并行处理的最大批次数",
        examples=[1, 2],
    )
    skip_subtitles: bool = Field(
        default=False,
        description="是否跳过字幕生成",
        examples=[False, True],
    )
    whisper_model: str = Field(
        default="base",
        description="Whisper 模型大小：tiny/base/small/medium/large",
        examples=["base", "small"],
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="任务完成后的回调 URL，服务器会向此 URL 发送 POST 请求",
        examples=["https://example.com/callback"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "你好，欢迎使用 tts-cli 服务。这是一个基于 ChatTTS 的文字转语音工具。",
                    "language": "zh",
                    "speed": 3,
                    "skip_subtitles": False,
                },
                {
                    "text": "Hello, welcome to tts-cli service.",
                    "language": "en",
                    "speed": 5,
                    "skip_subtitles": True,
                },
            ]
        }
    }


class CreateTtsTaskResponse(BaseModel):
    """创建 TTS 任务的响应模型"""

    request_id: str = Field(description="请求唯一标识", examples=["req_abc123def456"])
    task_id: str = Field(description="任务唯一标识", examples=["task_xyz789abc"])
    status: str = Field(description="任务状态", examples=["waiting"])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": "req_abc123def456",
                    "task_id": "task_xyz789abc",
                    "status": "waiting",
                }
            ]
        }
    }


class TaskInfo(BaseModel):
    """任务详细信息模型"""

    task_id: str = Field(description="任务唯一标识", examples=["task_xyz789abc"])
    status: Literal["waiting", "processing", "success", "failed"] = Field(
        description="任务状态：waiting（等待）、processing（处理中）、success（成功）、failed（失败）",
        examples=["success"],
    )
    audio_url: Optional[str] = Field(
        default=None,
        description="音频文件下载 URL（任务成功后可用）",
        examples=["/api/v1/files/task_xyz789abc/output.wav"],
    )
    subtitle_url: Optional[str] = Field(
        default=None,
        description="字幕文件下载 URL（任务成功且未跳过字幕时可用）",
        examples=["/api/v1/files/task_xyz789abc/output.srt"],
    )
    duration: Optional[float] = Field(
        default=None,
        description="音频时长（秒）",
        examples=[12.5],
    )
    character_count: Optional[int] = Field(
        default=None,
        description="文本字符数",
        examples=[256],
    )
    error_code: Optional[str] = Field(
        default=None,
        description="错误代码（任务失败时返回）",
        examples=["GENERATION_FAILED"],
    )
    error_message: Optional[str] = Field(
        default=None,
        description="错误信息（任务失败时返回）",
        examples=["Text contains invalid characters"],
    )
    create_time: datetime = Field(
        description="任务创建时间",
        examples=["2024-01-10T10:00:00Z"],
    )
    start_time: Optional[datetime] = Field(
        default=None,
        description="任务开始处理时间",
        examples=["2024-01-10T10:00:01Z"],
    )
    finish_time: Optional[datetime] = Field(
        default=None,
        description="任务完成时间",
        examples=["2024-01-10T10:00:15Z"],
    )


class DescribeTtsTaskResponse(BaseModel):
    """查询 TTS 任务状态的响应模型"""

    request_id: str = Field(description="请求唯一标识", examples=["req_def456ghi789"])
    task: Optional[TaskInfo] = Field(
        default=None,
        description="任务详细信息（任务存在时返回）",
    )
    error: Optional[dict] = Field(
        default=None,
        description="错误信息（任务不存在时返回）",
        examples=[{"code": "TASK_NOT_FOUND", "message": "Task not found: task_xxx"}],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": "req_def456ghi789",
                    "task": {
                        "task_id": "task_xyz789abc",
                        "status": "success",
                        "audio_url": "/api/v1/files/task_xyz789abc/output.wav",
                        "subtitle_url": "/api/v1/files/task_xyz789abc/output.srt",
                        "duration": 12.5,
                        "character_count": 256,
                        "error_code": None,
                        "error_message": None,
                        "create_time": "2024-01-10T10:00:00Z",
                        "start_time": "2024-01-10T10:00:01Z",
                        "finish_time": "2024-01-10T10:00:15Z",
                    },
                    "error": None,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """健康检查响应模型"""

    status: str = Field(description="服务状态", examples=["healthy"])
    model_loaded: bool = Field(description="ChatTTS 模型是否已加载", examples=[True])
    version: str = Field(description="服务版本号", examples=["0.1.0"])
    mongodb_connected: bool = Field(description="MongoDB 是否已连接", examples=[True])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "model_loaded": True,
                    "version": "0.1.0",
                    "mongodb_connected": True,
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """错误响应模型"""

    request_id: str = Field(description="请求唯一标识", examples=["req_err123"])
    error: dict = Field(
        description="错误详情",
        examples=[
            {"code": "INVALID_PARAMETER", "message": "Speed must be between 0 and 9"}
        ],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": "req_err123",
                    "error": {
                        "code": "INVALID_PARAMETER",
                        "message": "Speed must be between 0 and 9",
                    },
                }
            ]
        }
    }
