"""Health check routes."""

from fastapi import APIRouter

from ... import __version__
from ..db import Database
from ..models import HealthResponse
from ..services.tts_engine import TTSEngine

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="""
检查服务的运行状态。

### 返回信息
- `status`: 服务状态（healthy）
- `model_loaded`: ChatTTS 模型是否已加载
- `version`: 服务版本号
- `mongodb_connected`: MongoDB 数据库是否已连接

### 使用场景
- Kubernetes/Docker 健康检查探针
- 负载均衡器健康检查
- 监控系统状态检测
    """,
)
async def health_check():
    """健康检查"""
    tts_engine = TTSEngine()

    return HealthResponse(
        status="healthy",
        model_loaded=tts_engine.is_loaded(),
        version=__version__,
        mongodb_connected=Database.is_connected(),
    )
