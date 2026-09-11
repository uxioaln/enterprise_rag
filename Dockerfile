# =========================================================================
# 企业知识库 RAG 问答服务 Dockerfile（python:3.10-slim）
#
# 构建（默认）：
#   docker build -t kb-rag .
#
# 运行（secrets 通过 --env-file 注入，不写入镜像；data/ 挂载宿主机目录）：
#   docker run -d --name kb-api \
#     -p 8000:8000 \
#     --env-file .env \
#     -e KB_REDIS__URL=redis://redis:6379/0 \
#     -v $(pwd)/data:/app/data \
#     kb-rag
#
# Celery worker（与 API 容器共用 env-file 与 data 卷，另起一个容器执行）：
#   docker run --env-file .env -e KB_REDIS__URL=redis://redis:6379/0 \
#     -v $(pwd)/data:/app/data --entrypoint celery kb-rag \
#     -A app.celery_app:celery_app worker --loglevel=info --pool=threads --concurrency=4
# =========================================================================

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # pip 使用阿里云镜像作为主源（清华源 anti-crawler 拦截，USTC 会 302 到清华源同样 403）
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
    TZ=Asia/Shanghai \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken

# -------------------------------------------------------------------------
# 环境变量声明（secrets 一律留空，运行时通过 --env-file .env 或 -e 注入）
# -------------------------------------------------------------------------
# API 密钥（见 env 模板文件）
ENV AGICTO_API_KEY="" \
    OPENAI_API_KEY="" \
    GEMINI_API_KEY="" \
    JINA_API_KEY="" \
    # 会话存储后端：sqlite（默认）/ memory
    STORAGE_BACKEND=sqlite \
    # SQLite 会话库路径（容器内路径，配合 data 卷挂载持久化）
    KB_DB_PATH=data/chat_history.db \
    # CORS 允许来源（生产环境运行时覆盖）
    ALLOWED_ORIGINS="http://localhost:3000,http://localhost:5173,http://localhost:8080" \
    # Redis 连接（config.json 默认 localhost:6379，容器化时按实际地址覆盖，
    # 例如 docker-compose 中为 redis://redis:6379/0）
    KB_REDIS__URL="redis://localhost:6379/0"

WORKDIR /app

# 系统依赖：curl 仅用于健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖再拷贝源码，充分利用 Docker 层缓存（源码变更不触发重装）
COPY requirements.txt ./
# 先安装 uv（Rust 实现的 pip 替代品，依赖解析速度快 10-50 倍）
RUN pip install uv -i ${PIP_INDEX_URL} --trusted-host mirrors.aliyun.com
# 使用 uv 安装依赖（--system 装到系统 Python，uv 默认优先预编译 wheel）
RUN uv pip install --system \
    -r requirements.txt \
    --index-url ${PIP_INDEX_URL} \
    --extra-index-url ${PIP_EXTRA_INDEX_URL}

# 拷贝项目源码（data/、tests/、.git 等已在 .dockerignore 中排除）
COPY app/ ./app/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY main.py config.json ./

# pyprojroot.here() 以 requirements.txt 等文件定位项目根，/app 已满足；
# 创建 data 子目录，保证未挂载卷时 Pipeline 初始化目录不报错
RUN mkdir -p /app/data/stock_data

# -------------------------------------------------------------------------
# 模型预下载：tiktoken 编码表
# 统一下载到镜像层内，运行时无需联网下载
# -------------------------------------------------------------------------
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); tiktoken.get_encoding('o200k_base')"

# 非 root 用户运行；缓存目录归属该用户
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser
ENV HOME=/app

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# 默认启动 API 服务；Celery worker 启动命令见文件头部注释
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
