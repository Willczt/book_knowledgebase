.PHONY: dev dev-qdrant install install-frontend lint format migrate

# 启动开发环境（仅 Qdrant，后端和前端分别手动启动）
dev-qdrant:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up qdrant -d

# 启动后端开发服务器（使用项目 venv）
dev-backend:
	cd backend && /Users/bytedance/.venv/bin/python -m uvicorn main:app --reload --port 8000 --host 0.0.0.0

# 启动前端开发服务器（自动安装依赖）
dev-frontend:
	cd frontend && [ -d node_modules ] || npm install
	cd frontend && npm run dev

# 完整开发环境（三个终端）
dev: dev-qdrant
	@echo "Qdrant started at http://localhost:6333"
	@echo "Run 'make dev-backend' in another terminal"
	@echo "Run 'make dev-frontend' in another terminal"

# 安装后端依赖
install:
	cd backend && pip install -e ".[dev]" 2>/dev/null || pip install -r requirements.txt

# 安装前端依赖
install-frontend:
	cd frontend && npm install

# 数据库迁移
migrate:
	cd backend && alembic upgrade head

# 生成新迁移
migration:
	cd backend && alembic revision --autogenerate -m "$(msg)"

# 代码检查
lint:
	cd backend && ruff check .
	cd frontend && npm run lint

# 代码格式化
format:
	cd backend && ruff format .

# 生产构建
build:
	cd frontend && npm run build
	docker compose build

# 生产启动
up:
	docker compose up -d

# 停止所有服务
down:
	docker compose down

# 清理数据卷（危险操作）
clean:
	docker compose down -v

# 查看后端日志
logs-backend:
	docker compose logs -f backend

# 批量摄入文档
ingest:
	python -m cli.main ingest $(path) --kb-id $(kb)
