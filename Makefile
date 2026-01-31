.PHONY: format lint check install test clean

# 格式化代码
format:
	ruff format src/
	ruff check --fix src/

# 检查代码 (不修改)
lint:
	ruff check src/

# 格式化 + 检查
check: format lint

# 安装开发依赖
install:
	python -m pip install -e ".[dev]"
	python -m pip install ruff

# 运行测试
test:
	pytest

# 清理缓存
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
