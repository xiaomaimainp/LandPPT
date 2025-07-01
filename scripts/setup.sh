#!/bin/bash
# LandPPT Unix/Linux/macOS 快速设置脚本

set -e  # 遇到错误时退出

echo "🚀 LandPPT 快速设置开始..."
echo "=================================================="

# 检查Python版本
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python3未安装"
        echo "请先安装Python 3.11或更高版本"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "✅ Python版本: $python_version"
    
    # 检查版本是否满足要求
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
        echo "❌ Python版本过低，需要3.11或更高版本"
        exit 1
    fi
}

# 检查并安装uv
install_uv() {
    if command -v uv &> /dev/null; then
        echo "✅ uv已安装: $(uv --version)"
        return 0
    fi
    
    echo "📦 安装uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # 重新加载PATH
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if command -v uv &> /dev/null; then
        echo "✅ uv安装成功: $(uv --version)"
    else
        echo "❌ uv安装失败"
        echo "请手动安装uv: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
}

# 设置环境变量文件
setup_env() {
    echo "⚙️ 设置环境变量..."
    
    if [ ! -f ".env.example" ]; then
        echo "❌ .env.example文件不存在"
        exit 1
    fi
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
        echo "✅ 已创建.env文件"
        echo "📝 请编辑.env文件，配置你的AI API密钥"
    else
        echo "ℹ️ .env文件已存在，跳过创建"
    fi
}

# 同步依赖
sync_deps() {
    echo "📦 同步项目依赖..."
    uv sync
    echo "✅ 依赖同步成功"
}

# 主函数
main() {
    check_python
    install_uv
    setup_env
    sync_deps
    init_db
    
    echo "=================================================="
    echo "🎉 LandPPT 设置完成！"
    echo ""
    echo "下一步:"
    echo "1. 编辑 .env 文件，配置你的AI API密钥"
    echo "2. 运行服务: uv run python run.py"
    echo "3. 访问: http://localhost:8000"
    echo ""
    echo "开发模式:"
    echo "- 运行测试: uv run pytest"
    echo "- 代码格式化: uv run black src/"
    echo "- 类型检查: uv run mypy src/"
    echo ""
}

# 运行主函数
main "$@"
