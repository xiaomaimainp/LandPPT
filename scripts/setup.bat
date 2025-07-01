@echo off
REM LandPPT Windows 快速设置脚本

echo 🚀 LandPPT Windows 快速设置开始...
echo ==================================================

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python未安装或不在PATH中
    echo 请先安装Python 3.11或更高版本
    pause
    exit /b 1
)

echo ✅ Python已安装
python --version

REM 检查uv是否安装
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 安装uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo ❌ uv安装失败
        echo 请手动安装uv: https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
    echo ✅ uv安装成功
) else (
    echo ✅ uv已安装
    uv --version
)

REM 设置环境变量文件
if not exist .env (
    if exist .env.example (
        copy .env.example .env
        echo ✅ 已创建.env文件
    ) else (
        echo ❌ .env.example文件不存在
        pause
        exit /b 1
    )
) else (
    echo ℹ️ .env文件已存在，跳过创建
)

REM 同步依赖
echo 📦 同步项目依赖...
uv sync
if %errorlevel% neq 0 (
    echo ❌ 依赖同步失败
    pause
    exit /b 1
)
echo ✅ 依赖同步成功

echo ==================================================
echo 🎉 LandPPT 设置完成！
echo.
echo 下一步:
echo 1. 编辑 .env 文件，配置你的AI API密钥
echo 2. 运行服务: uv run python run.py
echo 3. 访问: http://localhost:8000
echo.
echo 开发模式:
echo - 运行测试: uv run pytest
echo - 代码格式化: uv run black src/
echo - 类型检查: uv run mypy src/
echo.
pause
