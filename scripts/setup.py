#!/usr/bin/env python3
"""
LandPPT 快速设置脚本
自动化项目环境设置过程
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, check=True, shell=False):
    """运行命令并处理错误"""
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ 命令执行失败: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        print(f"错误信息: {e.stderr}")
        return None


def check_python_version():
    """检查Python版本"""
    print("🔍 检查Python版本...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        print("请安装Python 3.11或更高版本")
        return False
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True


def check_uv_installed():
    """检查uv是否已安装"""
    print("🔍 检查uv是否已安装...")
    result = run_command(["uv", "--version"], check=False)
    if result and result.returncode == 0:
        print(f"✅ uv已安装: {result.stdout.strip()}")
        return True
    return False


def install_uv():
    """安装uv"""
    print("📦 安装uv...")
    
    if os.name == 'nt':  # Windows
        cmd = 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
        result = run_command(cmd, shell=True)
    else:  # macOS/Linux
        cmd = 'curl -LsSf https://astral.sh/uv/install.sh | sh'
        result = run_command(cmd, shell=True)
    
    if result and result.returncode == 0:
        print("✅ uv安装成功")
        return True
    else:
        print("❌ uv安装失败，请手动安装")
        return False


def setup_environment():
    """设置环境变量文件"""
    print("⚙️ 设置环境变量...")
    
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_example.exists():
        print("❌ .env.example文件不存在")
        return False
    
    if not env_file.exists():
        shutil.copy(env_example, env_file)
        print("✅ 已创建.env文件")
        print("📝 请编辑.env文件，配置你的AI API密钥")
    else:
        print("ℹ️ .env文件已存在，跳过创建")
    
    return True


def sync_dependencies():
    """同步依赖"""
    print("📦 同步项目依赖...")
    
    result = run_command(["uv", "sync"])
    if result and result.returncode == 0:
        print("✅ 依赖同步成功")
        return True
    else:
        print("❌ 依赖同步失败")
        return False
    


def main():
    """主函数"""
    print("🚀 LandPPT 快速设置开始...")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查并安装uv
    if not check_uv_installed():
        if not install_uv():
            print("请手动安装uv后重新运行此脚本")
            sys.exit(1)
    
    # 设置环境变量
    if not setup_environment():
        sys.exit(1)
    
    # 同步依赖
    if not sync_dependencies():
        sys.exit(1)
    
    
    print("=" * 50)
    print("🎉 LandPPT 设置完成！")
    print()
    print("下一步:")
    print("1. 编辑 .env 文件，配置你的AI API密钥")
    print("2. 运行服务: uv run python run.py")
    print("3. 访问: http://localhost:8000")
    print()
    print("开发模式:")
    print("- 运行测试: uv run pytest")
    print("- 代码格式化: uv run black src/")
    print("- 类型检查: uv run mypy src/")


if __name__ == "__main__":
    main()
