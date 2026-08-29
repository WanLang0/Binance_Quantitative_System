#!/usr/bin/env python3
"""
币安量化回测系统启动脚本（Flask 版）
"""

import sys
import io

# Windows 控制台默认 GBK，强制使用 UTF-8 以便正常打印 emoji 等字符
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import subprocess
import os

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import flask
        import ccxt
        import pandas
        import numpy
        import plotly
        import ta
        print("✅ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        return False

def install_dependencies():
    """安装依赖"""
    print("正在安装依赖...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖安装完成")
        return True
    except subprocess.CalledProcessError:
        print("❌ 依赖安装失败")
        return False

def main():
    """主函数"""
    print("🚀 币安量化回测系统")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("正在安装依赖...")
        if not install_dependencies():
            print("请手动运行: pip install -r requirements.txt")
            return
    
    # 启动 Flask 应用
    print("正在启动 Flask 应用 (端口 8502)...")
    os.environ.setdefault("FLASK_APP", "app.py")
    try:
        subprocess.run([sys.executable, "-m", "flask", "run", "--port", "8502", "--host", "127.0.0.1"])
    except KeyboardInterrupt:
        print("\n应用已停止")
    except Exception as e:
        print(f"启动失败: {e}")

if __name__ == "__main__":
    main()
