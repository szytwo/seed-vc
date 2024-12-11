@echo off
chcp 65001 >nul
echo  启动中，请耐心等待 

REM 激活目标虚拟环境
CALL "venv\Scripts\activate"

REM 检查是否激活成功
IF ERRORLEVEL 1 (
    echo 激活虚拟环境失败，请检查路径或环境名称！
    pause
    exit /b
)

REM 执行 Python 脚本
python api.py --cuda 0 --api True

REM 保持窗口打开
pause
