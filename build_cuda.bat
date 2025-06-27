@echo off
REM CUDA构建脚本 (Windows版本)
echo 构建CUDA版本的密码猜测程序...

REM 检查CUDA是否可用
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: NVCC编译器未找到，请确保CUDA Toolkit已安装
    echo 请检查以下路径是否在PATH环境变量中：
    echo C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
    pause
    exit /b 1
)

REM 显示CUDA版本信息
echo CUDA编译器信息:
nvcc --version

REM 编译CUDA版本
echo.
echo 正在编译...
nvcc -std=c++11 -O2 ^
    -o main_cuda.exe ^
    main_cuda.cpp ^
    guessing_cuda.cu ^
    train.cpp ^
    guessing.cpp ^
    md5.cpp ^
    -I. ^
    --expt-relaxed-constexpr ^
    --extended-lambda

if %errorlevel% equ 0 (
    echo.
    echo ==============================
    echo 编译成功！
    echo 可执行文件: main_cuda.exe
    echo 运行命令: main_cuda.exe
    echo ==============================
    echo.
    echo 是否现在运行程序？ (Y/N)
    set /p choice=
    if /i "%choice%"=="Y" (
        echo 正在运行程序...
        main_cuda.exe
    )
) else (
    echo.
    echo ==============================
    echo 编译失败！
    echo 请检查错误信息并修复后重试
    echo ==============================
    pause
    exit /b 1
)

pause
