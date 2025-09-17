@echo off
chcp 65001 >nul
echo.
echo 🔄 CPU PyTorch 롤백 스크립트
echo ================================================
echo.

echo 📋 현재 환경 확인 중...
python -c "import torch; print(f'현재 PyTorch: {torch.__version__}'); print(f'CUDA 사용 가능: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo ❌ PyTorch가 설치되지 않았습니다.
    pause
    exit /b 1
)

echo.
echo ⚠️  주의: 이 스크립트는 PyTorch를 CPU 버전으로 되돌립니다.
echo.
set /p confirm="계속하시겠습니까? (y/N): "
if /i not "%confirm%"=="y" (
    echo 취소되었습니다.
    pause
    exit /b 0
)

echo.
echo 🗑️  1단계: 현재 PyTorch 제거 중...
pip uninstall torch torchvision torchaudio -y
if errorlevel 1 (
    echo ❌ PyTorch 제거 실패
    pause
    exit /b 1
)

echo.
echo ⬇️  2단계: CPU PyTorch 설치 중...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
if errorlevel 1 (
    echo ❌ CPU PyTorch 설치 실패
    pause
    exit /b 1
)

echo.
echo 🔍 3단계: 설치 확인 중...
python -c "import torch; print(f'✅ PyTorch 버전: {torch.__version__}'); print(f'✅ CUDA 사용 가능: {torch.cuda.is_available()}'); print('✅ CPU 모드로 복구 완료')"
if errorlevel 1 (
    echo ❌ 설치 확인 실패
    pause
    exit /b 1
)

echo.
echo 🧪 4단계: 프로젝트 테스트 중...
python test_installation.py
if errorlevel 1 (
    echo ⚠️  테스트에서 일부 문제가 발생했지만 CPU PyTorch는 정상 설치되었습니다.
)

echo.
echo ✅ CPU PyTorch 롤백 완료!
echo.
echo 💡 Process3에서 "디바이스: cpu"로 표시되는지 확인해보세요!
echo.
pause 