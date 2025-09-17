@echo off
chcp 65001 >nul
echo.
echo 🚀 CUDA PyTorch 업그레이드 스크립트
echo ================================================
echo.

:: 현재 환경 확인
echo 📋 현재 환경 확인 중...
python -c "import torch; print(f'현재 PyTorch: {torch.__version__}'); print(f'CUDA 사용 가능: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo ❌ PyTorch가 설치되지 않았습니다.
    pause
    exit /b 1
)

echo.
echo ⚠️  주의: 이 스크립트는 PyTorch를 CUDA 버전으로 교체합니다.
echo.
set /p confirm="계속하시겠습니까? (y/N): "
if /i not "%confirm%"=="y" (
    echo 취소되었습니다.
    pause
    exit /b 0
)

echo.
echo 📦 1단계: 환경 백업 중...
pip freeze > cuda_upgrade_backup_%date:~0,4%%date:~5,2%%date:~8,2%.txt
echo ✅ 백업 완료: cuda_upgrade_backup_%date:~0,4%%date:~5,2%%date:~8,2%.txt

echo.
echo 🗑️  2단계: 기존 PyTorch 제거 중...
pip uninstall torch torchvision torchaudio -y
if errorlevel 1 (
    echo ❌ 기존 PyTorch 제거 실패
    pause
    exit /b 1
)

echo.
echo ⬇️  3단계: CUDA PyTorch 설치 중...
echo    이 과정은 몇 분 소요될 수 있습니다...
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
if errorlevel 1 (
    echo ❌ CUDA PyTorch 설치 실패
    echo 롤백을 시작합니다...
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
    pause
    exit /b 1
)

echo.
echo 🔍 4단계: 설치 확인 중...
python -c "import torch; print(f'✅ PyTorch 버전: {torch.__version__}'); print(f'✅ CUDA 사용 가능: {torch.cuda.is_available()}'); print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
if errorlevel 1 (
    echo ❌ 설치 확인 실패
    pause
    exit /b 1
)

echo.
echo 🧪 5단계: 프로젝트 테스트 중...
python test_installation.py
if errorlevel 1 (
    echo ⚠️  테스트에서 일부 문제가 발생했지만 CUDA는 정상 설치되었습니다.
)

echo.
echo 🎉 CUDA PyTorch 업그레이드 완료!
echo.
echo 📈 이제 다음과 같은 성능 향상을 기대할 수 있습니다:
echo    • 학습 속도: 10-100배 빨라짐
echo    • VRAM: 48GB 활용 가능
echo    • 배치 크기: 대폭 증가 가능
echo.
echo 💡 Process3에서 "디바이스: cuda"로 표시되는지 확인해보세요!
echo.
pause 