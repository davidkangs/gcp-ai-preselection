@echo off
echo ========================================
echo Process4 고도화 독립실행형 빌드 시작
echo ========================================
echo.

REM 기존 빌드 폴더 삭제
if exist "dist\Process4_Enhanced" (
    echo [정리] 기존 빌드 폴더 삭제 중...
    rmdir /s /q "dist\Process4_Enhanced"
)

if exist "build" (
    echo [정리] 기존 build 폴더 삭제 중...
    rmdir /s /q "build"
)

echo.
echo [빌드] PyInstaller 실행 중...
echo.

REM PyInstaller 실행
pyinstaller --clean --noconfirm process4_ENHANCED_FINAL.spec

echo.
echo ========================================

if exist "dist\Process4_Enhanced\Process4_Enhanced.exe" (
    echo [성공] 빌드 완료!
    echo.
    echo 실행 파일 위치:
    echo   dist\Process4_Enhanced\Process4_Enhanced.exe
    echo.
    echo 전체 폴더 크기 확인 중...
    
    REM PowerShell로 폴더 크기 계산
    powershell -Command "& {$size = (Get-ChildItem -Path 'dist\Process4_Enhanced' -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB; Write-Host ('총 크기: {0:N2} GB' -f $size)}"
    
    echo.
    echo [테스트] 프로그램 실행 테스트...
    echo.
    
    REM 테스트 실행 (5초 후 자동 종료)
    start /wait /b cmd /c "timeout /t 1 >nul & dist\Process4_Enhanced\Process4_Enhanced.exe --version"
    
    if %ERRORLEVEL% EQU 0 (
        echo [OK] 실행 테스트 성공
    ) else (
        echo [경고] 실행 테스트 실패 (에러 코드: %ERRORLEVEL%)
    )
) else (
    echo [실패] 빌드 실패!
    echo 로그를 확인하세요.
)

echo.
echo ========================================
echo 빌드 프로세스 완료
echo ========================================
pause
