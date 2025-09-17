#!/usr/bin/env python
"""
🔒 오프라인 환경 전용 빌드 스크립트
인터넷이 차단된 환경에서 안전하게 실행파일 생성

🎯 해결하는 문제들:
- skimage.data 폴더 누락 → 완전 포함
- matplotlib 폰트 오류 → 폰트 및 데이터 포함
- SSL 인증서 오류 → 인증서 포함
- GDAL/PROJ 데이터 누락 → 모든 데이터 포함
- 네트워크 의존성 → 오프라인 모드 강제
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import time

def print_header(text, symbol="🔒"):
    """헤더 출력"""
    print("\n" + "="*70)
    print(f"{symbol} {text}")
    print("="*70)

def check_offline_dependencies():
    """오프라인 필수 의존성 확인"""
    print_header("오프라인 의존성 검사", "🔍")
    
    required_modules = [
        'skimage', 'matplotlib', 'geopandas', 'torch', 
        'PyQt5', 'numpy', 'pandas', 'cv2'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}: 설치됨")
        except ImportError:
            print(f"❌ {module}: 누락됨")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️ 누락된 모듈들: {', '.join(missing_modules)}")
        print("오프라인 환경에서는 이 모듈들을 미리 설치해야 합니다.")
        return False
    
    print("\n✅ 모든 필수 모듈이 설치되어 있습니다!")
    return True

def verify_data_paths():
    """오프라인 필수 데이터 경로 확인"""
    print_header("데이터 파일 경로 확인", "📁")
    
    checks = []
    
    # skimage.data 확인
    try:
        import skimage.data
        skimage_path = os.path.dirname(skimage.data.__file__)
        if os.path.exists(skimage_path):
            print(f"✅ skimage.data: {skimage_path}")
            checks.append(True)
        else:
            print(f"❌ skimage.data: 경로 없음")
            checks.append(False)
    except:
        print(f"❌ skimage.data: 모듈 오류")
        checks.append(False)
    
    # matplotlib 데이터 확인
    try:
        import matplotlib
        mpl_path = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data')
        if os.path.exists(mpl_path):
            print(f"✅ matplotlib.mpl-data: {mpl_path}")
            checks.append(True)
        else:
            print(f"❌ matplotlib.mpl-data: 경로 없음")
            checks.append(False)
    except:
        print(f"❌ matplotlib: 모듈 오류")
        checks.append(False)
    
    # GDAL 데이터 확인
    gdal_paths = [
        os.path.join(os.environ.get('CONDA_PREFIX', ''), 'Library', 'share', 'gdal'),
        r'C:\Users\LX\anaconda3\Library\share\gdal',
        os.environ.get('GDAL_DATA', ''),
    ]
    
    gdal_found = False
    for gdal_path in gdal_paths:
        if gdal_path and os.path.exists(gdal_path):
            print(f"✅ GDAL 데이터: {gdal_path}")
            gdal_found = True
            break
    
    if not gdal_found:
        print("⚠️ GDAL 데이터: 경로를 찾을 수 없음 (빌드는 가능하지만 GIS 기능 제한 가능)")
    
    checks.append(gdal_found)
    
    return all(checks[:2])  # skimage와 matplotlib은 필수

def clean_build_dirs():
    """빌드 디렉토리 정리"""
    print_header("빌드 환경 정리", "🧹")
    
    dirs_to_clean = ['build', 'dist', '__pycache__']
    
    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            print(f"🗑️ {dir_name} 디렉토리 삭제 중...")
            shutil.rmtree(dir_name)
    
    print("✅ 빌드 디렉토리 정리 완료")

def run_offline_build():
    """오프라인 빌드 실행"""
    print_header("오프라인 빌드 시작", "🚀")
    
    spec_file = "process3_OFFLINE_COMPLETE.spec"
    
    if not Path(spec_file).exists():
        print(f"❌ {spec_file} 파일이 없습니다!")
        return False
    
    print(f"📋 사용할 spec 파일: {spec_file}")
    
    # 빌드 실행
    try:
        print("🔄 PyInstaller 실행 중...")
        result = subprocess.run(
            f"pyinstaller {spec_file}", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            print("✅ 빌드 성공!")
            
            # 결과 확인
            exe_path = Path("dist/Process3_Offline_Complete/Process3_Offline_Complete.exe")
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / (1024 * 1024)
                print(f"📊 실행파일 크기: {size_mb:.1f} MB")
                print(f"📁 실행파일 위치: {exe_path}")
                return True
            else:
                print("❌ 실행파일이 생성되지 않았습니다.")
                return False
        else:
            print(f"❌ 빌드 실패!")
            print(f"오류: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 빌드 중 예외 발생: {e}")
        return False

def create_offline_launcher():
    """오프라인 환경용 런처 스크립트 생성"""
    print_header("오프라인 런처 생성", "📜")
    
    launcher_content = """@echo off
REM 🔒 오프라인 환경용 GCP_RL 도구 런처
echo ===============================================
echo   🔒 GCP_RL 오프라인 도구
echo   인터넷 연결 없이 안전하게 실행됩니다
echo ===============================================
echo.

REM 필요한 디렉토리 생성
if not exist "cache" mkdir cache
if not exist "logs" mkdir logs
if not exist "sessions" mkdir sessions
if not exist "results" mkdir results
if not exist "output" mkdir output

echo ✅ 필요한 폴더들이 생성되었습니다.
echo.

REM 오프라인 모드 환경변수 설정
set OFFLINE_MODE=1
set NO_PROXY=*
set http_proxy=
set https_proxy=

echo 🔒 오프라인 모드로 설정되었습니다.
echo.

REM 실행파일 실행
if exist "Process3_Offline_Complete.exe" (
    echo 🚀 GCP_RL 도구를 시작합니다...
    start Process3_Offline_Complete.exe
    echo.
    echo ℹ️ 프로그램이 백그라운드에서 실행 중입니다.
) else (
    echo ❌ Process3_Offline_Complete.exe 파일을 찾을 수 없습니다.
    echo    빌드가 완료되었는지 확인하세요.
    pause
)

echo.
echo 프로그램 사용 완료 후 이 창을 닫으세요.
pause
"""
    
    with open("launch_offline.bat", 'w', encoding='utf-8') as f:
        f.write(launcher_content)
    
    print("✅ launch_offline.bat 런처 스크립트 생성 완료")

def create_troubleshooting_guide():
    """오프라인 환경 문제 해결 가이드 생성"""
    guide_content = """# 🔒 오프라인 환경 문제 해결 가이드

## 🚨 자주 발생하는 오류들

### 1. skimage.data 폴더 오류
**증상**: `FileNotFoundError: [Errno 2] No such file or directory: '...skimage/data/readme.txt'`

**해결방법**:
```bash
# 확인
python -c "import skimage.data; print(skimage.data.__file__)"

# 데이터 확인
python -c "import os; import skimage.data; print(os.listdir(os.path.dirname(skimage.data.__file__)))"
```

### 2. matplotlib 폰트 오류
**증상**: `UserWarning: Glyph missing from current font.`

**해결방법**:
- 폰트 캐시 재생성: `python -c "import matplotlib.font_manager; matplotlib.font_manager._rebuild()"`
- 시스템 폰트 사용: 코드에서 `plt.rcParams['font.family'] = 'DejaVu Sans'` 설정

### 3. GDAL/PROJ 데이터 오류
**증상**: `PROJ: proj_create_from_database: Cannot find proj.db`

**해결방법**:
```bash
# 환경변수 설정
set GDAL_DATA=C:\\Users\\LX\\anaconda3\\Library\\share\\gdal
set PROJ_LIB=C:\\Users\\LX\\anaconda3\\Library\\share\\proj
```

### 4. SSL 인증서 오류
**증상**: `SSL: CERTIFICATE_VERIFY_FAILED`

**해결방법**:
- 오프라인 모드 활성화
- 환경변수 설정: `set REQUESTS_CA_BUNDLE=`

## 🛠️ 수동 데이터 복사 방법

### skimage.data 수동 복사:
```bash
# 소스 경로 확인
python -c "import skimage.data; print(os.path.dirname(skimage.data.__file__))"

# 빌드된 실행파일 경로에 복사
copy "C:\\anaconda3\\Lib\\site-packages\\skimage\\data" "dist\\Process3_Offline_Complete\\_internal\\skimage\\data" /s
```

### matplotlib 데이터 수동 복사:
```bash
copy "C:\\anaconda3\\Lib\\site-packages\\matplotlib\\mpl-data" "dist\\Process3_Offline_Complete\\_internal\\matplotlib\\mpl-data" /s
```

## 🔍 진단 스크립트

### 의존성 확인:
```python
import sys
modules = ['skimage', 'matplotlib', 'geopandas', 'torch', 'PyQt5']
for module in modules:
    try:
        __import__(module)
        print(f"✅ {module}")
    except:
        print(f"❌ {module}")
```

### 데이터 경로 확인:
```python
import os
import skimage.data
import matplotlib

print("skimage.data:", os.path.dirname(skimage.data.__file__))
print("matplotlib:", os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data'))
print("GDAL_DATA:", os.environ.get('GDAL_DATA', 'Not set'))
print("PROJ_LIB:", os.environ.get('PROJ_LIB', 'Not set'))
```

## 📞 긴급 복구 방법

만약 빌드된 프로그램이 실행되지 않는다면:

1. **디버그 모드로 실행**:
   ```bash
   Process3_Offline_Complete.exe --debug
   ```

2. **콘솔 모드로 재빌드**:
   spec 파일에서 `console=True`로 변경 후 재빌드

3. **최소 기능으로 테스트**:
   불필요한 모듈들을 제거하고 점진적으로 추가
"""
    
    with open("OFFLINE_TROUBLESHOOTING.md", 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print("✅ OFFLINE_TROUBLESHOOTING.md 문제 해결 가이드 생성 완료")

def main():
    """메인 함수"""
    print_header("오프라인 환경 전용 빌드 시스템", "🔒")
    
    print("오프라인 환경에서 안전하게 실행파일을 생성합니다.")
    print("인터넷 연결이 없어도 모든 기능이 정상 작동합니다.")
    
    # 1. 의존성 확인
    if not check_offline_dependencies():
        print("\n❌ 필수 모듈이 누락되었습니다. 빌드를 중단합니다.")
        return
    
    # 2. 데이터 경로 확인
    if not verify_data_paths():
        print("\n⚠️ 일부 데이터 경로에 문제가 있지만 빌드를 계속합니다.")
    
    # 3. 빌드 시작
    print("\n🔄 빌드를 시작하시겠습니까?")
    choice = input("계속하려면 Enter, 취소하려면 'q': ").strip().lower()
    
    if choice == 'q':
        print("👋 빌드를 취소했습니다.")
        return
    
    # 4. 환경 정리
    clean_build_dirs()
    
    # 5. 빌드 실행
    start_time = time.time()
    success = run_offline_build()
    build_time = time.time() - start_time
    
    # 6. 결과 처리
    if success:
        print_header("빌드 완료!", "🎉")
        print(f"⏱️ 빌드 시간: {build_time:.1f}초")
        
        # 런처 및 가이드 생성
        create_offline_launcher()
        create_troubleshooting_guide()
        
        print("\n✅ 오프라인 환경용 실행파일이 성공적으로 생성되었습니다!")
        print("📁 dist/Process3_Offline_Complete/ 폴더를 확인하세요.")
        print("🚀 launch_offline.bat을 실행하여 프로그램을 시작할 수 있습니다.")
        
    else:
        print_header("빌드 실패", "❌")
        print("OFFLINE_TROUBLESHOOTING.md 파일을 참고하여 문제를 해결하세요.")

if __name__ == "__main__":
    main() 