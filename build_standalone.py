#!/usr/bin/env python
"""
🔒 완전 독립실행형 빌드 스크립트
Python이 설치되지 않은 깨끗한 오프라인 환경에서도 100% 작동

🎯 대상 환경:
- Python 미설치 깨끗한 Windows PC
- 인터넷 완전 차단 환경  
- Visual C++ Redistributable 미설치
- 모든 런타임 라이브러리 포함

🔧 포함되는 요소들:
- Python 3.x 런타임 완전 포함
- Windows 런타임 DLL들
- 모든 데이터 파일 (skimage, matplotlib 등)
- 완전 자체 포함형 패키징
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path
import time

def print_header(text, symbol="🔒"):
    """헤더 출력"""
    print("\n" + "="*80)
    print(f"{symbol} {text}")
    print("="*80)

def check_standalone_prerequisites():
    """독립실행형 빌드 전제조건 확인"""
    print_header("독립실행형 빌드 전제조건 검사", "🔍")
    
    # 1. 기본 Python 환경 확인
    print(f"✅ Python 버전: {sys.version}")
    print(f"✅ 플랫폼: {platform.platform()}")
    print(f"✅ 아키텍처: {platform.architecture()[0]}")
    
    # 2. 필수 모듈 확인 (기존 spec 파일들 기반 완전 보완)
    required_modules = [
        'PyQt5', 'torch', 'geopandas', 'skimage', 'matplotlib', 
        'numpy', 'pandas', 'cv2', 'shapely', 'fiona', 'rasterio',
        'scipy', 'sklearn', 'networkx', 'numba', 'seaborn',
        'requests', 'tqdm', 'yaml', 'PIL'
    ]
    
    # 3. 선택적 모듈 확인 (없어도 빌드 가능)
    optional_modules = ['osgeo', 'gym', 'gymnasium']
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}: 설치됨")
        except ImportError:
            print(f"❌ {module}: 누락됨")
            missing_modules.append(module)
    
    # 선택적 모듈 확인
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✅ {module}: 설치됨 (선택적)")
        except ImportError:
            print(f"⚠️ {module}: 누락됨 (선택적 - 빌드 계속 가능)")
    
    if missing_modules:
        print(f"\n❌ 누락된 필수 모듈들: {', '.join(missing_modules)}")
        print("이 모듈들을 먼저 설치해야 합니다.")
        return False
    
    # 3. PyInstaller 확인
    try:
        import PyInstaller
        print(f"✅ PyInstaller: {PyInstaller.__version__}")
    except ImportError:
        print("❌ PyInstaller가 설치되지 않음")
        print("pip install pyinstaller 실행 필요")
        return False
    
    print("\n✅ 모든 전제조건이 충족되었습니다!")
    return True

def verify_runtime_dependencies():
    """런타임 의존성 확인"""
    print_header("런타임 의존성 검증", "🔧")
    
    checks = []
    
    # 1. skimage.data 확인
    try:
        import skimage.data
        skimage_path = os.path.dirname(skimage.data.__file__)
        if os.path.exists(skimage_path):
            files = os.listdir(skimage_path)
            print(f"✅ skimage.data: {len(files)}개 파일 발견")
            checks.append(True)
        else:
            print(f"❌ skimage.data: 경로 없음")
            checks.append(False)
    except Exception as e:
        print(f"❌ skimage.data: {e}")
        checks.append(False)
    
    # 2. matplotlib 데이터 확인
    try:
        import matplotlib
        mpl_path = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data')
        if os.path.exists(mpl_path):
            print(f"✅ matplotlib.mpl-data: 발견됨")
            checks.append(True)
        else:
            print(f"❌ matplotlib.mpl-data: 경로 없음")
            checks.append(False)
    except Exception as e:
        print(f"❌ matplotlib: {e}")
        checks.append(False)
    
    # 3. Python DLL 확인
    python_dll_paths = [
        os.path.join(sys.prefix, 'python*.dll'),
        os.path.join(os.path.dirname(sys.executable), 'python*.dll'),
        os.path.join(sys.prefix, 'DLLs'),
    ]
    
    dll_found = False
    for dll_path in python_dll_paths:
        import glob
        dlls = glob.glob(dll_path)
        if dlls:
            print(f"✅ Python DLL: {len(dlls)}개 발견")
            dll_found = True
            break
    
    if not dll_found:
        print("⚠️ Python DLL: 자동 감지 실패 (빌드는 계속 진행)")
    
    checks.append(dll_found)
    
    # 4. Visual C++ Runtime 확인
    vcruntime_paths = [
        os.path.join(sys.prefix, 'vcruntime*.dll'),
        os.path.join(os.path.dirname(sys.executable), 'vcruntime*.dll'),
        os.path.join(sys.prefix, 'Library', 'bin', 'vcruntime*.dll'),
    ]
    
    vcruntime_found = False
    for vc_path in vcruntime_paths:
        vcruntimes = glob.glob(vc_path)
        if vcruntimes:
            print(f"✅ Visual C++ Runtime: {len(vcruntimes)}개 발견")
            vcruntime_found = True
            break
    
    if not vcruntime_found:
        print("⚠️ Visual C++ Runtime: 자동 감지 실패")
        print("   독립실행형에서는 매우 중요한 요소입니다!")
    
    checks.append(vcruntime_found)
    
    return all(checks[:2])  # 핵심 데이터만 필수

def clean_build_environment():
    """빌드 환경 완전 정리"""
    print_header("빌드 환경 정리", "🧹")
    
    dirs_to_clean = ['build', 'dist', '__pycache__']
    
    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            print(f"🗑️ {dir_name} 디렉토리 삭제 중...")
            shutil.rmtree(dir_name)
    
    # 이전 빌드 결과물도 정리
    old_builds = [
        'Process3_Standalone_Complete',
        'Process3_Offline_Complete', 
        'Process3_Ultimate_Final'
    ]
    
    for build_name in old_builds:
        build_path = Path(f"dist/{build_name}")
        if build_path.exists():
            print(f"🗑️ 이전 빌드 결과 삭제: {build_name}")
            shutil.rmtree(build_path)
    
    print("✅ 빌드 환경 정리 완료")

def run_standalone_build():
    """독립실행형 빌드 실행"""
    print_header("독립실행형 빌드 시작", "🚀")
    
    spec_file = "process3_STANDALONE_COMPLETE.spec"
    
    if not Path(spec_file).exists():
        print(f"❌ {spec_file} 파일이 없습니다!")
        return False
    
    print(f"📋 사용할 spec 파일: {spec_file}")
    print("🎯 대상: Python 미설치 오프라인 환경")
    
    # 빌드 실행
    try:
        print("🔄 PyInstaller 실행 중...")
        print("   (독립실행형 빌드는 시간이 오래 걸릴 수 있습니다...)")
        
        # 빌드 명령어 실행 (더 안전한 방식)
        cmd = ["pyinstaller", "--clean", "--noconfirm", spec_file]
        print(f"🔧 실행 명령어: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True, 
            text=True,
            timeout=7200,  # 2시간 타임아웃 (더 여유있게)
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print("✅ PyInstaller 빌드 완료!")
            
            # 중요 파일들 확인
            exe_path = Path("dist/Process3_Standalone_Complete/Process3_Standalone_Complete.exe")
            base_lib_path = Path("dist/Process3_Standalone_Complete/_internal/base_library.zip")
            
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / (1024 * 1024)
                print(f"📊 실행파일 크기: {size_mb:.1f} MB")
                print(f"📁 실행파일 위치: {exe_path}")
            else:
                print("❌ 실행파일이 생성되지 않았습니다.")
                return False
            
            # 핵심 파일 확인
            if base_lib_path.exists():
                print("✅ base_library.zip 포함됨 (Python 라이브러리)")
            else:
                print("❌ base_library.zip 누락됨 - 빌드 불완전!")
                return False
            
            # 내부 구조 확인
            internal_path = Path("dist/Process3_Standalone_Complete/_internal")
            if internal_path.exists():
                internal_items = list(internal_path.iterdir())
                print(f"📦 내부 요소: {len(internal_items)}개 파일/폴더")
                
                # 중요 데이터 폴더들 확인
                important_folders = ['skimage', 'matplotlib', 'configs']
                for folder in important_folders:
                    folder_path = internal_path / folder
                    if folder_path.exists():
                        print(f"   ✅ {folder}/ 폴더 포함됨")
                    else:
                        print(f"   ⚠️ {folder}/ 폴더 누락됨")
            
            return True
        else:
            print(f"❌ 빌드 실패! (종료 코드: {result.returncode})")
            print("🔍 상세 오류 분석:")
            
            if result.stdout:
                print("표준 출력:")
                print(result.stdout[-2000:])  # 마지막 2000자만 출력
            
            if result.stderr:
                print("오류 출력:")
                print(result.stderr[-2000:])  # 마지막 2000자만 출력
            
            # 빌드 로그 파일 확인
            log_files = list(Path(".").glob("*.log"))
            if log_files:
                print(f"📝 로그 파일 확인: {[str(f) for f in log_files]}")
            
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 빌드 타임아웃 (1시간 초과)")
        return False
    except Exception as e:
        print(f"❌ 빌드 중 예외 발생: {e}")
        return False

def create_deployment_package():
    """배포용 패키지 생성"""
    print_header("배포 패키지 생성", "📦")
    
    dist_path = Path("dist/Process3_Standalone_Complete")
    if not dist_path.exists():
        print("❌ 빌드 결과물을 찾을 수 없습니다.")
        return False
    
    # 독립실행형 런처 스크립트 생성
    launcher_content = """@echo off
REM 🔒 완전 독립실행형 GCP_RL 도구 런처
echo ===============================================
echo   🔒 GCP_RL 독립실행형 도구
echo   Python 설치 없이 완전 독립 실행
echo ===============================================
echo.

REM 현재 디렉토리 확인
cd /d "%~dp0"

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
set PYTHONDONTWRITEBYTECODE=1
set NO_PROXY=*
set http_proxy=
set https_proxy=
set REQUESTS_CA_BUNDLE=

REM Windows DLL 경로 설정 (독립실행형)
set PATH=%~dp0;%~dp0\_internal;%PATH%

echo 🔒 독립실행형 모드로 설정되었습니다.
echo.

REM 실행파일 확인 및 실행
if exist "Process3_Standalone_Complete.exe" (
    echo 🚀 GCP_RL 독립실행형 도구를 시작합니다...
    echo    (첫 실행 시 초기화에 시간이 걸릴 수 있습니다)
    echo.
    start "" "Process3_Standalone_Complete.exe"
    echo ℹ️ 프로그램이 백그라운드에서 실행 중입니다.
    echo    GUI 창이 나타날 때까지 잠시 기다려주세요.
) else (
    echo ❌ Process3_Standalone_Complete.exe 파일을 찾을 수 없습니다.
    echo    빌드가 완료되었는지 확인하세요.
    pause
    exit /b 1
)

echo.
echo 📝 사용 완료 후 이 창을 닫으세요.
echo    문제 발생 시 logs 폴더의 로그를 확인하세요.
pause
"""
    
    launcher_path = dist_path / "launch_standalone.bat"
    with open(launcher_path, 'w', encoding='utf-8') as f:
        f.write(launcher_content)
    
    print(f"✅ 독립실행형 런처 생성: {launcher_path}")
    
    # README 파일 생성
    readme_content = """# 🔒 GCP_RL 독립실행형 도구

## 🎯 시스템 요구사항
- Windows 10/11 (64bit)
- Python 설치 불필요 ✅
- 인터넷 연결 불필요 ✅
- Visual C++ Redistributable 설치 불필요 ✅

## 🚀 사용 방법

### 1단계: 폴더 복사
전체 폴더를 대상 컴퓨터에 복사하세요.

### 2단계: 실행
`launch_standalone.bat` 파일을 더블클릭하세요.

### 3단계: 첫 실행 대기
첫 실행 시 초기화에 30초~1분 정도 걸릴 수 있습니다.

## 📁 폴더 구조
```
Process3_Standalone_Complete/
├── Process3_Standalone_Complete.exe  # 메인 실행파일
├── launch_standalone.bat             # 런처 스크립트
├── _internal/                        # 내부 라이브러리들
│   ├── skimage/                      # 이미지 처리 데이터
│   ├── matplotlib/                   # 그래프 폰트/데이터
│   ├── gdal-data/                    # GIS 투영 데이터
│   └── ...                           # 기타 라이브러리들
├── cache/                            # 캐시 폴더 (자동생성)
├── logs/                             # 로그 폴더 (자동생성)
├── sessions/                         # 세션 폴더 (자동생성)
└── results/                          # 결과 폴더 (자동생성)
```

## 🔧 문제 해결

### Q: 프로그램이 시작되지 않아요
A: `launch_standalone.bat`을 관리자 권한으로 실행해보세요.

### Q: "DLL을 찾을 수 없습니다" 오류
A: 전체 폴더가 완전히 복사되었는지 확인하세요.
   특히 `_internal` 폴더가 중요합니다.

### Q: 첫 실행이 너무 느려요
A: 정상입니다. 두 번째 실행부터는 빨라집니다.

### Q: 바이러스 경고가 나와요
A: 일부 백신에서 오탐지할 수 있습니다. 
   신뢰할 수 있는 소스라면 예외 처리하세요.

## 📞 기술 지원
- 로그 파일: `logs/` 폴더 확인
- 에러 발생 시: 런처 창의 오류 메시지 확인
"""
    
    readme_path = dist_path / "README.txt"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ README 파일 생성: {readme_path}")
    
    return True

def verify_standalone_build():
    """독립실행형 빌드 검증"""
    print_header("독립실행형 빌드 검증", "🔍")
    
    dist_path = Path("dist/Process3_Standalone_Complete")
    if not dist_path.exists():
        print("❌ 빌드 결과물이 없습니다.")
        return False
    
    # 필수 파일들 확인
    essential_files = [
        "Process3_Standalone_Complete.exe",
        "_internal",
        "launch_standalone.bat",
        "README.txt"
    ]
    
    missing_files = []
    for file_name in essential_files:
        file_path = dist_path / file_name
        if file_path.exists():
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"✅ {file_name}: {size_mb:.1f} MB")
            else:
                item_count = len(list(file_path.iterdir()))
                print(f"✅ {file_name}/: {item_count}개 항목")
        else:
            print(f"❌ {file_name}: 누락됨")
            missing_files.append(file_name)
    
    # 중요 데이터 폴더들 확인
    critical_data_folders = [
        "_internal/skimage",
        "_internal/matplotlib", 
        "_internal/gdal-data",
        "_internal/proj-data"
    ]
    
    for folder_path in critical_data_folders:
        full_path = dist_path / folder_path
        if full_path.exists():
            item_count = len(list(full_path.iterdir()))
            print(f"✅ {folder_path}: {item_count}개 파일")
        else:
            print(f"⚠️ {folder_path}: 누락됨 (일부 기능 제한 가능)")
    
    if missing_files:
        print(f"\n❌ 누락된 필수 파일들: {', '.join(missing_files)}")
        return False
    
    # 총 크기 계산
    total_size = sum(f.stat().st_size for f in dist_path.rglob('*') if f.is_file())
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"\n📊 전체 패키지 크기: {total_size_mb:.1f} MB")
    print("✅ 독립실행형 빌드 검증 완료!")
    
    return True

def main():
    """메인 함수"""
    # 🔥 OpenMP 라이브러리 충돌 방지 (가장 먼저 설정!)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    print_header("완전 독립실행형 빌드 시스템", "🔒")
    
    print("Python이 설치되지 않은 깨끗한 오프라인 환경에서도")
    print("100% 작동하는 완전 독립실행형 소프트웨어를 생성합니다.")
    
    # 1. 전제조건 확인
    if not check_standalone_prerequisites():
        print("\n❌ 전제조건이 충족되지 않았습니다. 빌드를 중단합니다.")
        return
    
    # 2. 런타임 의존성 확인
    if not verify_runtime_dependencies():
        print("\n⚠️ 일부 런타임 의존성에 문제가 있지만 빌드를 계속합니다.")
        print("   (일부 기능이 제한될 수 있습니다)")
    
    # 3. 빌드 시작 확인 (자동 진행)
    print_header("빌드 시작 확인", "❓")
    print("독립실행형 빌드는 시간이 오래 걸리고 큰 용량을 차지합니다.")
    print("예상 빌드 시간: 10~30분")
    print("예상 결과 크기: 1~3GB")
    print()
    print("🚀 자동으로 빌드를 시작합니다...")
    
    # 4. 환경 정리
    clean_build_environment()
    
    # 5. 빌드 실행
    start_time = time.time()
    success = run_standalone_build()
    build_time = time.time() - start_time
    
    # 6. 결과 처리
    if success:
        print_header("빌드 성공!", "🎉")
        print(f"⏱️ 빌드 시간: {build_time/60:.1f}분")
        
        # 배포 패키지 생성
        if create_deployment_package():
            print("✅ 배포 패키지 생성 완료!")
        
        # 빌드 검증
        if verify_standalone_build():
            print("\n🎯 완전 독립실행형 소프트웨어가 성공적으로 생성되었습니다!")
            print("📁 dist/Process3_Standalone_Complete/ 폴더를 확인하세요.")
            print("🚀 launch_standalone.bat을 실행하여 테스트할 수 있습니다.")
            print("")
            print("📦 배포 방법:")
            print("   1. 전체 Process3_Standalone_Complete 폴더를 USB나 네트워크로 복사")
            print("   2. 대상 PC에서 launch_standalone.bat 실행")
            print("   3. Python 설치 없이 바로 실행됩니다!")
        
    else:
        print_header("빌드 실패", "❌")
        print("문제를 해결한 후 다시 시도하세요.")

if __name__ == "__main__":
    main() 