#!/usr/bin/env python3
"""
패치파일.exe 빌드 스크립트
"""

import subprocess
import sys
from pathlib import Path
import shutil

def build_patch():
    """패치 파일을 실행파일로 빌드"""
    
    print("🔧 프로세스4 패치파일 빌드 시작...")
    
    # PyInstaller 명령어 구성
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",                    # 단일 실행파일
        "--windowed",                   # 콘솔 창 숨기기 (GUI 모드)
        "--name=Process4_Patch",        # 실행파일 이름
        "--icon=NONE",                  # 아이콘 없음
        "--clean",                      # 빌드 캐시 정리
        "--noconfirm",                  # 덮어쓰기 확인 안함
        
        # 추가 데이터 파일들 포함
        "--add-data=src;src",           # src 폴더 포함
        
        # 최적화 옵션
        "--optimize=2",                 # 바이트코드 최적화
        
        "patch_process4_fix.py"         # 소스 파일
    ]
    
    try:
        print("📦 PyInstaller 실행 중...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ 빌드 성공!")
        
        # 빌드된 파일 위치 확인
        exe_path = Path("dist") / "Process4_Patch.exe"
        if exe_path.exists():
            file_size = exe_path.stat().st_size / (1024 * 1024)  # MB
            print(f"📁 실행파일: {exe_path}")
            print(f"📏 파일 크기: {file_size:.1f} MB")
            
            # 배포용 디렉토리 생성
            dist_dir = Path("patch_distribution")
            dist_dir.mkdir(exist_ok=True)
            
            # 실행파일 복사
            shutil.copy2(exe_path, dist_dir / "Process4_Patch.exe")
            
            # 수정된 소스 파일들도 복사 (패치 소스로 사용)
            shutil.copy2("src/core/district_road_clipper.py", dist_dir / "district_road_clipper.py")
            shutil.copy2("process4_inference.py", dist_dir / "process4_inference.py")
            
            # 사용 설명서 생성
            readme_content = f"""# 프로세스4 오류 수정 패치

## 사용 방법

1. **Process4_Patch.exe**를 프로세스4 실행파일과 같은 폴더에 복사
2. **프로세스4를 완전히 종료**
3. **Process4_Patch.exe 실행**
4. **환경 검사** 버튼 클릭
5. **패치 적용** 버튼 클릭
6. 완료 후 프로세스4 실행하여 오류 해결 확인

## 해결되는 문제

- ❌ TopologyException: side location conflict 오류
- ❌ 좌표계 변환 오류: 'polygons' 
- ❌ 폴리곤 데이터가 없습니다 경고

## 패치 내용

- **district_road_clipper.py**: 4단계 안전한 클리핑 시스템
- **process4_inference.py**: 좌표계 변환 안전장치 강화

## 안전 기능

- ✅ 자동 백업 (원본 파일 보존)
- ✅ 롤백 기능 (문제 시 원본 복구)
- ✅ 실행 중인 프로세스 자동 감지/종료

## 문의

패치 적용 중 문제가 발생하면 백업 파일로 롤백하세요.

버전: v1.0.0
날짜: 2025-01-28
"""
            
            with open(dist_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(readme_content)
            
            print(f"📦 배포 패키지: {dist_dir}")
            print("   ├── Process4_Patch.exe")
            print("   ├── district_road_clipper.py")
            print("   ├── process4_inference.py")
            print("   └── README.md")
            
        else:
            print("❌ 빌드된 실행파일을 찾을 수 없습니다")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 빌드 실패: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

if __name__ == "__main__":
    build_patch()
