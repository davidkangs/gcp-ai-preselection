#!/usr/bin/env python
"""
🚀 GCP_RL_Tool EXE 빌드 스크립트
- 다양한 빌드 옵션 제공
- 오류 처리 및 로깅
- 빌드 후 검증
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import time

def print_header(text):
    """헤더 출력"""
    print("\n" + "="*60)
    print(f"🚀 {text}")
    print("="*60)

def run_command(command, description):
    """명령어 실행 및 결과 확인"""
    print(f"\n📋 {description}")
    print(f"실행: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 성공!")
            if result.stdout:
                print(f"출력: {result.stdout}")
            return True
        else:
            print(f"❌ 실패! (코드: {result.returncode})")
            if result.stderr:
                print(f"오류: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        return False

def clean_build_dirs():
    """빌드 디렉토리 정리"""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    
    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            print(f"🧹 {dir_name} 디렉토리 삭제 중...")
            shutil.rmtree(dir_name)
    
    print("✅ 빌드 디렉토리 정리 완료")

def check_file_size(file_path):
    """파일 크기 확인"""
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"📊 파일 크기: {size_mb:.1f} MB")
        return size_mb
    return 0

def build_onefile():
    """단일 EXE 파일 빌드"""
    print_header("단일 EXE 파일 빌드")
    
    spec_file = "GCP_RL_Tool.spec"
    if not Path(spec_file).exists():
        print(f"❌ {spec_file} 파일이 없습니다!")
        return False
    
    # 빌드 실행
    success = run_command(f"pyinstaller {spec_file}", "PyInstaller 실행")
    
    if success:
        exe_path = Path("dist/GCP_RL_Tool.exe")
        if exe_path.exists():
            size_mb = check_file_size(exe_path)
            print(f"✅ 단일 EXE 파일 생성 완료: {exe_path}")
            
            if size_mb > 1000:  # 1GB 이상이면 경고
                print("⚠️ 파일 크기가 매우 큽니다. 디렉토리 방식을 고려해보세요.")
            
            return True
        else:
            print("❌ EXE 파일이 생성되지 않았습니다.")
            return False
    
    return False

def build_directory():
    """디렉토리 형태로 빌드 (더 안정적)"""
    print_header("디렉토리 형태 빌드")
    
    # 임시 spec 파일 생성 (디렉토리 방식)
    temp_spec = "GCP_RL_Tool_dir.spec"
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path

work_dir = Path.cwd()

datas_list = [
    (str(work_dir / 'src'), 'src'),
    (str(work_dir / 'configs'), 'configs'),
    (str(work_dir / 'src' / 'learning' / 'models'), 'src/learning/models'),
]

hidden_imports_list = [
    'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets', 'PyQt5.sip',
    'geopandas', 'fiona', 'shapely', 'pyproj',
    'torch', 'torch.nn', 'torch.optim', 'torchvision',
    'numpy', 'pandas', 'scipy', 'sklearn', 'networkx', 'cv2',
    'src.core', 'src.learning', 'src.process3', 'src.ui', 'src.utils',
]

a = Analysis(
    ['process3_inference.py'],
    pathex=[str(work_dir)],
    binaries=[],
    datas=datas_list,
    hiddenimports=hidden_imports_list,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib.tests', 'numpy.tests'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # 디렉토리 방식
    name='GCP_RL_Tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GCP_RL_Tool',
)
'''
    
    # 임시 spec 파일 작성
    with open(temp_spec, 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    # 빌드 실행
    success = run_command(f"pyinstaller {temp_spec}", "디렉토리 방식 빌드")
    
    # 임시 파일 정리
    if Path(temp_spec).exists():
        Path(temp_spec).unlink()
    
    if success:
        dist_dir = Path("dist/GCP_RL_Tool")
        if dist_dir.exists():
            exe_file = dist_dir / "GCP_RL_Tool.exe"
            if exe_file.exists():
                check_file_size(exe_file)
                print(f"✅ 디렉토리 빌드 완료: {dist_dir}")
                print(f"🚀 실행 파일: {exe_file}")
                return True
    
    return False

def create_installer_script():
    """간단한 설치 스크립트 생성"""
    installer_content = '''@echo off
echo ========================================
echo   GCP_RL_Tool 설치 도우미
echo ========================================
echo.

REM 필요한 디렉토리 생성
if not exist "cache" mkdir cache
if not exist "logs" mkdir logs
if not exist "sessions" mkdir sessions
if not exist "results" mkdir results

echo ✅ 필요한 폴더들이 생성되었습니다.
echo.
echo 🚀 GCP_RL_Tool.exe를 실행하세요!
echo.
pause
'''
    
    with open("install.bat", 'w', encoding='utf-8') as f:
        f.write(installer_content)
    
    print("📜 install.bat 설치 스크립트 생성 완료")

def main():
    """메인 함수"""
    print_header("GCP_RL_Tool EXE 빌드 시스템")
    
    print("빌드 옵션을 선택하세요:")
    print("1. 단일 EXE 파일 (느리지만 배포 쉬움)")
    print("2. 디렉토리 형태 (빠르고 안정적)")
    print("3. 빌드 디렉토리 정리만")
    print("4. 종료")
    
    choice = input("\n선택 (1-4): ").strip()
    
    if choice == "1":
        clean_build_dirs()
        success = build_onefile()
        if success:
            create_installer_script()
            print("\n🎉 단일 EXE 빌드 완료!")
            print("📁 dist/GCP_RL_Tool.exe 파일을 확인하세요.")
    
    elif choice == "2":
        clean_build_dirs()
        success = build_directory()
        if success:
            create_installer_script()
            print("\n🎉 디렉토리 빌드 완료!")
            print("📁 dist/GCP_RL_Tool/ 폴더를 배포하세요.")
    
    elif choice == "3":
        clean_build_dirs()
        print("✅ 정리 완료!")
    
    elif choice == "4":
        print("👋 빌드 시스템을 종료합니다.")
    
    else:
        print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main() 