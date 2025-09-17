#!/usr/bin/env python3
"""
AI Survey Control Point Pre-Selection - 설치 테스트
Python 3.11.4 및 필수 패키지 설치 확인
"""

import sys
import importlib
from pathlib import Path

def test_python_version():
    """Python 버전 테스트"""
    print("🐍 Python 버전 확인:")
    print(f"   버전: {sys.version}")
    print(f"   버전 정보: {sys.version_info}")
    
    if sys.version_info >= (3, 11, 4):
        print("   ✅ Python 3.11.4+ 확인됨")
        return True
    else:
        print("   ❌ Python 3.11.4+ 필요")
        return False

def test_packages():
    """핵심 패키지 테스트"""
    print("\n📦 패키지 설치 확인:")
    
    # 필수 패키지 목록
    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('geopandas', 'GeoPandas'),
        ('shapely', 'Shapely'),
        ('fiona', 'Fiona'),
        ('pyproj', 'PyProj'),
        ('rasterio', 'Rasterio'),
        ('torch', 'PyTorch'),
        ('PyQt5', 'PyQt5'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('cv2', 'OpenCV (opencv-python)'),
        ('tqdm', 'TQDM'),
        ('networkx', 'NetworkX')
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for module_name, display_name in packages:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"   ✅ {display_name}: {version}")
            success_count += 1
        except ImportError as e:
            print(f"   ❌ {display_name}: 설치 필요 ({e})")
        except Exception as e:
            print(f"   ⚠️ {display_name}: 확인 실패 ({e})")
    
    print(f"\n   📊 결과: {success_count}/{total_count} 패키지 설치됨")
    return success_count == total_count

def test_gpu():
    """GPU 지원 테스트"""
    print("\n🔥 GPU 지원 확인:")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            
            print(f"   ✅ CUDA 사용 가능")
            print(f"   🎮 GPU 개수: {gpu_count}")
            print(f"   🎯 GPU 이름: {gpu_name}")
            print(f"   🔧 CUDA 버전: {cuda_version}")
            return True
        else:
            print("   ❌ CUDA 사용 불가 (CPU 모드)")
            print("   💡 GPU 가속을 원한다면 CUDA용 PyTorch 설치 필요")
            return False
            
    except ImportError:
        print("   ❌ PyTorch 설치되지 않음")
        return False
    except Exception as e:
        print(f"   ⚠️ GPU 테스트 실패: {e}")
        return False

def test_project_structure():
    """프로젝트 구조 테스트"""
    print("\n📁 프로젝트 구조 확인:")
    
    required_dirs = [
        'src',
        'src/core',
        'src/learning', 
        'src/ui',
        'src/filters',
        'src/utils',
        'configs',
        'data',
        'models',
        'sessions',
        'results',
        'docs'
    ]
    
    required_files = [
        'requirements.txt',
        'README.md',
        'INSTALLATION.md',
        '.gitignore',
        'src/__init__.py',
        'configs/config.json',
        'configs/dqn_config.json'
    ]
    
    success_count = 0
    total_count = len(required_dirs) + len(required_files)
    
    # 디렉토리 확인
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"   ✅ 디렉토리: {dir_path}")
            success_count += 1
        else:
            print(f"   ❌ 디렉토리: {dir_path} (누락)")
    
    # 파일 확인  
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ 파일: {file_path}")
            success_count += 1
        else:
            print(f"   ❌ 파일: {file_path} (누락)")
    
    print(f"\n   📊 결과: {success_count}/{total_count} 구조 요소 확인됨")
    return success_count == total_count

def test_config_files():
    """설정 파일 테스트"""
    print("\n⚙️ 설정 파일 확인:")
    
    config_files = [
        'configs/config.json',
        'configs/dqn_config.json'
    ]
    
    success_count = 0
    
    for config_path in config_files:
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"   ✅ {config_path}: 유효한 JSON")
            success_count += 1
        except FileNotFoundError:
            print(f"   ❌ {config_path}: 파일 없음")
        except json.JSONDecodeError as e:
            print(f"   ❌ {config_path}: JSON 오류 ({e})")
        except Exception as e:
            print(f"   ⚠️ {config_path}: 읽기 실패 ({e})")
    
    return success_count == len(config_files)

def main():
    """메인 테스트 실행"""
    print("🧪 AI Survey Control Point Pre-Selection - 설치 테스트")
    print("=" * 60)
    
    tests = [
        ("Python 버전", test_python_version),
        ("패키지 설치", test_packages),
        ("GPU 지원", test_gpu),
        ("프로젝트 구조", test_project_structure),
        ("설정 파일", test_config_files)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ {test_name} 테스트 중 오류: {e}")
            results.append((test_name, False))
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("📊 최종 테스트 결과:")
    
    success_count = 0
    for test_name, success in results:
        status = "✅ 통과" if success else "❌ 실패"
        print(f"   {test_name}: {status}")
        if success:
            success_count += 1
    
    total_tests = len(results)
    print(f"\n🎯 전체 결과: {success_count}/{total_tests} 테스트 통과")
    
    if success_count == total_tests:
        print("🎉 모든 테스트 통과! 시스템 사용 준비 완료!")
        return 0
    else:
        print("⚠️ 일부 테스트 실패. INSTALLATION.md 참조 후 재설치 필요")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 