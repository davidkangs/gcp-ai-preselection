"""
laptop_optimization.py - 노트북 환경 최적화 설정
시연용 저사양 환경에서의 원활한 실행을 위한 설정
"""

import json
import torch
import psutil
from pathlib import Path

def create_laptop_config():
    """노트북용 최적화 설정 생성"""
    
    # 시스템 정보 확인
    ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count(logical=False)
    has_gpu = torch.cuda.is_available()
    
    print(f"시스템 정보:")
    print(f"- RAM: {ram_gb:.1f} GB")
    print(f"- CPU 코어: {cpu_count}개")
    print(f"- GPU: {'있음' if has_gpu else '없음'}")
    
    # 최적화 설정
    if ram_gb < 8:
        # 저사양
        config = {
            "batch_size": 16,
            "epochs": 20,
            "skeleton_max_points": 3000,
            "network_size": "small",
            "num_workers": 1
        }
        print("\n⚠️ 저사양 모드 설정")
    elif ram_gb < 16:
        # 중간 사양
        config = {
            "batch_size": 32,
            "epochs": 30,
            "skeleton_max_points": 5000,
            "network_size": "medium",
            "num_workers": 2
        }
        print("\nℹ️ 중간 사양 모드 설정")
    else:
        # 고사양
        config = {
            "batch_size": 64,
            "epochs": 50,
            "skeleton_max_points": 10000,
            "network_size": "large",
            "num_workers": 4
        }
        print("\n✅ 고사양 모드 설정")
    
    # GPU 설정
    if has_gpu:
        # GPU 메모리 확인
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 4:
            config["batch_size"] = min(config["batch_size"], 16)
            print(f"GPU 메모리 {gpu_memory:.1f}GB - 배치 크기 제한")
    else:
        config["use_gpu"] = False
        config["batch_size"] = min(config["batch_size"], 32)
        print("CPU 모드 - 배치 크기 제한")
    
    # 전체 설정
    full_config = {
        "system_info": {
            "ram_gb": ram_gb,
            "cpu_cores": cpu_count,
            "has_gpu": has_gpu,
            "optimized_for": "laptop_demo"
        },
        "learning": {
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "learning_rate": 0.001,
            "action_size": 4,
            "state_size": 7,
            "network_size": config.get("network_size", "small"),
            "hidden_sizes": {
                "small": [128, 64],
                "medium": [256, 128, 64],
                "large": [512, 256, 128, 64]
            }[config.get("network_size", "small")]
        },
        "ui": {
            "skeleton_max_points": config["skeleton_max_points"],
            "skeleton_sampling_rate": 1 if config["skeleton_max_points"] >= 5000 else 2,
            "antialiasing": False,  # 성능을 위해 비활성화
            "road_opacity": 0.5,
            "update_mode": "smart",  # 스마트 업데이트
            "cache_mode": True       # 캐싱 활성화
        },
        "performance": {
            "use_gpu": config.get("use_gpu", has_gpu),
            "num_workers": config["num_workers"],
            "prefetch_factor": 2,
            "persistent_workers": False,  # 메모리 절약
            "pin_memory": has_gpu and ram_gb >= 8
        },
        "demo_mode": {
            "enabled": True,
            "auto_save": False,  # 시연 중 자동 저장 비활성화
            "show_fps": False,   # FPS 표시 비활성화
            "simplified_ui": True # 단순화된 UI
        }
    }
    
    # 저장
    with open("laptop_config.json", 'w', encoding='utf-8') as f:
        json.dump(full_config, f, indent=2, ensure_ascii=False)
    
    print(f"\n설정 저장 완료: laptop_config.json")
    print(f"\n권장 설정:")
    print(f"- 배치 크기: {config['batch_size']}")
    print(f"- 에폭: {config['epochs']}")
    print(f"- 최대 스켈레톤 포인트: {config['skeleton_max_points']}")
    
    return full_config

def apply_laptop_config():
    """노트북 설정 적용"""
    
    config_file = Path("laptop_config.json")
    if not config_file.exists():
        print("설정 파일이 없습니다. 생성합니다...")
        config = create_laptop_config()
    else:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # Process 2 설정 파일 생성
    process2_config = {
        "batch_size": config["learning"]["batch_size"],
        "epochs": config["learning"]["epochs"],
        "learning_rate": config["learning"]["learning_rate"],
        "network_size": config["learning"]["network_size"]
    }
    
    with open("process2_laptop_config.json", 'w', encoding='utf-8') as f:
        json.dump(process2_config, f, indent=2)
    
    print("\n✅ 노트북 최적화 설정이 적용되었습니다!")
    
    # 추가 팁
    print("\n💡 시연 팁:")
    print("1. 작은 shapefile로 시작 (1-2MB)")
    print("2. 배터리 절약 모드 해제")
    print("3. 불필요한 프로그램 종료")
    print("4. 학습 전 GPU 메모리 정리:")
    print("   import torch")
    print("   torch.cuda.empty_cache()")
    
    return config

def quick_performance_test():
    """빠른 성능 테스트"""
    import time
    import numpy as np
    
    print("\n빠른 성능 테스트 중...")
    
    # CPU 테스트
    start = time.time()
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)
    c = np.dot(a, b)
    cpu_time = time.time() - start
    print(f"CPU 행렬곱 (1000x1000): {cpu_time:.3f}초")
    
    # GPU 테스트
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = time.time()
        a = torch.rand(1000, 1000).cuda()
        b = torch.rand(1000, 1000).cuda()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"GPU 행렬곱 (1000x1000): {gpu_time:.3f}초")
        print(f"GPU 가속: {cpu_time/gpu_time:.1f}배")
    
    # 권장사항
    if cpu_time > 0.5:
        print("\n⚠️ CPU 성능이 낮습니다. 배치 크기를 줄이세요.")
    else:
        print("\n✅ CPU 성능이 적절합니다.")

if __name__ == "__main__":
    print("=" * 60)
    print("노트북 환경 최적화 도구")
    print("=" * 60)
    
    # 설정 생성 및 적용
    config = apply_laptop_config()
    
    # 성능 테스트
    print("\n성능 테스트를 실행하시겠습니까? (y/n)")
    if input().lower() == 'y':
        quick_performance_test()
    
    print("\n완료! 이제 시스템을 실행할 준비가 되었습니다.")
    print("python main_launcher.py")