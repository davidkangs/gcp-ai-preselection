"""설치 테스트"""
import sys
print(f"Python: {sys.version}")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except ImportError:
    print("✗ NumPy 설치 필요")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("✗ PyTorch 설치 필요")

try:
    import geopandas as gpd
    print(f"✓ GeoPandas: {gpd.__version__}")
except ImportError:
    print("✗ GeoPandas 설치 필요")

try:
    from PyQt5.QtCore import QT_VERSION_STR
    print(f"✓ PyQt5: {QT_VERSION_STR}")
except ImportError:
    print("✗ PyQt5 설치 필요")