# Runtime hook for Process4 Enhanced - 고도화된 환경 설정
import os
import sys
import warnings

# 경고 메시지 억제 (깔끔한 실행)
warnings.filterwarnings('ignore')

# MKL 완전 비활성화 및 OpenBLAS 강제 사용
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['MKL_INTERFACE_LAYER'] = 'LP64'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
os.environ['MKL_DISABLE_FAST_MM'] = '1'
os.environ['DISABLE_MKL'] = '1'

# PyQt5 플랫폼 플러그인 경로 설정
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '.'

# SSL 인증서 경로 설정 (위성영상 다운로드용)
if hasattr(sys, '_MEIPASS'):
    # 실행파일 환경
    base_path = sys._MEIPASS
    
    # GDAL 데이터 경로 설정
    gdal_data_path = os.path.join(base_path, 'gdal_data')
    if os.path.exists(gdal_data_path):
        os.environ['GDAL_DATA'] = gdal_data_path
        print(f"[Runtime] GDAL_DATA 설정: {gdal_data_path}")
    
    # PROJ 라이브러리 경로 설정 (좌표계 변환)
    proj_lib_path = os.path.join(base_path, 'proj_lib')
    if os.path.exists(proj_lib_path):
        os.environ['PROJ_LIB'] = proj_lib_path
        os.environ['PROJ_DATA'] = proj_lib_path
        print(f"[Runtime] PROJ_LIB 설정: {proj_lib_path}")
    
    # SSL 인증서 설정
    cert_path = os.path.join(base_path, 'certifi', 'cacert.pem')
    if os.path.exists(cert_path):
        os.environ['SSL_CERT_FILE'] = cert_path
        os.environ['REQUESTS_CA_BUNDLE'] = cert_path
        print(f"[Runtime] SSL 인증서 설정: {cert_path}")
else:
    # 개발 환경
    print("[Runtime] 개발 환경에서 실행 중")

# CUDA 설정 (GPU 사용 가능시)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 디버깅용
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # 메모리 최적화

# 한글 인코딩 설정
import locale
try:
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'Korean_Korea.949')
    except:
        pass

# 시스템 경로 설정
if hasattr(sys, '_MEIPASS'):
    # src 폴더를 경로에 추가
    src_path = os.path.join(sys._MEIPASS, 'src')
    if os.path.exists(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"[Runtime] src 경로 추가: {src_path}")
    
    # models 폴더 확인
    models_path = os.path.join(sys._MEIPASS, 'models')
    if os.path.exists(models_path):
        print(f"[Runtime] models 폴더 확인: {models_path}")
    
    # data 폴더 확인 (시군구 도로망)
    data_path = os.path.join(sys._MEIPASS, 'data')
    if os.path.exists(data_path):
        print(f"[Runtime] data 폴더 확인: {data_path}")

print("[Runtime Hook] 고도화된 환경 설정 완료")
print("=" * 50)

# NumPy 설정 확인 (디버깅용)
try:
    import numpy as np
    np_config = np.show_config()
    if 'openblas' in str(np_config).lower():
        print("[Runtime] NumPy: OpenBLAS 사용 확인")
    elif 'mkl' in str(np_config).lower():
        print("[Runtime] NumPy: MKL 사용 중 (주의)")
except:
    pass

# PyTorch 설정 확인
try:
    import torch
    if torch.cuda.is_available():
        print(f"[Runtime] PyTorch: CUDA 사용 가능 ({torch.cuda.get_device_name(0)})")
    else:
        print("[Runtime] PyTorch: CPU 모드")
except:
    pass

print("=" * 50)
