# Runtime hook for Process4 - MKL 문제 해결
import os
import sys

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

# GDAL 데이터 경로 설정 (독립실행형용)
if hasattr(sys, '_MEIPASS'):
    gdal_data_path = os.path.join(sys._MEIPASS, 'gdal_data')
    if os.path.exists(gdal_data_path):
        os.environ['GDAL_DATA'] = gdal_data_path
    
    proj_lib_path = os.path.join(sys._MEIPASS, 'proj_lib')
    if os.path.exists(proj_lib_path):
        os.environ['PROJ_LIB'] = proj_lib_path

print("[Runtime Hook] MKL 비활성화 및 OpenBLAS 설정 완료")
