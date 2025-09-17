# 🚀 AI Survey Control Point Pre-Selection - 설치 가이드

## 📋 시스템 요구사항

### 운영체제
- **Windows 10/11** (권장)
- **메모리**: 8GB 이상 (16GB 권장)
- **저장공간**: 10GB 이상
- **네트워크**: 패키지 다운로드용 인터넷 연결

### Python 환경
- **Python 3.11.4** (정확한 버전 필요)
- **pip**: 최신 버전
- **가상환경**: 이미 `gcp_env`로 생성됨

### GPU 지원 (선택사항)
- **NVIDIA GPU**: CUDA 12.6 지원
- **VRAM**: 4GB 이상 권장
- **드라이버**: 최신 NVIDIA 드라이버

## 🔧 설치 과정

### 1단계: 가상환경 활성화

```powershell
# 현재 디렉토리에서 가상환경 활성화
.\gcp_env\Scripts\Activate.ps1

# 활성화 확인 (프롬프트에 (gcp_env) 표시됨)
python --version  # Python 3.11.4 확인
```

### 2단계: 기본 패키지 설치

```powershell
# pip 업그레이드
python -m pip install --upgrade pip

# 기본 패키지 설치
pip install -r requirements.txt

# 설치 확인
pip list
```

### 3단계: GPU 지원 설치 (선택사항)

```powershell
# CUDA 12.6 지원 PyTorch 설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# GPU 인식 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 4단계: 프로젝트 구조 생성

```powershell
# 필요한 디렉토리 생성
mkdir src, configs, data, models, sessions, results, docs

# 하위 디렉토리 생성
mkdir src\core, src\learning, src\ui, src\filters, src\utils
```

## 📁 프로젝트 구조 확인

설치 완료 후 다음과 같은 구조가 생성됩니다:

```
AI_Survey_Control_Point_Pre-Selection/
├── README.md                    ✅ 프로젝트 문서
├── INSTALLATION.md              ✅ 설치 가이드  
├── requirements.txt             ✅ 패키지 목록
├── .gitignore                   🔄 생성 예정
├── gcp_env/                     ✅ 가상환경
│   ├── Scripts/
│   ├── Lib/
│   └── pyvenv.cfg
├── src/                         🔄 소스 코드
│   ├── __init__.py
│   ├── core/                    # 핵심 처리 모듈
│   │   ├── __init__.py
│   │   ├── skeleton_extractor.py
│   │   ├── road_processor.py
│   │   └── batch_processor.py
│   ├── learning/                # DQN 학습 모듈
│   │   ├── __init__.py
│   │   ├── dqn_model.py
│   │   ├── dqn_trainer.py
│   │   └── session_predictor.py
│   ├── ui/                      # 사용자 인터페이스
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   └── canvas_widget.py
│   ├── filters/                 # 필터링 모듈
│   │   ├── __init__.py
│   │   └── hybrid_filter.py
│   └── utils/                   # 유틸리티 함수
│       ├── __init__.py
│       └── utils.py
├── configs/                     🔄 설정 파일
│   ├── config.json
│   ├── dqn_config.json
│   └── ui_config.json
├── data/                        📁 학습 데이터 (자동 생성)
├── models/                      📁 학습된 모델 (자동 생성)
├── sessions/                    📁 세션 파일 (자동 생성)
├── results/                     📁 분석 결과 (자동 생성)
└── docs/                        📁 문서 (자동 생성)
```

## 🧪 설치 테스트

### 기본 패키지 테스트

```python
# test_installation.py 생성 후 실행
import sys
print(f"Python: {sys.version}")

# 핵심 패키지 테스트
packages = [
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'), 
    ('geopandas', 'GeoPandas'),
    ('shapely', 'Shapely'),
    ('torch', 'PyTorch'),
    ('PyQt5', 'PyQt5'),
    ('matplotlib', 'Matplotlib')
]

for module, name in packages:
    try:
        __import__(module)
        print(f"✅ {name}: OK")
    except ImportError:
        print(f"❌ {name}: 설치 필요")

# GPU 테스트
try:
    import torch
    print(f"🔥 CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except:
    print("❌ PyTorch GPU 테스트 실패")
```

### 실행 테스트

```powershell
# 테스트 실행
python test_installation.py

# 예상 출력:
# ✅ NumPy: OK
# ✅ Pandas: OK  
# ✅ GeoPandas: OK
# ✅ Shapely: OK
# ✅ PyTorch: OK
# ✅ PyQt5: OK
# ✅ Matplotlib: OK
# 🔥 CUDA: True (GPU 있는 경우)
```

## 🚀 첫 실행

### 프로그램 시작

```powershell
# 메인 런처 실행 (생성 예정)
python main_launcher.py

# 또는 개별 프로세스 실행
python process1_labeling_tool.py   # 라벨링 도구
python process2_training.py        # 모델 학습
python process3_inference.py       # AI 예측
```

## 🛠️ 문제 해결

### 일반적인 오류

#### 1. 가상환경 활성화 실패
```powershell
# PowerShell 실행 정책 변경
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 다시 활성화 시도
.\gcp_env\Scripts\Activate.ps1
```

#### 2. 패키지 설치 실패
```powershell
# pip 캐시 정리
pip cache purge

# 강제 재설치
pip install --force-reinstall -r requirements.txt
```

#### 3. CUDA 설치 실패
```powershell
# CUDA 버전 확인
nvidia-smi

# 올바른 CUDA 버전에 맞는 PyTorch 설치
# https://pytorch.org/get-started/locally/ 참조
```

#### 4. 한글 폰트 문제
```python
# matplotlib 한글 폰트 설정
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 시스템 폰트 확인
fonts = [f.name for f in fm.fontManager.ttflist if 'Gothic' in f.name]
print(fonts)

# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
```

### 성능 최적화

#### 메모리 사용량 감소
```python
# configs/config.json에서 설정
{
    "batch_size": 16,          # 기본값: 32
    "image_width": 800,        # 기본값: 1200  
    "max_workers": 2           # 기본값: 4
}
```

#### GPU 메모리 최적화
```python
# PyTorch GPU 메모리 설정
import torch
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
```

## 📞 지원 및 문의

### 기술 지원
- **이메일**: ksw3037@lx.or.kr
- **GitHub Issues**: 프로젝트 저장소에서 이슈 등록
- **문서**: `docs/` 폴더의 상세 매뉴얼 참조

### 버그 리포트
버그 발견 시 다음 정보와 함께 리포트:
1. 운영체제 및 버전
2. Python 버전
3. 오류 메시지 전문
4. 재현 단계
5. 예상 동작 vs 실제 동작

---

**🎉 설치 완료! 이제 AI Survey Control Point Pre-Selection 시스템을 사용할 준비가 되었습니다.** 