# 🚀 GCP_RL_Tool EXE 빌드 가이드

## 📋 빌드 방법

### **방법 1: 자동 빌드 스크립트 (추천)**

```bash
# 빌드 스크립트 실행
python build_exe.py
```

옵션 선택:
- **1번**: 단일 EXE 파일 (배포하기 쉬움, 크기 클 수 있음)
- **2번**: 디렉토리 형태 (더 안정적, 빠름)

### **방법 2: 수동 빌드**

```bash
# 1. 기존 빌드 파일 정리
rmdir /s build dist

# 2. 단일 EXE 빌드
pyinstaller GCP_RL_Tool.spec

# 3. 디렉토리 형태 빌드 (대안)
pyinstaller --onedir --windowed process3_inference.py
```

## 📦 빌드 결과

### **단일 EXE 방식**
```
dist/
└── GCP_RL_Tool.exe    (약 500MB~2GB)
```

### **디렉토리 방식**
```
dist/
└── GCP_RL_Tool/
    ├── GCP_RL_Tool.exe
    ├── 라이브러리 파일들/
    └── 데이터 파일들/
```

## 🛠️ 문제 해결

### **일반적인 오류들**

**1. 메모리 부족 오류**
```bash
# 해결: 가상 메모리 늘리기 또는 디렉토리 방식 사용
python build_exe.py → 2번 선택
```

**2. 모듈 누락 오류**
```bash
# 해결: spec 파일의 hiddenimports에 추가
ModuleNotFoundError: No module named 'xxx'
→ GCP_RL_Tool.spec 파일 수정
```

**3. PyQt5 오류**
```bash
# 해결: PyQt5 재설치
pip uninstall PyQt5
pip install PyQt5==5.15.9
```

**4. Torch 관련 오류**
```bash
# 해결: CPU 버전 사용 확인
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### **빌드 최적화**

**파일 크기 줄이기:**
1. `excludes` 목록에 불필요한 패키지 추가
2. UPX 압축 활성화 (기본 설정됨)
3. 디렉토리 방식 사용

**속도 향상:**
```python
# spec 파일에서 debug=False, optimize=2 설정
exe = EXE(
    # ...
    debug=False,
    optimize=2,  # 최적화 레벨 증가
    # ...
)
```

## 📁 배포용 파일 구성

### **최소 배포 파일**
```
📦 배포_패키지/
├── 🚀 GCP_RL_Tool.exe          # 실행파일
├── 📜 install.bat              # 설치 도우미
├── 📖 사용법.txt               # 간단한 사용 설명
└── 📂 필수_폴더들/             # 런타임 생성됨
    ├── cache/
    ├── logs/
    ├── sessions/
    └── results/
```

### **전체 배포 파일 (디렉토리 방식)**
```
📦 배포_패키지/
├── 📂 GCP_RL_Tool/             # 실행 파일들
│   ├── GCP_RL_Tool.exe
│   ├── 라이브러리_파일들/
│   └── 데이터_파일들/
├── 📜 install.bat
└── 📖 README.txt
```

## 🔧 고급 설정

### **커스텀 아이콘 추가**
```python
# spec 파일에서
exe = EXE(
    # ...
    icon='icon.ico',  # 아이콘 파일 경로
    # ...
)
```

### **추가 데이터 파일 포함**
```python
# spec 파일의 datas_list에 추가
datas_list = [
    # 기존 파일들...
    (str(work_dir / 'data'), 'data'),           # 데이터 폴더
    (str(work_dir / 'docs'), 'docs'),           # 문서 폴더
    ('icon.ico', '.'),                          # 아이콘 파일
]
```

### **성능 프로파일링**
```bash
# 빌드 시간 측정
time pyinstaller GCP_RL_Tool.spec

# 실행 시간 측정
python -m cProfile process3_inference.py
```

## 🚨 중요 사항

### **라이선스 확인**
- PyQt5: GPL/Commercial 라이선스
- Torch: BSD 라이선스  
- Geopandas: BSD 라이선스

### **배포 시 주의사항**
1. **크기**: 단일 EXE는 500MB~2GB 될 수 있음
2. **속도**: 첫 실행 시 압축 해제로 인해 느릴 수 있음
3. **바이러스 검사**: 일부 백신에서 오탐지 가능
4. **의존성**: 사용자 PC에 별도 Python 설치 불필요

### **테스트 권장사항**
1. 깨끗한 Windows PC에서 테스트
2. Python이 설치되지 않은 환경에서 테스트
3. 다양한 Windows 버전에서 테스트

## 📞 문제 발생 시

**로그 확인:**
```bash
# 디버그 모드로 빌드
pyinstaller --debug=all GCP_RL_Tool.spec

# 실행 로그 확인
GCP_RL_Tool.exe > debug.log 2>&1
```

**일반적인 해결책:**
1. 가상환경에서 빌드하기
2. 디렉토리 방식으로 변경
3. 의존성 최소화
4. 32bit/64bit 호환성 확인 