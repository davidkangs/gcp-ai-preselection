# 🎯 AI Survey Control Point Pre-Selection

**GCP 강화학습 기반 지적측량기준점 AI 선점 시스템**

도로망 shapefile에서 교차점, 커브, 끝점을 자동으로 검출하여 지적측량기준점 선점을 지원하는 AI 기반 분석 도구입니다.

## 📌 프로젝트 개요

- **목적**: 도로망 데이터에서 주요 특징점을 자동 검출하여 측량기준점 최적 위치 선정
- **기술**: 스켈레톤 추출 + 휴리스틱 분석 + DQN 강화학습
- **특징**: 3단계 프로세스로 구성된 완전 자동화 워크플로우

## ✨ 주요 기능

### 🔍 Process 1: 데이터 라벨링
- **다중 레이어 시각화**
  - 🛣️ 원본 도로망 (반투명 회색)
  - 🔵 추출된 중심선 (파란색)
  - 🔴 분석된 포인트 (색상별 구분)
- **스켈레톤 추출**: Shapefile → 도로 중심선 자동 추출
- **휴리스틱 분석**: 각도 기반 교차점 자동 검출
- **인터랙티브 편집**: 마우스/키보드 조작으로 수동 보정

### 🧠 Process 2: AI 모델 학습
- **DQN 강화학습**: 사용자 라벨링 패턴 학습
- **실시간 모니터링**: 학습 진행 상황 시각화
- **성능 최적화**: 자동 하이퍼파라미터 튜닝

### 🚀 Process 3: AI 예측 및 개선
- **실시간 예측**: AI 기반 특징점 자동 검출
- **인간-AI 협업**: AI 제안 + 인간 수정
- **지속 학습**: 수정 데이터로 모델 개선

## 🛠️ 설치 가이드

### 시스템 요구사항
- **OS**: Windows 10/11
- **Python**: 3.11.4
- **메모리**: 8GB 이상 (16GB 권장)
- **GPU**: NVIDIA GPU (선택사항, CUDA 지원)

### 빠른 설치

1. **가상환경 활성화**
```bash
# 가상환경이 이미 생성되어 있습니다
.\gcp_env\Scripts\Activate.ps1
```

2. **패키지 설치**
```bash
# 기본 패키지 설치
pip install -r requirements.txt

# GPU 지원 (CUDA 12.6)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

3. **프로그램 실행**
```bash
# 메인 런처 실행
python main_launcher.py

# 또는 개별 프로세스 실행
python process1_labeling_tool.py
python process2_training.py
python process3_inference.py
```

## 📖 사용 방법

### 🔄 전체 워크플로우

1. **데이터 라벨링 (Process 1)**
   - Shapefile 선택 → 자동 교차점 검출 → 수동 보정 → 세션 저장

2. **모델 학습 (Process 2)**
   - 라벨링 세션 로드 → DQN 학습 → 모델 저장

3. **AI 예측 (Process 3)**
   - AI 자동 검출 → 결과 수정 → 재학습

### ⌨️ 키보드 단축키

- `좌클릭`: 커브 포인트 추가
- `우클릭`: 끝점 추가
- `Shift+클릭`: 포인트 제거
- `D`: 가장 가까운 점 삭제
- `Space`: 화면 맞춤
- `1,2,3`: 각 레이어 토글
- `Q`: DQN 예측 실행
- `T`: AI 예측 토글

## 📁 프로젝트 구조

```
AI_Survey_Control_Point_Pre-Selection/
├── README.md                    # 프로젝트 문서
├── INSTALLATION.md              # 설치 가이드
├── requirements.txt             # 패키지 목록
├── .gitignore                   # Git 무시 파일
├── gcp_env/                     # 가상환경 (생성됨)
├── src/                         # 소스 코드
│   ├── core/                    # 핵심 처리 모듈
│   ├── learning/                # DQN 학습 모듈
│   ├── ui/                      # 사용자 인터페이스
│   ├── filters/                 # 필터링 모듈
│   └── utils/                   # 유틸리티 함수
├── configs/                     # 설정 파일
├── data/                        # 학습 데이터
├── models/                      # 학습된 모델
├── sessions/                    # 세션 파일
├── results/                     # 분석 결과
└── docs/                        # 문서
```

## 🎯 특징점 분류

- **🔴 교차점 (Intersection)**: 3개 이상 도로가 만나는 지점
- **🔵 커브 (Curve)**: 도로 방향이 급격히 변하는 지점
- **🟢 끝점 (Endpoint)**: 도로가 시작되거나 끝나는 지점

## 🔧 설정

주요 설정은 `configs/` 폴더에서 관리됩니다:
- `config.json`: 전체 시스템 설정
- `dqn_config.json`: DQN 학습 파라미터
- `ui_config.json`: UI 설정

## 🆘 문제 해결

### 일반적인 오류

1. **Import 오류**: 가상환경 활성화 확인
2. **Shapefile 로드 실패**: 파일 경로와 권한 확인
3. **GPU 메모리 부족**: 배치 크기 감소
4. **한글 폰트 문제**: 시스템 폰트 설정 확인

### 성능 최적화

- **메모리 사용량 감소**: 배치 크기 조정
- **처리 속도 향상**: GPU 사용 또는 numba 최적화
- **정확도 개선**: 더 많은 라벨링 데이터 확보

## 🤝 기여 방법

1. Fork 프로젝트
2. Feature 브랜치 생성
3. 변경사항 커밋
4. Pull Request 생성

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

## 👥 개발팀

- **프로젝트**: GCP 강화학습 기반 AI 선점 시스템
- **목적**: 지적재조사 업무에 AI 기술 적용 (연구/교육용)

---

**💡 이 시스템은 기존의 복잡한 코드를 정리하고 최적화하여 새롭게 구성된 클린 아키텍처 버전입니다.**
