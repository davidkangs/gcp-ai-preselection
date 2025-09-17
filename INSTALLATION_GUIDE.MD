# 도로망 AI 분석 시스템 - 설치 및 사용 가이드

## 📋 시스템 요구사항

- **OS**: Windows 10/11
- **Python**: 3.11.4
- **메모리**: 8GB 이상 (16GB 권장)
- **GPU**: NVIDIA GPU (선택사항, CUDA 지원)
- **저장공간**: 10GB 이상

## 🚀 빠른 설치

### 1. 프로젝트 설정

```bash
# 프로젝트 폴더로 이동
cd I:\gcp_rl

# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate
```

### 2. 패키지 설치

```bash
# 기본 패키지 설치
pip install -r requirements.txt

# GPU 사용 시 (CUDA 12.6)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 3. 프로젝트 구조 확인

```
I:\gcp_rl\
├── src/                    # 기존 소스 코드
│   ├── core/
│   ├── learning/
│   └── ui/
├── process1_labeling_tool.py    # 프로세스 1: 라벨링
├── process1_batch.py
├── process2_training.py         # 프로세스 2: 학습
├── process3_inference.py        # 프로세스 3: AI 예측
├── process3_retrain.py
├── process3_batch.py
├── main_launcher.py            # 메인 런처
├── run_process1.py            # 개별 실행 스크립트
├── run_process2.py
├── run_process3.py
├── sessions/                  # 세션 저장 (자동 생성)
├── models/                    # AI 모델 (자동 생성)
├── results/                   # 결과 파일 (자동 생성)
└── logs/                      # 로그 파일 (자동 생성)
```

## 📖 사용 방법

### 전체 런처 실행

```bash
python main_launcher.py
```

### 개별 프로세스 실행

```bash
# 프로세스 1: 라벨링
python run_process1.py

# 프로세스 2: 모델 학습
python run_process2.py

# 프로세스 3: AI 예측 및 수정
python run_process3.py
```

## 🔄 워크플로우

### 1단계: 데이터 라벨링 (프로세스 1)

1. **프로세스 1** 실행
2. **Shapefile 선택** 버튼 클릭
3. 자동으로 검출된 **교차점(빨간색)** 확인
4. 수동으로 라벨링:
   - **좌클릭**: 커브 추가 (파란색)
   - **우클릭**: 끝점 추가 (녹색)
   - **Shift+클릭**: 포인트 제거
   - **D키**: 가장 가까운 점 삭제
5. **세션 저장** 클릭
6. 100개 파일 반복

#### 배치 처리
- **폴더 일괄 처리 시작** 클릭
- 여러 파일의 교차점을 자동으로 검출

### 2단계: 모델 학습 (프로세스 2)

1. **프로세스 2** 실행
2. 라벨링된 세션 목록에서 학습할 데이터 선택
   - **전체 선택** 가능
   - 커브나 끝점이 라벨링된 세션만 표시됨
3. 학습 설정:
   - **에폭 수**: 50 (기본값)
   - **학습률**: 0.001
   - **배치 크기**: 64
   - **네트워크**: Medium (권장)
4. **학습 시작** 클릭
5. 실시간 그래프로 진행 상황 확인
6. 학습 완료 후 모델 자동 저장

### 3단계: AI 예측 및 개선 (프로세스 3)

1. **프로세스 3** 실행
2. 학습된 모델 선택
3. **Shapefile 선택** → **AI 예측 실행**
4. AI 예측 결과 확인:
   - **보라색 다이아몬드**: AI 교차점
   - **주황색 삼각형**: AI 커브
   - **갈색 육각형**: AI 끝점
5. 결과 수정:
   - **모든 AI 예측 수락**: AI 결과를 그대로 사용
   - 수동으로 추가/제거 가능
6. **수정된 세션 저장**
7. 여러 파일 수정 후 **수정 데이터로 재학습**

## 💡 팁과 요령

### 라벨링 팁
- 스켈레톤(중심선)에 자동으로 스냅됩니다
- 레이어 컨트롤에서 투명도 조절 가능
- Space키로 화면 맞춤
- 1,2,3키로 각 레이어 토글

### 학습 팁
- 최소 50개 이상의 라벨링된 파일 권장
- 다양한 도로 형태 포함 필요
- 학습 중 정확도가 떨어지면 조기 종료됨

### AI 예측 팁
- 첫 예측 후 수정 → 재학습으로 성능 향상
- 배치 처리로 여러 파일 한 번에 처리
- CSV 내보내기로 통계 분석 가능

## 🐛 문제 해결

### ImportError 발생
```bash
# src 폴더가 Python 경로에 없을 때
set PYTHONPATH=%PYTHONPATH%;I:\gcp_rl
```

### CUDA 오류
```bash
# CPU 버전으로 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

### 메모리 부족
- 배치 크기 줄이기 (32 또는 16)
- 스켈레톤 해상도 낮추기 (config.py에서 width 조정)

### 파일 인코딩 오류
- Shapefile이 UTF-8 인코딩인지 확인
- 한글 경로 피하기

## 📊 결과 파일

### 세션 파일 (sessions/*.json)
```json
{
  "session_id": "road_20241230_143022",
  "file_path": "road.shp",
  "labels": {
    "intersection": [[x1, y1], ...],
    "curve": [[x2, y2], ...],
    "endpoint": [[x3, y3], ...]
  },
  "metadata": {...}
}
```

### 모델 파일 (models/*.pth)
- PyTorch 모델 파일
- 메타데이터 포함 (.json)

### 결과 CSV (results/*.csv)
- 배치 처리 결과
- 파일별 검출 통계

## 📞 지원

문제가 발생하면:
1. logs/ 폴더의 로그 파일 확인
2. 이메일: ksw3037@lx.or.kr
3. GitHub Issues (있는 경우)

## 🔄 업데이트 예정

- [ ] 웹 기반 UI
- [ ] 더 많은 도로 특징 (회전교차로, 램프 등)
- [ ] 실시간 협업 기능
- [ ] 성능 최적화