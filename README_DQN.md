# 🤖 도로망 AI 분석 시스템 - DQN 통합 버전

## 🚀 빠른 시작

### 1. 통합 실행기 사용 (권장)
```bash
python integrated_launcher.py
```

### 2. 개별 실행
```bash
# 프로세스 1: 라벨링
python process1_labeling_tool.py

# 프로세스 2: DQN 학습  
python process2_training_improved.py
```

## 📋 워크플로우

1. **프로세스 1에서 라벨링**
   - Shapefile 로드
   - 향상된 휴리스틱으로 자동 검출
   - 수동 편집으로 정확도 향상
   - Session 저장

2. **프로세스 2에서 DQN 학습**
   - Session 파일들 스캔
   - DQN 학습 데이터 변환
   - 신경망 모델 학습
   - 모델 저장

3. **통합 모드에서 AI 예측**
   - 기존 라벨링 + DQN 예측 
   - Q키로 DQN 예측 실행
   - T키로 AI 예측 토글

## 🎯 DQN 예측 클래스

- **0: 일반 포인트** (예측 안 함)
- **1: 교차점** (빨간색 다이아몬드)
- **2: 커브** (주황색 삼각형)  
- **3: 끝점** (갈색 육각형)

## ⌨️ 키보드 단축키

- `Q`: DQN 예측 실행
- `T`: AI 예측 토글
- `Space`: 화면 맞춤
- `D`: 포인트 삭제
- `1/2/3`: 레이어 토글

## 📁 파일 구조

```
├── integrated_launcher.py      # 통합 실행기
├── process1_labeling_tool.py   # 라벨링 도구
├── process2_training_improved.py # DQN 학습 도구
├── sessions/                   # Session 파일들
├── src/learning/
│   ├── session_predictor.py   # DQN 예측기
│   └── models/                # 학습된 모델들
└── dqn_config.json           # 설정 파일
```

## 🔧 설정

`dqn_config.json`에서 설정 변경 가능:
- DQN 모델 경로
- 신뢰도 임계값  
- 학습 파라미터

## 🆘 문제 해결

### DQN 예측이 안 될 때
1. 프로세스 2에서 학습 완료했는지 확인
2. `src/learning/models/session_dqn_model.pth` 파일 존재 확인
3. Session 파일이 충분한지 확인 (최소 3-5개 권장)

### 학습이 안 될 때  
1. `sessions/` 디렉토리에 `session_*.json` 파일들 존재 확인
2. 각 Session에 라벨링된 포인트가 있는지 확인
3. 최소 100개 이상의 라벨링된 포인트 권장

---
💡 **팁**: 더 정확한 AI 예측을 위해 다양한 도로망에서 충분한 라벨링을 수행하세요!
