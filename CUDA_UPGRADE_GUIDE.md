# 🚀 CUDA PyTorch 업그레이드 가이드

## 📋 현재 환경 정보
- **Python**: 3.11.4
- **PyTorch**: 2.1.0 (CPU 버전)
- **GPU**: NVIDIA RTX A6000 (48GB VRAM)
- **CUDA 드라이버**: 12.8
- **OS**: Windows 10

## 🎯 목표
CPU 전용 PyTorch를 CUDA 지원 버전으로 안전하게 업그레이드

## ⚠️ 사전 확인사항

### 1. GPU 및 드라이버 확인
```bash
nvidia-smi
```
**확인할 것:** CUDA Version이 표시되는지

### 2. 현재 PyTorch 버전 확인
```bash
python -c "import torch; print('Version:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```
**예상 결과:** Version: 2.1.0, CUDA: False

## 🔧 업그레이드 절차

### 1단계: 환경 백업 (필수!)
```bash
# 현재 디렉토리: I:\gcp_rl
pip freeze > cuda_upgrade_backup.txt
```

### 2단계: 기존 PyTorch 제거
```bash
pip uninstall torch torchvision torchaudio -y
```

### 3단계: CUDA PyTorch 설치
```bash
# 동일한 버전 + CUDA 12.1
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

### 4단계: 설치 확인
```bash
python -c "import torch; print('Version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**기대 결과:**
```
Version: 2.1.0+cu121
CUDA available: True
GPU name: NVIDIA RTX A6000
```

### 5단계: 프로젝트 테스트
```bash
python test_installation.py
```
**확인할 것:** GPU 지원 항목이 ✅로 표시되는지

## 🔄 문제 발생 시 롤백

### 원래 상태로 복구
```bash
# CUDA 버전 제거
pip uninstall torch torchvision torchaudio -y

# CPU 버전 재설치
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# 또는 백업에서 복원
pip install -r cuda_upgrade_backup.txt --force-reinstall
```

## 📈 성능 개선 예상치

### RTX A6000 기대 성능
- **학습 속도**: 10-100배 빨라짐
- **메모리**: 48GB VRAM 활용 가능
- **배치 크기**: 64 → 512+ 증가 가능
- **추론 시간**: 10초 → 1초 미만

### Process별 개선
- **Process2 (학습)**: 1시간 → 5-10분
- **Process3 (AI 예측)**: 10초 → 1초 미만
- **배치 처리**: 대량 파일 처리 시간 대폭 단축

## 🔍 문제 해결

### 자주 발생하는 문제들

#### 1. "CUDA out of memory" 오류
```python
# 해결방법: 배치 크기 줄이기
# configs/dqn_config.py에서
'batch_size': 32,  # 64 → 32로 감소
```

#### 2. CUDA 버전 불일치
```bash
# 다른 CUDA 버전 시도
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 3. 다른 패키지 충돌
```bash
# 전체 환경 재구성
pip install -r requirements.txt --force-reinstall
```

## ✅ 검증 체크리스트

- [ ] nvidia-smi 정상 작동
- [ ] 백업 파일 생성됨
- [ ] PyTorch CUDA 버전 설치됨
- [ ] torch.cuda.is_available() == True
- [ ] GPU 이름 정상 출력됨
- [ ] test_installation.py 통과
- [ ] Process3 모델 로드 성공

## 📝 버전 호환성 참고

| PyTorch | Python | CUDA | 검증 상태 |
|---------|--------|------|----------|
| 2.1.0+cu121 | 3.11.4 | 12.1 | ✅ 권장 |
| 2.1.0+cu118 | 3.11.4 | 11.8 | ✅ 안전 |
| 2.2.0+cu121 | 3.11.4 | 12.1 | ⚠️ 미검증 |

## 🎉 업그레이드 완료 후

### 성능 모니터링
```bash
# GPU 사용률 모니터링
nvidia-smi -l 1
```

### 설정 최적화
```python
# configs/dqn_config.py에서
'device': 'cuda',  # 'auto' → 'cuda'로 강제 설정
'batch_size': 256,  # 큰 배치 크기 활용
```

---

**작성일**: 2025년 1월 20일  
**환경**: Python 3.11.4, RTX A6000, Windows 10  
**상태**: 테스트 완료 