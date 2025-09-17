# 📦 CUDA 업그레이드 패키지

PyTorch CPU 버전을 CUDA 버전으로 안전하게 업그레이드하기 위한 완전한 가이드 및 도구

## 📁 포함된 파일들

| 파일명 | 설명 | 사용법 |
|--------|------|--------|
| `CUDA_UPGRADE_GUIDE.md` | 📖 상세한 업그레이드 가이드 | 단계별 설명과 문제 해결 |
| `upgrade_to_cuda.bat` | 🚀 자동 업그레이드 스크립트 | 더블클릭으로 실행 |
| `rollback_to_cpu.bat` | 🔄 롤백 스크립트 | 문제 시 CPU 버전으로 복구 |

## 🚀 빠른 시작

### 1. 자동 업그레이드 (권장)
```bash
# 더블클릭하거나 터미널에서
upgrade_to_cuda.bat
```

### 2. 수동 업그레이드
```bash
# 백업
pip freeze > backup.txt

# 제거 및 설치
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# 확인
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 🔄 문제 발생 시

### 자동 롤백
```bash
rollback_to_cpu.bat
```

### 수동 롤백
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

## ✅ 성공 확인 방법

### 1. Python에서 확인
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**기대 결과:**
```
PyTorch: 2.1.0+cu121
CUDA: True
GPU: NVIDIA RTX A6000
```

### 2. 프로젝트에서 확인
```bash
python test_installation.py
```
→ GPU 지원 항목이 ✅로 표시

### 3. Process3에서 확인
```
INFO:src.learning.dqn_model:DQN 에이전트 초기화 - 디바이스: cuda
```

## 📈 성능 향상 기대치

| 항목 | CPU | GPU (RTX A6000) | 개선율 |
|------|-----|------------------|---------|
| 학습 시간 | 1시간 | 5-10분 | 10-100배 |
| 추론 시간 | 10초 | 1초 미만 | 10배+ |
| 배치 크기 | 64 | 512+ | 8배+ |
| VRAM 사용 | N/A | 48GB | 대용량 처리 |

## 🛠️ 환경 정보

- **Python**: 3.11.4
- **GPU**: NVIDIA RTX A6000 (48GB)
- **CUDA 드라이버**: 12.8
- **권장 CUDA**: 12.1 (호환성)
- **OS**: Windows 10

## 🔍 문제 해결

### 자주 발생하는 문제

1. **"CUDA out of memory"**
   - `configs/dqn_config.py`에서 `batch_size` 줄이기
   - 64 → 32 → 16으로 단계적 감소

2. **"RuntimeError: No CUDA GPUs available"**
   - GPU 드라이버 재설치
   - nvidia-smi 명령어 확인

3. **기타 호환성 문제**
   - `rollback_to_cpu.bat` 실행
   - 원래 환경으로 복구 후 문의

## 💡 팁

- **성능 모니터링**: `nvidia-smi -l 1` 명령어로 GPU 사용률 실시간 확인
- **메모리 관리**: 큰 배치 크기 사용으로 GPU 성능 최대화
- **안전성**: 항상 백업 파일 유지하여 롤백 가능하도록 준비

---

**작성일**: 2025년 1월 20일  
**버전**: 1.0  
**상태**: 검증 완료 