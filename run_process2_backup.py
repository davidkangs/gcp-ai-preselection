# ===== run_process2.py =====
"""프로세스 2 실행 - 새로운 DQN 학습 시스템"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 필수 디렉토리 생성
for folder in ['src/learning/models', 'data/training_samples', 'logs']:
    Path(folder).mkdir(parents=True, exist_ok=True)

print("✨ 새로운 DQN 학습 시스템 실행")
print("📁 폴더 구조 확인 완료")

# 새로운 프로세스 2 실행
from process2_training import main

if __name__ == '__main__':
    main()
