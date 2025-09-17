# ===== run_process3.py =====
"""프로세스 3 실행 - AI 예측 및 수정"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# 필수 디렉토리 생성
for folder in ['sessions', 'models', 'results', 'logs']:
    Path(folder).mkdir(exist_ok=True)

# 프로세스 3 실행
from process3_inference import main
main()