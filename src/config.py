"""프로젝트 설정"""
import os
from pathlib import Path

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" 
RESULTS_DIR = PROJECT_ROOT / "results"
SESSION_DIR = PROJECT_ROOT / "sessions"

# 디렉토리 생성
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, SESSION_DIR]:
    dir_path.mkdir(exist_ok=True)

# 도로 분석 설정
SKELETON_WIDTH = 1200
INTERSECTION_THRESHOLD = 3
CLUSTER_DISTANCE = 5.0
BUFFER_PIXELS = 15

# UI 설정
CANVAS_SIZE = (1200, 800)
COLORS = {
    'intersection': '#FF0000',     # 빨강
    'curve': '#0000FF',           # 파랑  
    'endpoint': '#00FF00',        # 초록
    'skeleton': '#808080',        # 회색
    'ai_suggestion': '#FF00FF'    # 보라 (AI 제안)
}

# 학습 설정
LEARNING_RATE = 0.001
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000