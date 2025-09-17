"""프로세스 1 실행 - 라벨링 도구"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# 필수 디렉토리 생성
for folder in ['sessions', 'models', 'results', 'logs']:
    Path(folder).mkdir(exist_ok=True)

# 프로세스 1 실행
from process1_labeling_tool import main
main()