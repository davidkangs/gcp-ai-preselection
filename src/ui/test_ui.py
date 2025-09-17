"""UI 테스트"""
import sys
from pathlib import Path

# 부모 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.main_window import main

if __name__ == "__main__":
    main()