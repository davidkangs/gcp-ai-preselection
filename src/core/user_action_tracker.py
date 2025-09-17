"""
사용자 편집 행동 추적기
"""

import time
import threading
from collections import deque
from PyQt5.QtCore import QObject, pyqtSignal
from datetime import datetime

class UserActionTracker(QObject):
    """사용자 편집 행동 시간순 추적"""
    
    def __init__(self, max_events=5000):
        super().__init__()
        self.max_events = max_events
        self.current_session = None
        self.events = deque(maxlen=max_events)
        self.lock = threading.RLock()
        self.session_start_time = None
    
    def start_session(self, file_path):
        """세션 시작"""
        with self.lock:
            self.current_session = file_path
            self.session_start_time = time.time()
            self.events.clear()
    
    def record_action(self, action_type, category, x, y, additional_data=None):
        """사용자 행동 기록"""
        if not self.current_session:
            return
        
        with self.lock:
            event = {
                'timestamp': time.time(),
                'relative_time': time.time() - self.session_start_time,
                'action': action_type,
                'category': category,
                'position': [float(x), float(y)],
                'data': additional_data or {}
            }
            self.events.append(event)
    
    def get_session_actions(self):
        """현재 세션의 모든 행동 반환"""
        with self.lock:
            return list(self.events)
    
    def end_session(self):
        """세션 종료"""
        with self.lock:
            actions = list(self.events)
            self.current_session = None
            self.events.clear()
            return actions
