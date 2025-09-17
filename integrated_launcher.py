#!/usr/bin/env python3
"""
통합 실행 스크립트
프로세스 1, 2를 통합 관리
"""

import sys
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap

class LauncherDialog(QDialog):
    """런처 다이얼로그"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 도로망 AI 분석 시스템")
        self.setFixedSize(500, 300)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #667eea, stop:1 #764ba2);
            }
            QPushButton {
                background-color: rgba(255, 255, 255, 0.9);
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: white;
                transform: scale(1.05);
            }
            QLabel {
                color: white;
                font-weight: bold;
            }
        """)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 타이틀
        title = QLabel("🤖 도로망 AI 분석 시스템")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; margin: 20px;")
        layout.addWidget(title)
        
        # 설명
        desc = QLabel("AI 기반 도로망 교차점/커브/끝점 자동 검출 시스템")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("font-size: 14px; margin-bottom: 30px;")
        layout.addWidget(desc)
        
        # 버튼들
        button_layout = QVBoxLayout()
        
        # 프로세스 1 버튼
        process1_btn = QPushButton("📝 프로세스 1: 라벨링 도구\n(향상된 휴리스틱 + 수동 편집)")
        process1_btn.clicked.connect(self.launch_process1)
        button_layout.addWidget(process1_btn)
        
        # 프로세스 2 버튼  
        process2_btn = QPushButton("🧠 프로세스 2: DQN 학습 도구\n(Session 데이터 → AI 모델)")
        process2_btn.clicked.connect(self.launch_process2)
        button_layout.addWidget(process2_btn)
        
        # 통합 모드 버튼
        integrated_btn = QPushButton("⚡ 통합 모드: 라벨링 + AI 예측\n(DQN 학습 완료 후 권장)")
        integrated_btn.clicked.connect(self.launch_integrated)
        integrated_btn.setStyleSheet(integrated_btn.styleSheet() + "background-color: rgba(76, 175, 80, 0.9);")
        button_layout.addWidget(integrated_btn)
        
        layout.addLayout(button_layout)
        
        # 하단 정보
        info = QLabel("💡 워크플로우: 프로세스 1 → 라벨링 → 프로세스 2 → 학습 → 통합 모드")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("font-size: 12px; margin-top: 20px;")
        layout.addWidget(info)
        
        self.setLayout(layout)
    
    def launch_process1(self):
        """프로세스 1 실행"""
        try:
            subprocess.Popen([sys.executable, "process1_labeling_tool.py"])
            QMessageBox.information(self, "실행", "프로세스 1: 라벨링 도구가 시작되었습니다!")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"프로세스 1 실행 실패:\n{e}")
    
    def launch_process2(self):
        """프로세스 2 실행"""
        try:
            subprocess.Popen([sys.executable, "process2_training_improved.py"])
            QMessageBox.information(self, "실행", "프로세스 2: DQN 학습 도구가 시작되었습니다!")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"프로세스 2 실행 실패:\n{e}")
    
    def launch_integrated(self):
        """통합 모드 실행"""
        # DQN 모델 존재 여부 확인
        model_path = Path("src/learning/models/session_dqn_model.pth")
        if not model_path.exists():
            QMessageBox.warning(
                self, "경고", 
                "DQN 모델이 없습니다!\n\n"
                "먼저 다음 단계를 완료하세요:\n"
                "1. 프로세스 1에서 라벨링 작업\n"  
                "2. 프로세스 2에서 DQN 학습\n"
                "3. 모델 생성 확인 후 통합 모드 실행"
            )
            return
        
        try:
            subprocess.Popen([sys.executable, "process1_labeling_tool.py"])
            QMessageBox.information(
                self, "통합 모드", 
                "✨ 통합 모드가 시작되었습니다!\n\n"
                "사용법:\n"
                "• 파일 처리 후 Q키로 DQN 예측\n"
                "• T키로 AI 예측 토글\n"
                "• 보라색 = AI 예측 포인트"
            )
        except Exception as e:
            QMessageBox.critical(self, "오류", f"통합 모드 실행 실패:\n{e}")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    launcher = LauncherDialog()
    launcher.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
