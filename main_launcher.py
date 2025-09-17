"""
도로망 AI 분석 시스템 - 메인 런처
3개의 프로세스를 선택하여 실행할 수 있는 통합 런처
"""

import sys
import os
from pathlib import Path
import subprocess

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QProcess, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor

# 프로세스 모듈 경로 추가
sys.path.append(str(Path(__file__).parent))


class ProcessLauncher(QWidget):
    """개별 프로세스 런처 위젯"""
    
    def __init__(self, process_name, process_file, description, color):
        super().__init__()
        self.process_name = process_name
        self.process_file = process_file
        self.process = None
        
        self.init_ui(description, color)
    
    def init_ui(self, description, color):
        """UI 초기화"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 프로세스 이름
        title_label = QLabel(self.process_name)
        title_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                padding: 10px;
                font-size: 16pt;
                font-weight: bold;
                border-radius: 5px;
            }}
        """)
        layout.addWidget(title_label)
        
        # 설명
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        layout.addWidget(desc_label)
        
        # 버튼
        button_layout = QHBoxLayout()
        
        self.launch_btn = QPushButton("실행")
        self.launch_btn.clicked.connect(self.launch_process)
        self.launch_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                padding: 8px 20px;
                font-weight: bold;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {self.darken_color(color)};
            }}
        """)
        button_layout.addWidget(self.launch_btn)
        
        self.stop_btn = QPushButton("중지")
        self.stop_btn.clicked.connect(self.stop_process)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        # 상태
        self.status_label = QLabel("대기 중")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 5px;
                background-color: #e0e0e0;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.status_label)
    
    def darken_color(self, color):
        """색상을 어둡게"""
        c = QColor(color)
        c.setHsv(c.hue(), c.saturation(), int(c.value() * 0.8))
        return c.name()
    
    def launch_process(self):
        """프로세스 실행"""
        if self.process and self.process.state() == QProcess.Running:
            QMessageBox.warning(self, "경고", f"{self.process_name}이(가) 이미 실행 중입니다.")
            return
        
        # QProcess 생성
        self.process = QProcess()
        self.process.started.connect(self.on_process_started)
        self.process.finished.connect(self.on_process_finished)
        self.process.errorOccurred.connect(self.on_process_error)
        
        # Python 실행
        python_exe = sys.executable
        script_path = Path(__file__).parent / self.process_file
        
        self.process.start(python_exe, [str(script_path)])
    
    def stop_process(self):
        """프로세스 중지"""
        if self.process and self.process.state() == QProcess.Running:
            self.process.terminate()
            if not self.process.waitForFinished(5000):  # 5초 대기
                self.process.kill()  # 강제 종료
    
    def on_process_started(self):
        """프로세스 시작됨"""
        self.launch_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("실행 중...")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 5px;
                background-color: #c8e6c9;
                border-radius: 3px;
            }
        """)
    
    def on_process_finished(self, exit_code, exit_status):
        """프로세스 종료됨"""
        self.launch_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if exit_status == QProcess.NormalExit and exit_code == 0:
            self.status_label.setText("정상 종료")
            self.status_label.setStyleSheet("""
                QLabel {
                    padding: 5px;
                    background-color: #e0e0e0;
                    border-radius: 3px;
                }
            """)
        else:
            self.status_label.setText(f"비정상 종료 (코드: {exit_code})")
            self.status_label.setStyleSheet("""
                QLabel {
                    padding: 5px;
                    background-color: #ffcdd2;
                    border-radius: 3px;
                }
            """)
    
    def on_process_error(self, error):
        """프로세스 오류"""
        self.launch_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"오류 발생: {error}")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 5px;
                background-color: #ffcdd2;
                border-radius: 3px;
            }
        """)
        
        QMessageBox.critical(self, "오류", f"{self.process_name} 실행 중 오류가 발생했습니다.")


class MainLauncher(QMainWindow):
    """메인 런처 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("도로망 AI 분석 시스템 - 메인 런처")
        self.setGeometry(200, 200, 900, 700)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 헤더
        header_label = QLabel("도로망 AI 분석 시스템")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("""
            QLabel {
                background-color: #1976D2;
                color: white;
                padding: 20px;
                font-size: 24pt;
                font-weight: bold;
            }
        """)
        main_layout.addWidget(header_label)
        
        # 프로세스 설명
        info_label = QLabel(
            "도로망에서 교차점, 커브, 끝점을 검출하는 AI 시스템입니다.\n"
            "아래 3개의 프로세스를 순서대로 실행하세요."
        )
        info_label.setStyleSheet("""
            QLabel {
                padding: 15px;
                background-color: #E3F2FD;
                border: 1px solid #BBDEFB;
                font-size: 11pt;
            }
        """)
        main_layout.addWidget(info_label)
        
        # 프로세스 런처들
        process_group = QGroupBox("프로세스 선택")
        process_layout = QVBoxLayout()
        
        # 프로세스 1
        process1 = ProcessLauncher(
            "프로세스 1: 라벨링",
            "process1_labeling_tool.py",
            "• 휴리스틱으로 교차점 자동 검출\n"
            "• 인간이 커브와 끝점을 수동 라벨링\n"
            "• 세션 저장 및 배치 처리 지원",
            "#4CAF50"
        )
        process_layout.addWidget(process1)
        
        # 프로세스 2
        process2 = ProcessLauncher(
            "프로세스 2: 모델 학습",
            "process2_training.py",
            "• 수집된 라벨링 데이터로 DQN 모델 학습\n"
            "• 학습 진행 상황 실시간 모니터링\n"
            "• 최적 모델 자동 저장",
            "#FF9800"
        )
        process_layout.addWidget(process2)
        
        # 프로세스 3
        process3 = ProcessLauncher(
            "프로세스 3: AI 예측 및 개선",
            "process3_inference.py",
            "• 학습된 모델로 자동 검출\n"
            "• 인간이 AI 결과 수정 및 개선\n"
            "• 수정 데이터로 재학습 (Fine-tuning)",
            "#9C27B0"
        )
        process_layout.addWidget(process3)
        
        process_group.setLayout(process_layout)
        main_layout.addWidget(process_group)
        
        # 도움말
        help_group = QGroupBox("워크플로우 가이드")
        help_layout = QVBoxLayout()
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setMaximumHeight(150)
        help_text.setHtml("""
<h3>🔄 전체 워크플로우</h3>
<ol>
<li><b>프로세스 1</b>: 100개 정도의 도로망 shapefile에 대해 라벨링 수행</li>
<li><b>프로세스 2</b>: 라벨링된 데이터로 AI 모델 학습 (DQN)</li>
<li><b>프로세스 3</b>: AI가 자동 검출 → 인간이 수정 → 재학습으로 성능 개선</li>
</ol>

<h3>📁 필요 파일</h3>
<ul>
<li>Shapefile (.shp, .shx, .dbf, .prj)</li>
<li>한국 좌표계 권장 (EPSG:5186)</li>
</ul>

<h3>💾 결과물</h3>
<ul>
<li><b>sessions/</b>: 라벨링 세션 파일 (JSON)</li>
<li><b>models/</b>: 학습된 AI 모델 (PTH)</li>
<li><b>results/</b>: 배치 처리 결과 (CSV)</li>
</ul>
        """)
        help_layout.addWidget(help_text)
        
        help_group.setLayout(help_layout)
        main_layout.addWidget(help_group)
        
        # 하단 버튼
        bottom_layout = QHBoxLayout()
        
        folder_btn = QPushButton("폴더 열기")
        folder_btn.clicked.connect(self.open_folders)
        bottom_layout.addWidget(folder_btn)
        
        bottom_layout.addStretch()
        
        about_btn = QPushButton("정보")
        about_btn.clicked.connect(self.show_about)
        bottom_layout.addWidget(about_btn)
        
        main_layout.addLayout(bottom_layout)
        
        # 상태바
        self.statusBar().showMessage("준비됨")
    
    def open_folders(self):
        """결과 폴더 열기"""
        import webbrowser
        folders = ['sessions', 'models', 'results']
        
        for folder in folders:
            folder_path = Path(folder)
            folder_path.mkdir(exist_ok=True)
        
        # 탐색기 열기
        webbrowser.open(str(Path.cwd()))
    
    def show_about(self):
        """프로그램 정보"""
        QMessageBox.about(
            self, "정보",
            "도로망 AI 분석 시스템 v1.0\n\n"
            "개발자: 강상우\n"
            "소속: 한국국토정보공사\n"
            "이메일: ksw3037@lx.or.kr\n\n"
            "이 시스템은 도로망 shapefile에서\n"
            "교차점, 커브, 끝점을 자동으로 검출하는\n"
            "AI 기반 GIS 분석 도구입니다."
        )


def main():
    """메인 실행 함수"""
    # High DPI 지원
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 다크 테마 적용 (선택사항)
    # palette = QPalette()
    # palette.setColor(QPalette.Window, QColor(53, 53, 53))
    # palette.setColor(QPalette.WindowText, Qt.white)
    # app.setPalette(palette)
    
    # 메인 윈도우
    window = MainLauncher()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    # 필수 디렉토리 생성
    for folder in ['sessions', 'models', 'results', 'logs', 'cache']:
        Path(folder).mkdir(exist_ok=True)
    
    main()