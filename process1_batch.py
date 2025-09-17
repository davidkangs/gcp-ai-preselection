"""
프로세스 1 - 배치 처리 다이얼로그
여러 shapefile을 순차적으로 처리하고 라벨링
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QListWidget, QListWidgetItem, QProgressBar, QGroupBox,
    QTextEdit, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

sys.path.append(str(Path(__file__).parent.parent))
from src.core.skeleton_extractor import SkeletonExtractor
from src.ui.canvas_widget import CanvasWidget
from src.utils import save_session

import logging
logger = logging.getLogger(__name__)


class BatchWorker(QThread):
    """배치 처리 워커 스레드"""
    file_started = pyqtSignal(str)
    file_completed = pyqtSignal(str, dict)
    progress_updated = pyqtSignal(int, str)
    error_occurred = pyqtSignal(str, str)
    
    def __init__(self, file_list, auto_save=True):
        super().__init__()
        self.file_list = file_list
        self.auto_save = auto_save
        self.skeleton_extractor = SkeletonExtractor()
        self.is_running = True
    
    def run(self):
        """배치 처리 실행"""
        total_files = len(self.file_list)
        
        for idx, file_path in enumerate(self.file_list):
            if not self.is_running:
                break
            
            try:
                # 파일 처리 시작
                self.file_started.emit(file_path)
                
                # 진행률 업데이트
                progress = int((idx / total_files) * 100)
                self.progress_updated.emit(
                    progress, 
                    f"처리 중 ({idx + 1}/{total_files}): {Path(file_path).name}"
                )
                
                # 스켈레톤 추출
                skeleton, intersections = self.skeleton_extractor.process_shapefile(file_path)
                
                # 결과 구성
                result = {
                    'success': True,
                    'skeleton': skeleton,
                    'intersections': intersections,
                    'skeleton_count': len(skeleton),
                    'intersection_count': len(intersections)
                }
                
                # 자동 저장
                if self.auto_save:
                    labels = {
                        'intersection': [(float(x), float(y)) for x, y in intersections],
                        'curve': [],
                        'endpoint': []
                    }
                    
                    metadata = {
                        'process': 'batch_labeling',
                        'auto_processed': True,
                        'skeleton_points': len(skeleton),
                        'detected_intersections': len(intersections)
                    }
                    
                    user_actions = []
                    save_session(file_path, labels, skeleton, metadata, user_actions)
                
                # 완료 신호
                self.file_completed.emit(file_path, result)
                
            except Exception as e:
                logger.error(f"파일 처리 오류 {file_path}: {e}")
                self.error_occurred.emit(file_path, str(e))
                
                result = {
                    'success': False,
                    'error': str(e)
                }
                self.file_completed.emit(file_path, result)
        
        self.progress_updated.emit(100, "배치 처리 완료")
    
    def stop(self):
        """처리 중지"""
        self.is_running = False


class BatchProcessingDialog(QDialog):
    """배치 처리 다이얼로그"""
    
    def __init__(self, folder_path, parent=None):
        super().__init__(parent)
        self.folder_path = folder_path
        self.file_list = []
        self.current_file_idx = 0
        self.results = {}
        self.worker = None
        
        self.init_ui()
        self.load_shapefiles()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("배치 처리 - 교차로 자동 검출")
        self.setGeometry(200, 200, 800, 600)
        self.setModal(True)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 폴더 정보
        folder_label = QLabel(f"폴더: {self.folder_path}")
        layout.addWidget(folder_label)
        
        # 파일 목록 그룹
        file_group = QGroupBox("Shapefile 목록")
        file_layout = QVBoxLayout()
        
        self.file_list_widget = QListWidget()
        file_layout.addWidget(self.file_list_widget)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 진행 상황
        progress_group = QGroupBox("진행 상황")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("대기 중...")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # 로그
        log_group = QGroupBox("처리 로그")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # 옵션
        option_layout = QHBoxLayout()
        
        self.auto_save_checkbox = QCheckBox("자동 세션 저장")
        self.auto_save_checkbox.setChecked(True)
        option_layout.addWidget(self.auto_save_checkbox)
        
        self.continue_on_error_checkbox = QCheckBox("오류 시 계속 진행")
        self.continue_on_error_checkbox.setChecked(True)
        option_layout.addWidget(self.continue_on_error_checkbox)
        
        layout.addLayout(option_layout)
        
        # 버튼
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("처리 시작")
        self.start_btn.clicked.connect(self.start_processing)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("중지")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        self.close_btn = QPushButton("닫기")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # 요약 레이블
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #f0f0f0;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.summary_label)
    
    def load_shapefiles(self):
        """폴더에서 shapefile 목록 로드"""
        self.file_list_widget.clear()
        self.file_list = []
        
        # .shp 파일 찾기
        for file_path in Path(self.folder_path).glob("*.shp"):
            self.file_list.append(str(file_path))
            
            item = QListWidgetItem(file_path.name)
            item.setData(Qt.UserRole, str(file_path))
            self.file_list_widget.addItem(item)
        
        self.status_label.setText(f"{len(self.file_list)}개의 shapefile 발견")
        self.update_summary()
    
    def start_processing(self):
        """배치 처리 시작"""
        if not self.file_list:
            QMessageBox.warning(self, "경고", "처리할 파일이 없습니다.")
            return
        
        # UI 상태 변경
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.close_btn.setEnabled(False)
        
        # 로그 초기화
        self.log_text.clear()
        self.log_text.append(f"=== 배치 처리 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        
        # 워커 스레드 시작
        self.worker = BatchWorker(
            self.file_list, 
            auto_save=self.auto_save_checkbox.isChecked()
        )
        
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_completed.connect(self.on_file_completed)
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.error_occurred.connect(self.on_error_occurred)
        
        self.worker.start()
    
    def stop_processing(self):
        """처리 중지"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            
        self.on_processing_stopped()
    
    def on_file_started(self, file_path):
        """파일 처리 시작"""
        filename = Path(file_path).name
        self.log_text.append(f"\n[시작] {filename}")
        
        # 목록에서 현재 파일 강조
        for i in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(i)
            if item.data(Qt.UserRole) == file_path:
                item.setBackground(Qt.yellow)
                self.file_list_widget.scrollToItem(item)
                break
    
    def on_file_completed(self, file_path, result):
        """파일 처리 완료"""
        filename = Path(file_path).name
        self.results[file_path] = result
        
        # 로그 업데이트
        if result['success']:
            self.log_text.append(
                f"[완료] {filename} - "
                f"스켈레톤: {result['skeleton_count']}점, "
                f"교차점: {result['intersection_count']}개"
            )
            
            # 목록에서 완료 표시
            for i in range(self.file_list_widget.count()):
                item = self.file_list_widget.item(i)
                if item.data(Qt.UserRole) == file_path:
                    item.setBackground(Qt.green)
                    item.setText(f"✓ {filename}")
                    break
        else:
            self.log_text.append(f"[오류] {filename} - {result.get('error', '알 수 없는 오류')}")
            
            # 목록에서 오류 표시
            for i in range(self.file_list_widget.count()):
                item = self.file_list_widget.item(i)
                if item.data(Qt.UserRole) == file_path:
                    item.setBackground(Qt.red)
                    item.setText(f"✗ {filename}")
                    break
        
        self.update_summary()
    
    def on_progress_updated(self, progress, message):
        """진행률 업데이트"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
        
        if progress == 100:
            self.on_processing_completed()
    
    def on_error_occurred(self, file_path, error_msg):
        """오류 발생"""
        filename = Path(file_path).name
        self.log_text.append(f"[오류] {filename}: {error_msg}")
        
        if not self.continue_on_error_checkbox.isChecked():
            self.stop_processing()
    
    def on_processing_completed(self):
        """처리 완료"""
        self.log_text.append(f"\n=== 배치 처리 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        
        # UI 상태 복원
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        
        # 결과 요약
        success_count = sum(1 for r in self.results.values() if r['success'])
        fail_count = len(self.results) - success_count
        
        QMessageBox.information(
            self, "완료",
            f"배치 처리가 완료되었습니다.\n\n"
            f"전체: {len(self.results)}개\n"
            f"성공: {success_count}개\n"
            f"실패: {fail_count}개"
        )
    
    def on_processing_stopped(self):
        """처리 중지됨"""
        self.log_text.append("\n[중지] 사용자에 의해 중지됨")
        
        # UI 상태 복원
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        
        self.status_label.setText("처리 중지됨")
    
    def update_summary(self):
        """요약 정보 업데이트"""
        total = len(self.file_list)
        processed = len(self.results)
        success = sum(1 for r in self.results.values() if r['success'])
        fail = processed - success
        
        summary = f"전체: {total} | 처리: {processed} | 성공: {success} | 실패: {fail}"
        
        if processed > 0:
            # 전체 통계
            total_skeletons = sum(
                r['skeleton_count'] for r in self.results.values() 
                if r['success']
            )
            total_intersections = sum(
                r['intersection_count'] for r in self.results.values() 
                if r['success']
            )
            
            summary += f"\n총 스켈레톤: {total_skeletons:,}점 | 총 교차점: {total_intersections:,}개"
        
        self.summary_label.setText(summary)
    
    def closeEvent(self, event):
        """다이얼로그 닫기"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "확인",
                "처리가 진행 중입니다. 중지하고 닫으시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
            
            self.stop_processing()
        
        event.accept()