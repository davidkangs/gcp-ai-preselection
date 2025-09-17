# main_window.py - DQN 데이터 수집 연동 (Import 수정)

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import geopandas as gpd

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# PyQt5 imports
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QProgressBar, QFileDialog, QMessageBox,
    QAction, QToolBar, QCheckBox, QDialog, QTableWidget, QTableWidgetItem,
    QTextEdit, QListWidgetItem, QGroupBox, QTabWidget, QProgressDialog,
    QApplication, QSpinBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont

# 프로젝트 내부 imports (절대 import로 수정)
try:
    from src.core.skeleton_extractor import SkeletonExtractor
    from src.config import *
    from src.utils import save_session, load_session, list_sessions
except ImportError:
    # 백업: 기본 스켈레톤 추출기
    class SkeletonExtractor:
        def process_shapefile(self, file_path):
            # 간단한 스켈레톤 추출 (실제 구현 필요)
            import geopandas as gpd
            gdf = gpd.read_file(file_path)
            
            # 임시: 모든 좌표를 스켈레톤으로 사용
            skeleton = []
            for geom in gdf.geometry:
                if hasattr(geom, 'coords'):
                    skeleton.extend(list(geom.coords))
                elif hasattr(geom, 'geoms'):
                    for g in geom.geoms:
                        if hasattr(g, 'coords'):
                            skeleton.extend(list(g.coords))
            
            # 임시: 첫 10개 점을 교차점으로 설정
            intersections = skeleton[:10] if len(skeleton) > 10 else skeleton
            
            return skeleton, intersections
    
    # 더미 함수들
    def save_session(file_path, labels, skeleton, metadata=None):
        sessions_dir = Path("sessions")
        sessions_dir.mkdir(exist_ok=True)
        
        session_data = {
            'file_path': str(file_path),
            'labels': labels,
            'skeleton': skeleton,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        session_file = sessions_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, default=str)
            return session_file
        except:
            return None
    
    def load_session(session_path):
        try:
            with open(session_path, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def list_sessions():
        sessions_dir = Path("sessions")
        if not sessions_dir.exists():
            return []
        
        sessions = []
        for session_file in sessions_dir.glob("session_*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                    sessions.append({
                        'path': str(session_file),
                        'file_path': data.get('file_path', ''),
                        'timestamp': data.get('timestamp', ''),
                        'label_counts': {
                            'intersection': len(data.get('labels', {}).get('intersection', [])),
                            'curve': len(data.get('labels', {}).get('curve', [])),
                            'endpoint': len(data.get('labels', {}).get('endpoint', []))
                        }
                    })
            except:
                continue
        
        return sessions

from canvas_widget import CanvasWidget

# ✨ 새로운 DQN 시스템 import
from src.learning import DQNDataCollector, DQNTrainer, DQNPredictor

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 기본 설정
        self.setWindowTitle("도로망 AI 분석 시스템 - DQN 버전")
        self.setGeometry(100, 100, 1200, 800)
        
        # 속성 초기화
        self.folder_path = None
        self.current_file = None
        
        # 스켈레톤 추출기 초기화
        self.skeleton_extractor = SkeletonExtractor()
        
        # ✨ 새로운 DQN 시스템 초기화
        self.dqn_collector = DQNDataCollector()
        self.dqn_trainer = DQNTrainer()
        self.dqn_predictor = DQNPredictor()
        
        # 자동 저장 타이머
        self.auto_save_timer = None
        
        # UI 초기화
        self.init_ui()
        self.setup_dqn_ui()
        
        # 상태바
        self.statusBar().showMessage("새로운 DQN 시스템 준비완료")
        
    def init_ui(self):
        """기본 UI 초기화 - 기존과 동일"""
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 왼쪽 패널
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 오른쪽 패널 (캔버스)
        self.canvas_widget = CanvasWidget(self)
        main_layout.addWidget(self.canvas_widget, 3)
        
        # ✨ DQN 데이터 수집기를 캔버스에 연결
        self.dqn_collector.connect_to_canvas(self.canvas_widget.canvas)
        
    def create_left_panel(self):
        """왼쪽 패널 생성 - 기존과 동일"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 폴더 선택
        folder_btn = QPushButton("폴더 선택")
        folder_btn.clicked.connect(self.select_folder)
        layout.addWidget(folder_btn)
        
        # 현재 폴더 표시
        self.folder_label = QLabel("폴더를 선택하세요")
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)
        
        # 파일 목록
        layout.addWidget(QLabel("Shapefile 목록:"))
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        layout.addWidget(self.file_list)
        
        # 진행 상황
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # 처리 정보
        self.info_label = QLabel("파일을 선택하세요")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def setup_dqn_ui(self):
        """✨ DQN 관련 UI 설정"""
        # 메뉴바에 DQN 메뉴 추가
        dqn_menu = self.menuBar().addMenu("DQN 학습")
        
        # 데이터 수집 관리
        start_collection_action = QAction("데이터 수집 시작", self)
        start_collection_action.triggered.connect(self.start_data_collection)
        dqn_menu.addAction(start_collection_action)
        
        end_collection_action = QAction("데이터 수집 종료", self)
        end_collection_action.triggered.connect(self.end_data_collection)
        dqn_menu.addAction(end_collection_action)
        
        view_data_action = QAction("수집된 데이터 보기", self)
        view_data_action.triggered.connect(self.view_collected_data)
        dqn_menu.addAction(view_data_action)
        
        dqn_menu.addSeparator()
        
        # 모델 학습
        train_action = QAction("모델 학습 시작", self)
        train_action.triggered.connect(self.start_training)
        dqn_menu.addAction(train_action)
        
        # AI 예측
        predict_action = QAction("AI 포인트 예측", self)
        predict_action.triggered.connect(self.run_ai_prediction)
        dqn_menu.addAction(predict_action)
        
        # 툴바에 상태 표시
        dqn_toolbar = self.addToolBar("DQN Status")
        
        self.collection_status_label = QLabel("수집: 대기중")
        dqn_toolbar.addWidget(self.collection_status_label)
        
        self.model_status_label = QLabel("모델: 미학습")
        dqn_toolbar.addWidget(self.model_status_label)
        
    def select_folder(self):
        """폴더 선택 - 기존과 동일"""
        folder = QFileDialog.getExistingDirectory(self, "Shapefile 폴더 선택")
        if folder:
            self.folder_path = folder
            self.folder_label.setText(f"폴더: {folder}")
            self.load_shapefiles()
            
    def load_shapefiles(self):
        """shapefile 목록 로드 - 기존과 동일"""
        self.file_list.clear()
        if not self.folder_path:
            return
            
        shp_files = list(Path(self.folder_path).glob("*.shp"))
        for shp_file in shp_files:
            self.file_list.addItem(str(shp_file))
            
        self.info_label.setText(f"{len(shp_files)}개의 shapefile 발견")
        
    def on_file_selected(self, item):
        """파일 선택 - 기존과 동일"""
        file_path = item.text()
        self.current_file = file_path
        self.process_file(file_path)
        
    def process_file(self, file_path):
        """파일 처리 - 기존과 동일, DQN 세션 시작 추가"""
        try:
            self.canvas_widget.clear_all()
            self.progress_bar.setValue(0)
            self.info_label.setText("처리 중...")
            
            # 도로 데이터 읽기
            gdf = gpd.read_file(file_path)
            self.canvas_widget.set_road_data(gdf)
            
            # 스켈레톤 추출
            start_time = time.time()
            skeleton, intersections = self.skeleton_extractor.process_shapefile(file_path)
            processing_time = time.time() - start_time
            
            # 캔버스에 데이터 설정
            self.canvas_widget.current_file = file_path
            self.canvas_widget.skeleton = skeleton
            self.canvas_widget.processing_time = processing_time
            
            # 교차점 설정
            self.canvas_widget.canvas.points = {
                'intersection': [(float(x), float(y)) for x, y in intersections],
                'curve': [],
                'endpoint': []
            }
            
            # 화면 업데이트
            self.canvas_widget.update_display()
            
            # ✨ DQN 세션 시작
            self.dqn_collector.start_session(file_path, skeleton, self.canvas_widget.canvas.points)
            self.collection_status_label.setText("수집: 활성")
            
            # 정보 업데이트
            self.info_label.setText(
                f"처리 완료!\n"
                f"스켈레톤 포인트: {len(skeleton)}\n"
                f"교차점: {len(intersections)}개\n"
                f"처리 시간: {processing_time:.2f}초"
            )
            self.progress_bar.setValue(100)
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"파일 처리 실패:\n{str(e)}")
            print(f"처리 오류: {e}")
            import traceback
            traceback.print_exc()
            
    def start_data_collection(self):
        """✨ 데이터 수집 시작"""
        if not self.current_file:
            QMessageBox.warning(self, "경고", "먼저 파일을 선택하세요.")
            return
        
        # 이미 시작된 경우 재시작
        if hasattr(self.canvas_widget, 'skeleton'):
            self.dqn_collector.start_session(
                self.current_file, 
                self.canvas_widget.skeleton, 
                self.canvas_widget.canvas.points
            )
            self.collection_status_label.setText("수집: 활성")
            QMessageBox.information(self, "정보", "데이터 수집이 시작되었습니다.\n포인트를 추가/제거하면 학습 데이터가 수집됩니다.")
    
    def end_data_collection(self):
        """✨ 데이터 수집 종료"""
        saved_file = self.dqn_collector.end_session()
        if saved_file:
            self.collection_status_label.setText("수집: 대기중")
            QMessageBox.information(self, "완료", f"데이터 수집이 완료되었습니다.\n저장 위치: {saved_file}")
        else:
            QMessageBox.warning(self, "경고", "수집 중인 세션이 없습니다.")
    
    def view_collected_data(self):
        """✨ 수집된 데이터 보기"""
        summary = self.dqn_collector.get_collected_data_summary()
        
        info_text = f"""=== 수집된 학습 데이터 요약 ===

총 세션 수: {summary.get('total_sessions', 0)}개
총 샘플 수: {summary.get('total_samples', 0)}개
세션당 평균 샘플: {summary.get('average_samples_per_session', 0):.1f}개

데이터 저장 위치:
{summary.get('data_directory', 'N/A')}

✨ 포인트 편집시마다 학습 데이터가 자동 수집됩니다.
✨ 충분한 데이터가 모이면 모델 학습을 시작하세요."""
        
        QMessageBox.information(self, "수집된 데이터", info_text)
    
    def start_training(self):
        """✨ 모델 학습 시작"""
        # 간단한 학습 실행
        progress = QProgressDialog("모델 학습 중...", "취소", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        try:
            # 학습 실행
            results = self.dqn_trainer.train()
            
            if results['success']:
                self.model_status_label.setText("모델: 학습완료")
                QMessageBox.information(
                    self, "학습 완료",
                    f"모델 학습이 완료되었습니다!\n\n"
                    f"에포크: {results['epochs']}\n"
                    f"정확도: {results['accuracy']:.2%}\n"
                    f"손실: {results['loss']:.4f}"
                )
            else:
                QMessageBox.warning(self, "경고", "학습할 데이터가 부족합니다.")
                
        except Exception as e:
            QMessageBox.critical(self, "오류", f"학습 중 오류 발생:\n{str(e)}")
        finally:
            progress.close()
    
    def run_ai_prediction(self):
        """✨ AI 예측 실행"""
        if not hasattr(self.canvas_widget, 'skeleton') or not self.canvas_widget.skeleton:
            QMessageBox.warning(self, "경고", "예측할 데이터가 없습니다.")
            return
        
        # 예측기 로드
        if not self.dqn_predictor.load_model():
            QMessageBox.warning(self, "경고", "학습된 모델이 없습니다. 먼저 학습을 수행하세요.")
            return
        
        try:
            # AI 예측 실행
            detected_points = self.canvas_widget.canvas.points
            skeleton_data = self.canvas_widget.skeleton
            
            recommendations = self.dqn_predictor.get_removal_candidates(
                detected_points, skeleton_data, confidence_threshold=0.7
            )
            
            if recommendations:
                # AI 추천 표시
                self.canvas_widget.canvas.ai_points = recommendations
                self.canvas_widget.canvas.draw_ai_predictions()
                
                total_suggestions = sum(len(points) for points in recommendations.values())
                QMessageBox.information(
                    self, "AI 예측 완료",
                    f"AI가 {total_suggestions}개의 포인트 수정을 추천했습니다.\n\n"
                    f"캔버스에서 확인하고 적용하세요."
                )
            else:
                QMessageBox.information(self, "AI 예측", "추천할 수정 사항이 없습니다.")
                
        except Exception as e:
            QMessageBox.critical(self, "오류", f"AI 예측 중 오류:\n{str(e)}")


def main():
    """메인 실행 함수"""
    import sys
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    
    # High DPI 지원
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 메인 윈도우 생성 및 표시
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
