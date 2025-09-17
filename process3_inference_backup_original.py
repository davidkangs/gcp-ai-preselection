"""프로세스 3: AI 예측 + 인간 수정 + 재학습 (지구계 지원) - 리팩토링된 버전"""

import sys
import os
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog, QMessageBox,
    QGroupBox, QCheckBox, QProgressDialog, QTableWidget, QTableWidgetItem,
    QTextEdit, QSplitter, QComboBox, QSpinBox, QRadioButton, QButtonGroup,
    QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont

# 프로젝트 경로 설정
sys.path.append(str(Path(__file__).parent))
from src.ui.canvas_widget import CanvasWidget
from src.process3 import PipelineManager, DataProcessor, FilterManager, AIPredictor, SessionManager

# 필요한 추가 import들
import tempfile
import torch
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from src.core.skeleton_extractor import SkeletonExtractor
from src.core.district_road_clipper import DistrictRoadClipper
from src.learning.dqn_model import create_agent
from src.utils import save_session, load_session, get_polygon_session_name
from src.filters.hybrid_filter import create_hybrid_filter

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionWorker(QThread):
    """리팩토링된 예측 워커 - PipelineManager 사용"""
    progress = pyqtSignal(int, str)
    prediction_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, file_path, model_path, file_mode='road'):
        super().__init__()
        self.file_path = file_path
        self.model_path = model_path
        self.file_mode = file_mode
        self.temp_path = None
        
        # 파이프라인 매니저 초기화
        self.pipeline_manager = PipelineManager(model_path=model_path)
        
        # 진행률 콜백 설정
        self.pipeline_manager.set_progress_callback(self._emit_progress)

    def _emit_progress(self, progress: int, message: str):
        """진행률 전송"""
        self.progress.emit(progress, message)

    def run(self):
        try:
            # 파일 경로 결정
            target_file = self.temp_path if self.temp_path else self.file_path
            
            # 파이프라인 실행
            if self.file_mode == 'district':
                result = self.pipeline_manager.run_district_pipeline(
                    target_file, 
                    enable_ai=True, 
                    save_session=False
                )
            else:
                result = self.pipeline_manager.run_road_pipeline(
                    target_file, 
                    enable_ai=True, 
                    save_session=False
                )
            
            if result['success']:
                # 기존 포맷으로 변환
                if self.file_mode == 'district' and 'polygon_results' in result:
                    # 지구계 모드: 첫 번째 폴리곤 결과 사용
                    if result['polygon_results']:
                        first_result = result['polygon_results'][0]
                        converted_result = {
                            'success': True,
                            'skeleton': first_result['skeleton'],
                            'ai_points': first_result.get('ai_result', {}).get('ai_points', {}),
                            'predictions': [],
                            'confidence_data': first_result.get('ai_result', {}).get('confidence_data', [])
                        }
                    else:
                        converted_result = {'success': False, 'error': '폴리곤 처리 결과가 없습니다'}
                else:
                    # 도로망 모드
                    converted_result = {
                        'success': True,
                        'skeleton': result['skeleton'],
                        'ai_points': result.get('ai_result', {}).get('ai_points', {}),
                        'predictions': [],
                        'confidence_data': result.get('ai_result', {}).get('confidence_data', [])
                    }
                
                self.prediction_completed.emit(converted_result)
            else:
                self.error_occurred.emit(result.get('error', '알 수 없는 오류'))
                
        except Exception as e:
            logger.error(f"예측 워커 오류: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))


class InferenceTool(QMainWindow):
    """리팩토링된 추론 도구 - 모듈화된 파이프라인 사용"""
    
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.model_path = None
        self.original_predictions = None
        self.modified_sessions = []
        
        # 지구계 관련 추가
        self.file_mode = 'road'  # 'district' or 'road'
        self.selected_crs = 'EPSG:5186'
        self.current_polygon_data = None
        self.current_polygon_index = 0
        self.ai_confidence_threshold = 0.7  # 고정값 (슬라이더 제거됨)
        
        # Excel 기준점 추가
        self.excel_points = []
        

        
        # 모듈화된 관리자들 초기화
        self.pipeline_manager = None
        self.data_processor = DataProcessor()
        self.session_manager = SessionManager()
        self.filter_manager = FilterManager(
            dbscan_eps=10.0,      # 10m 클러스터링 (더 엄격)
            network_max_dist=30.0, # 30m 네트워크 연결 (더 엄격)
            road_buffer=2.0       # 2m 도로 버퍼
        )
        
        self.init_ui()
        self.check_models()
        
        # 자동으로 수동 편집 모드 활성화
        self.enable_manual_edit_mode()

    def init_ui(self):
        self.setWindowTitle("도로망 AI 예측 및 수정 - 프로세스 3 (지구계 지원)")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        splitter = QSplitter(Qt.Horizontal)  # type: ignore
        main_layout.addWidget(splitter)
        
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        self.canvas_widget = CanvasWidget()
        self.canvas_widget.canvas.show_ai_predictions = True
        
        # Canvas 신호 연결 (점 변경 시 자동 업데이트)
        self.canvas_widget.canvas.point_added.connect(self.on_point_changed)
        self.canvas_widget.canvas.point_removed.connect(self.on_point_changed)
        
        splitter.addWidget(self.canvas_widget)
        
        splitter.setSizes([450, 950])  # 왼쪽 패널을 조금 더 넓게
        self.statusBar().showMessage("준비")

    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # ===== 1. 파일 모드 선택 (프로세스 1과 동일) =====
        mode_group = QGroupBox("파일 모드")
        mode_layout = QVBoxLayout()
        
        self.mode_button_group = QButtonGroup()  # 버튼 그룹 추가
        
        self.district_radio = QRadioButton("지구계 파일 (자동 도로망 추출)")
        self.district_radio.toggled.connect(lambda checked: self.set_file_mode('district' if checked else 'road'))
        self.mode_button_group.addButton(self.district_radio)
        mode_layout.addWidget(self.district_radio)
        
        self.road_radio = QRadioButton("도로망 파일 (직접 선택)")
        self.road_radio.setChecked(True)
        self.mode_button_group.addButton(self.road_radio)
        mode_layout.addWidget(self.road_radio)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # ===== 2. 좌표계 선택 (지구계 모드에서만 표시) =====
        self.crs_group = QGroupBox("좌표계 선택")
        crs_layout = QVBoxLayout()
        
        self.crs_button_group = QButtonGroup()  # 버튼 그룹 추가
        
        self.crs_5186_radio = QRadioButton("EPSG:5186 (중부원점)")
        self.crs_5186_radio.setChecked(True)
        self.crs_button_group.addButton(self.crs_5186_radio)
        crs_layout.addWidget(self.crs_5186_radio)
        
        self.crs_5187_radio = QRadioButton("EPSG:5187 (동부원점)")
        self.crs_button_group.addButton(self.crs_5187_radio)
        crs_layout.addWidget(self.crs_5187_radio)
        
        self.crs_group.setLayout(crs_layout)
        self.crs_group.setVisible(False)  # 초기에는 숨김
        layout.addWidget(self.crs_group)
        
        # ===== 3. AI 모델 선택 =====
        model_group = QGroupBox("1. AI 모델 선택")
        model_layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        
        refresh_model_btn = QPushButton("모델 목록 새로고침")
        refresh_model_btn.clicked.connect(self.check_models)
        model_layout.addWidget(refresh_model_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # ===== 4. 파일 처리 =====
        file_group = QGroupBox("2. 파일 처리")
        file_layout = QVBoxLayout()
        
        select_file_btn = QPushButton("파일 선택")
        select_file_btn.clicked.connect(self.select_file)
        select_file_btn.setStyleSheet("QPushButton {font-weight: bold; padding: 6px;}")
        file_layout.addWidget(select_file_btn)
        
        self.file_label = QLabel("파일: 선택 안됨")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("QLabel {background-color: #f0f0f0; padding: 5px; border-radius: 3px;}")
        file_layout.addWidget(self.file_label)
        
        # 멀티폴리곤 네비게이션 (지구계 모드에서만 표시)
        self.polygon_nav_widget = QWidget()
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(0, 5, 0, 5)
        
        self.polygon_info_label = QLabel("")
        self.polygon_info_label.setStyleSheet("QLabel {font-weight: bold;}")
        nav_layout.addWidget(self.polygon_info_label)
        
        nav_layout.addStretch()
        
        self.prev_polygon_btn = QPushButton("◀ 이전")
        self.prev_polygon_btn.clicked.connect(self.prev_polygon)
        self.prev_polygon_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_polygon_btn)
        
        self.next_polygon_btn = QPushButton("다음 ▶")
        self.next_polygon_btn.clicked.connect(self.next_polygon)
        self.next_polygon_btn.setEnabled(False)
        nav_layout.addWidget(self.next_polygon_btn)
        
        self.polygon_nav_widget.setLayout(nav_layout)
        self.polygon_nav_widget.setVisible(False)
        file_layout.addWidget(self.polygon_nav_widget)
        
        # AI 분석 버튼 (지구계/도로망 모드 통합)
        self.ai_analyze_btn = QPushButton("🤖 AI 분석 실행")
        self.ai_analyze_btn.clicked.connect(self.run_ai_analysis)
        self.ai_analyze_btn.setStyleSheet("QPushButton {background-color: #2196F3; color: white; font-weight: bold; padding: 10px; font-size: 14px;}")
        file_layout.addWidget(self.ai_analyze_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # ===== 5. 예측 결과 =====
        result_group = QGroupBox("3. 예측 결과")
        result_layout = QVBoxLayout()
        
        # 신뢰도는 고정값 0.7 사용 (슬라이더 제거됨)
        
        self.result_label = QLabel("예측 전...")
        self.result_label.setStyleSheet("QLabel {padding: 10px; background-color: #f0f0f0; border-radius: 5px;}")
        result_layout.addWidget(self.result_label)
        
        # 거리 정보 표시 (간단화)
        self.distance_label = QLabel("점간 거리: AI 분석 후 자동 표시")
        self.distance_label.setStyleSheet("QLabel {padding: 5px; background-color: #e8f4fd; border-radius: 3px; font-size: 11px;}")
        result_layout.addWidget(self.distance_label)
        
        # Excel 업로드 버튼 추가
        excel_btn = QPushButton("📊 실제 기준점 Excel 업로드")
        excel_btn.clicked.connect(self.upload_excel)
        excel_btn.setStyleSheet("QPushButton {background-color: #009688; color: white; font-weight: bold; padding: 8px;}")
        result_layout.addWidget(excel_btn)
        
        # 점 개수 비교 테이블 (컴팩트 사이즈)
        self.point_count_table = QTableWidget()
        self.point_count_table.setColumnCount(2)
        self.point_count_table.setHorizontalHeaderLabels(["구분", "개수"])
        self.point_count_table.setRowCount(3)
        self.point_count_table.setItem(0, 0, QTableWidgetItem("AI 검출점"))
        self.point_count_table.setItem(1, 0, QTableWidgetItem("실제 기준점"))
        self.point_count_table.setItem(2, 0, QTableWidgetItem("차이"))
        self.point_count_table.setItem(0, 1, QTableWidgetItem("0"))
        self.point_count_table.setItem(1, 1, QTableWidgetItem("0"))
        self.point_count_table.setItem(2, 1, QTableWidgetItem("0"))
        self.point_count_table.setMinimumHeight(120)  # 높이 축소
        self.point_count_table.setMaximumHeight(140)  # 최대 높이도 축소
        self.point_count_table.horizontalHeader().setStretchLastSection(True)
        # 폰트 크기 축소
        font = QFont()
        font.setPointSize(9)   # 폰트 크기 축소 (14 → 9)
        font.setBold(False)    # 굵게 해제
        self.point_count_table.setFont(font)
        # 행 높이 축소
        for i in range(3):
            self.point_count_table.setRowHeight(i, 28)  # 행 높이 축소 (50 → 28)
        result_layout.addWidget(self.point_count_table)
        
        # 체크박스들을 위한 레이아웃
        checkbox_layout = QVBoxLayout()
        
        self.show_ai_checkbox = QCheckBox("AI 예측 표시")
        self.show_ai_checkbox.setChecked(True)
        self.show_ai_checkbox.toggled.connect(self.toggle_ai_predictions)
        checkbox_layout.addWidget(self.show_ai_checkbox)
        
        self.show_excel_checkbox = QCheckBox("실제 기준점 표시")
        self.show_excel_checkbox.setChecked(True)
        self.show_excel_checkbox.toggled.connect(self.toggle_excel_points)
        checkbox_layout.addWidget(self.show_excel_checkbox)
        
        result_layout.addLayout(checkbox_layout)
        
        # 버튼들을 2열로 배치
        button_grid_layout = QVBoxLayout()
        
        row1_layout = QHBoxLayout()
        accept_all_btn = QPushButton("모든 AI 예측 수락")
        accept_all_btn.clicked.connect(self.accept_all_predictions)
        row1_layout.addWidget(accept_all_btn)
        
        clear_user_btn = QPushButton("사용자 수정 초기화")
        clear_user_btn.clicked.connect(self.clear_user_modifications)
        row1_layout.addWidget(clear_user_btn)
        
        # 거리 재계산 버튼은 자동으로 처리되므로 제거됨
        
        button_grid_layout.addLayout(row1_layout)
        
        # 수동 삭제 기능들 제거됨 - 자동 처리로 통합
        
        result_layout.addLayout(button_grid_layout)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # ===== 4. 결과 내보내기 =====
        export_group = QGroupBox("4. 결과 내보내기")
        export_layout = QVBoxLayout()
        
        # SHP 내보내기 버튼
        export_shp_btn = QPushButton("💾 점 SHP 파일로 저장")
        export_shp_btn.clicked.connect(self.export_points_to_shp)
        export_shp_btn.setStyleSheet("QPushButton {background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;}")
        export_layout.addWidget(export_shp_btn)
        
        # GPKG 내보내기 버튼
        export_gpkg_btn = QPushButton("🗂️ 전체 결과 GPKG로 저장")
        export_gpkg_btn.clicked.connect(self.export_all_to_gpkg)
        export_gpkg_btn.setStyleSheet("QPushButton {background-color: #FF9800; color: white; font-weight: bold; padding: 8px;}")
        export_layout.addWidget(export_gpkg_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # ===== 5. 배치 처리 =====
        batch_btn = QPushButton("📁 폴더 배치 처리")
        batch_btn.clicked.connect(self.start_batch_processing)
        batch_btn.setStyleSheet("QPushButton {background-color: #9C27B0; color: white; font-weight: bold; padding: 10px;}")
        layout.addWidget(batch_btn)
        
        layout.addStretch()
        return panel

    def set_file_mode(self, mode):
        """파일 모드 설정"""
        self.file_mode = mode
        self.crs_group.setVisible(mode == 'district')
        self.polygon_nav_widget.setVisible(mode == 'district' and self.current_polygon_data is not None)
        
        if mode == 'district':
            self.statusBar().showMessage("지구계 모드 - 도로망 자동 클리핑")
        else:
            self.statusBar().showMessage("도로망 모드 - 직접 파일 선택")
    
    def get_target_crs(self):
        """선택된 좌표계 반환"""
        if self.crs_5187_radio.isChecked():
            return 'EPSG:5187'
        return 'EPSG:5186'

    def check_models(self):
        self.model_combo.clear()
        models_dir = Path("models")
        
        if not models_dir.exists():
            return
        
        # true_dqn_model.pth를 우선적으로 찾기
        priority_model = models_dir / "true_dqn_model.pth"
        if priority_model.exists():
            self.model_combo.addItem("⭐ " + priority_model.name, str(priority_model))
        
        # 다른 모델들 추가
        for model_file in models_dir.glob("*.pth"):
            if model_file != priority_model:
                self.model_combo.addItem(model_file.name, str(model_file))
        
        if self.model_combo.count() > 0:
            self.model_path = self.model_combo.currentData()
        else:
            QMessageBox.warning(self, "경고", "학습된 모델이 없습니다.\n먼저 프로세스 2에서 모델을 학습하세요.")

    def select_file(self):
        """파일 선택"""
        if self.file_mode == 'district':
            file_path, _ = QFileDialog.getOpenFileName(
                self, "지구계 Shapefile 선택", "", "Shapefiles (*.shp)"
            )
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "도로망 Shapefile 선택", "", "Shapefiles (*.shp)"
            )
        
        if file_path:
            self.current_file = file_path
            self.file_label.setText(f"파일: {Path(file_path).name}")
            self.canvas_widget.clear_all()
            
            # 멀티폴리곤 데이터 초기화
            self.current_polygon_data = None
            self.current_polygon_index = 0
            self.polygon_nav_widget.setVisible(False)
            
            if self.file_mode == 'road':
                # 도로망 모드에서는 바로 처리
                self.process_road_file(file_path)

    def process_file(self):
        """처리 시작 버튼 클릭 (지구계 모드)"""
        if not self.current_file:
            QMessageBox.warning(self, "경고", "파일을 먼저 선택하세요.")
            return
        
        if self.file_mode == 'district':
            self.process_district_file(self.current_file)

    def process_road_file(self, file_path):
        """도로망 파일 자동화 처리 (리팩토링된 버전)"""
        try:
            progress = QProgressDialog("🤖 AI가 도로 특징점 분석 중...", "취소", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)  # type: ignore
            progress.show()
            
            # 파이프라인 매니저 초기화
            if not self.pipeline_manager:
                self.pipeline_manager = PipelineManager(model_path=self.model_path)
            
            # 진행률 콜백 설정
            def update_progress(value, message):
                progress.setValue(value)
                progress.setLabelText(message)
            
            self.pipeline_manager.set_progress_callback(update_progress)
            
            # 파이프라인 실행
            result = self.pipeline_manager.run_road_pipeline(
                file_path, 
                enable_ai=True, 
                save_session=False  # UI에서 별도 저장
            )
            
            if result['success']:
                # 결과를 캔버스에 적용
                skeleton = result['skeleton']
                points = result['points']
                road_gdf = result['road_gdf']
                
                # 캔버스 설정
                self.canvas_widget.set_road_data(road_gdf)
                
                # 스켈레톤 설정 (안전하게)
                if hasattr(self.canvas_widget, 'skeleton'):
                    self.canvas_widget.skeleton = skeleton
                if hasattr(self.canvas_widget.canvas, 'skeleton'):
                    self.canvas_widget.canvas.skeleton = skeleton
                
                # 포인트 설정
                self.canvas_widget.canvas.points = {
                    'intersection': points.get('intersection', []),
                    'curve': points.get('curve', []),
                    'endpoint': points.get('endpoint', [])
                }
                
                # AI 결과 설정
                if result.get('ai_result'):
                    self.canvas_widget.canvas.ai_points = result['ai_result'].get('ai_points', {})
                
                # 색상 설정
                self.canvas_widget.canvas.colors = {
                    'road': QColor(200, 200, 200),
                    'road_stroke': QColor(150, 150, 150),
                    'skeleton': QColor(50, 50, 200),
                    'background': QColor(230, 230, 230),
                    'background_stroke': QColor(200, 200, 200),
                    'intersection': QColor(100, 100, 255),
                    'curve': QColor(100, 100, 255),
                    'endpoint': QColor(100, 100, 255),
                    'excel': QColor(255, 100, 100),
                    'ai_intersection': QColor(255, 140, 0, 150),
                    'ai_curve': QColor(255, 140, 0, 150),
                    'ai_endpoint': QColor(255, 140, 0, 150)
                }
                
                # 점 크기 설정
                self.canvas_widget.canvas.point_size = 8
                
                # 점 개수 테이블 업데이트
                self.update_point_count_table()
                
                # 화면 업데이트
                self.canvas_widget.update_display()
                
                # 통계 표시
                total_points = sum(len(points_list) for points_list in points.values())
                self.statusBar().showMessage(f"자동화 파이프라인 완료 - 총 {total_points}개 점 검출")
                
                # 결과 라벨 업데이트
                if result.get('stats'):
                    stats = result['stats']
                    self.result_label.setText(
                        f"파이프라인 완료:\n"
                        f"스켈레톤: {stats.get('total_skeleton_points', 0)}점\n"
                        f"교차점: {stats.get('detected_intersections', 0)}개\n"
                        f"커브: {stats.get('detected_curves', 0)}개\n"
                        f"끝점: {stats.get('detected_endpoints', 0)}개"
                    )
                
                # 자동 거리 계산 실행
                self.auto_calculate_distances()
            else:
                QMessageBox.critical(self, "오류", f"파이프라인 실패:\n{result.get('error', '알 수 없는 오류')}")
            
            progress.setValue(100)
            progress.close()
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self, "오류", f"자동화 파이프라인 실패:\n{str(e)}")
            logger.error(f"자동화 파이프라인 오류: {e}")

    def process_district_file(self, district_file):
        """지구계 파일 처리 (리팩토링된 버전)"""
        try:
            progress = QProgressDialog("지구계 파일 처리 중...", "취소", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)  # type: ignore
            progress.show()
            
            progress.setLabelText("지구계 파일 읽는 중...")
            progress.setValue(10)
            
            # 파이프라인 매니저를 통한 지구계 처리
            if not self.pipeline_manager:
                self.pipeline_manager = PipelineManager(model_path=self.model_path)
            
            # 진행률 콜백 설정
            def update_progress(value, message):
                progress.setValue(value)
                progress.setLabelText(message)
            
            self.pipeline_manager.set_progress_callback(update_progress)
            
            # 지구계 파이프라인 실행
            results = self.pipeline_manager.run_district_pipeline(
                district_file,
                target_crs=self.get_target_crs(),
                enable_ai=True,
                save_session=False
            )
            
            if not results['success']:
                progress.close()
                
                if results['error'] == "도로망 파일을 찾을 수 없음":
                    QMessageBox.information(
                        self, "도로망 찾기 실패",
                        "파일명에서 행정구역을 찾을 수 없습니다.\n"
                        "도로망 폴더를 직접 선택해주세요."
                    )
                    
                    folder = QFileDialog.getExistingDirectory(
                        self, "도로망 폴더 선택",
                        "./road_by_sigungu"
                    )
                    
                    if folder:
                        self.process_with_manual_road(results['polygons'], folder)
                else:
                    QMessageBox.critical(self, "오류", results['error'])
                return
            
            progress.setValue(50)
            progress.setLabelText("클리핑 결과 처리 중...")
            
            # 멀티폴리곤 데이터 저장
            self.current_polygon_data = results
            self.current_polygon_index = 0
            
            # 첫 번째 폴리곤 로드
            self.load_polygon_result(results)
            
            # 멀티폴리곤 네비게이션 설정
            if results['total_polygons'] > 1:
                self.polygon_nav_widget.setVisible(True)
                self.update_polygon_navigation()
            
            progress.setValue(100)
            progress.close()
            
            info_text = f"지구계: {Path(district_file).name}\n"
            if 'sido' in results:
                info_text += f"지역: {results['sido']}"
                if 'sigungu' in results:
                    info_text += f" {results['sigungu']}"
            info_text += f"\n폴리곤 수: {results['total_polygons']}개"
            
            self.statusBar().showMessage(info_text)
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self, "오류", f"지구계 파일 처리 실패:\n{str(e)}")
            logger.error(f"지구계 처리 오류: {e}")

    def process_with_manual_road(self, polygons, road_folder):
        """수동 선택한 도로망으로 처리"""
        try:
            if not polygons:
                return
            
            # 첫 번째 폴리곤에 대해 클리핑
            first_polygon = polygons[0]
            clipped = self.district_clipper.clip_with_manual_road(
                first_polygon['geometry'],
                road_folder,
                self.get_target_crs()
            )
            
            if clipped is not None and not clipped.empty:
                # 클리핑된 도로망 저장
                first_polygon['clipped_road'] = clipped
                
                # 전체 결과 구성
                self.current_polygon_data = {
                    'success': True,
                    'polygons': polygons,
                    'total_polygons': len(polygons),
                    'target_crs': self.get_target_crs()
                }
                
                # 첫 번째 폴리곤 로드
                self.load_polygon_result(self.current_polygon_data)
                
                # 멀티폴리곤 네비게이션 설정
                if len(polygons) > 1:
                    self.polygon_nav_widget.setVisible(True)
                    self.update_polygon_navigation()
            else:
                QMessageBox.warning(self, "경고", "도로망 클리핑 결과가 없습니다.")
                
        except Exception as e:
            QMessageBox.critical(self, "오류", f"수동 도로망 처리 실패:\n{str(e)}")

    def load_polygon_result(self, result):
        """폴리곤 처리 결과 자동화 로드 (휴리스틱 → AI → 필터링 → 거리)"""
        if not result['success']:
            return
            
        # 새로운 파이프라인 매니저 결과 구조 처리
        if 'polygon_results' in result:
            # 파이프라인 매니저에서 온 결과 - 이미 처리된 데이터
            if not result['polygon_results']:
                return
            
            # 첫 번째 처리된 결과 사용
            polygon_result = result['polygon_results'][0]
            
            # 캔버스에 결과 적용
            if 'skeleton' in polygon_result:
                skeleton = polygon_result['skeleton']
                points = polygon_result['points']
                road_gdf = polygon_result.get('road_gdf')
                
                if road_gdf is not None:
                    self.canvas_widget.set_road_data(road_gdf)
                
                # 스켈레톤 설정
                self.canvas_widget.skeleton = skeleton
                self.canvas_widget.canvas.skeleton = skeleton
                
                # 포인트 설정
                self.canvas_widget.canvas.points = {
                    'intersection': points.get('intersection', []),
                    'curve': points.get('curve', []),
                    'endpoint': points.get('endpoint', [])
                }
                
                # AI 결과 설정 (있는 경우)
                if 'ai_result' in polygon_result and polygon_result['ai_result']:
                    self.canvas_widget.canvas.ai_points = polygon_result['ai_result'].get('ai_points', {})
                
                # 화면 업데이트
                self.canvas_widget.update_display()
                self.update_point_count_table()
                
                # 폴리곤 정보 표시
                if 'polygon_info' in polygon_result:
                    polygon_info = polygon_result['polygon_info']
                    if 'geometry' in polygon_info:
                        self.canvas_widget.set_background_data(polygon_info['geometry'])
                
                # 다단계 자동 점 최적화 실행
                total_removed, final_points = self.run_multi_stage_optimization()
                
                info_text = f"스켈레톤: {len(skeleton)}점\n"
                info_text += f"최적화된 점: {final_points}개\n"
                info_text += f"다단계 최적화 완료 ({total_removed}개 제거)"
                self.result_label.setText(f"처리 완료:\n{info_text}")
                self.statusBar().showMessage(f"지구계 다단계 최적화 완료 - 총 {final_points}개 점 유지 ({total_removed}개 제거)")
                
                self.result_label.setText(f"처리 완료:\n{info_text}")
            
            return
        
        # 기존 구조 처리 (polygons 키 사용)
        if not result.get('polygons'):
            return
            
        current_polygon = result['polygons'][self.current_polygon_index]
        
        # 지구계 경계 표시
        if 'geometry' in current_polygon:
            self.canvas_widget.set_background_data(current_polygon['geometry'])
        
        # 도로망이 있으면 자동화 파이프라인 실행
        if 'clipped_road' in current_polygon and current_polygon['clipped_road'] is not None:
            road_gdf = current_polygon['clipped_road']
            self.canvas_widget.set_road_data(road_gdf)
            
            try:
                # 자동화 파이프라인 실행
                progress = QProgressDialog("🌍 AI가 지구계 도로 분석 중...", "취소", 0, 100, self)
                progress.setWindowModality(Qt.WindowModal)  # type: ignore
                progress.show()
                
                # 1단계: 임시 파일로 스켈레톤 추출
                progress.setLabelText("🔍 1/5 단계: AI 도로 구조 분석 중...")
                progress.setValue(10)
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = os.path.join(temp_dir, "temp_road.shp")
                    road_gdf.to_file(temp_path)
                    
                    skeleton_extractor = SkeletonExtractor()
                    skeleton, intersections = skeleton_extractor.process_shapefile(temp_path)
                
                    # 2단계: 기본 캔버스 설정
                    progress.setLabelText("🎯 2/5 단계: AI 특징점 기본 분석 중...")
                    progress.setValue(20)
                    
                    self.canvas_widget.skeleton = skeleton  # type: ignore
                    self.canvas_widget.canvas.skeleton = skeleton
                    self.canvas_widget.canvas.points = {
                        'intersection': [(float(x), float(y)) for x, y in intersections],
                        'curve': [],
                        'endpoint': []
                    }
                        
                # 2-2단계: AI 기반 끝점 검출
                    endpoints = self.detect_heuristic_endpoints(skeleton, None)
                    self.canvas_widget.canvas.points['endpoint'] = [(float(x), float(y)) for x, y in endpoints]
                    
                    # 2-3단계: 🔄 도로 경계선 기반 커브 검출 (최적 설정값)
                    boundary_curves = self.detect_boundary_based_curves(
                        skeleton, 
                        sample_distance=15.0,      # 샘플링 거리
                        curvature_threshold=0.20,  # 곡률 임계값
                        road_buffer=3.0,           # 도로 버퍼
                        cluster_radius=20.0        # 군집 반경
                    )
                    
                    # 교차점 근처 커브점 제거 (10m 이내)
                    curves = self.remove_curves_near_intersections(
                        boundary_curves, intersections, threshold=10.0
                    )
                    
                    self.canvas_widget.canvas.points['curve'] = [(float(x), float(y)) for x, y in curves]
                    logger.info(f"도로 경계선 기반 검출 완료 - 교차점: {len(intersections)}개, 커브: {len(curves)}개, 끝점: {len(endpoints)}개")
                    
                    # 3단계: AI 예측 (삭제만, 커브는 도로 경계선 기반 사용)
                    progress.setLabelText("🤖 3/5 단계: AI 스마트 예측 실행 중...")
                    progress.setValue(40)
                    
                    if self.model_path:
                        ai_result = self.run_ai_prediction_auto(skeleton, temp_path)
                        if ai_result and ai_result['success']:
                            # AI 스마트 최적화 (삭제만 담당)
                            deleted_points = ai_result['ai_points'].get('delete', [])
                            if deleted_points:
                                self.apply_deletions(deleted_points)
                                logger.info(f"AI 품질 개선: {len(deleted_points)}개 점 최적화")
                    
                    # 4단계: 중복 점 필터링
                    progress.setLabelText("4/6 단계: 중복 점 필터링 중...")
                    progress.setValue(50)
                    
                    self.filter_overlapping_points()
                    
                    # 5단계: 최종 점 정리 (연결성 검사 제거됨)
                    progress.setLabelText("5/6 단계: 최종 점 정리 중...")
                    progress.setValue(65)
                    
                    logger.info("📍 3-액션 시스템: AI 분석(교차점,끝점,커브) + 스마트 삭제 완료")
                    
                    # 6단계: 거리 계산 및 표시
                    progress.setLabelText("6/6 단계: 점간 거리 계산 중...")
                    progress.setValue(80)
                    
                    self.calculate_and_display_distances()
                    
                    # 점 색깔 통일 (모든 점을 파란색으로)
                    self.canvas_widget.canvas.colors = {
                        'road': QColor(200, 200, 200),         # 기본 도로 색상
                        'road_stroke': QColor(150, 150, 150),  # 도로 테두리 색상
                        'skeleton': QColor(50, 50, 200),       # 스켈레톤 색상
                        'background': QColor(230, 230, 230),   # 배경 색상
                        'background_stroke': QColor(200, 200, 200),  # 배경 테두리
                        'intersection': QColor(100, 100, 255),  # 파란색
                        'curve': QColor(100, 100, 255),         # 파란색
                        'endpoint': QColor(100, 100, 255),      # 파란색
                        'excel': QColor(255, 100, 100),         # Excel 점은 다이아몬드로 표시
                        'ai_intersection': QColor(255, 140, 0, 150),  # AI 예측 색상
                        'ai_curve': QColor(255, 140, 0, 150),
                        'ai_endpoint': QColor(255, 140, 0, 150)
                    }
                    
                    # 점 크기도 통일 (2mm = 약 8픽셀)
                    self.canvas_widget.canvas.point_size = 8
                    
                    # 점 개수 테이블 업데이트
                    self.update_point_count_table()
                    
                    self.canvas_widget.update_display()
                    
                    progress.setValue(100)
                    progress.close()
                    
                    total_points = (len(self.canvas_widget.canvas.points['intersection']) +
                                  len(self.canvas_widget.canvas.points['curve']) +
                                  len(self.canvas_widget.canvas.points['endpoint']))
                    
                    # 다단계 자동 점 최적화 실행
                    total_removed, final_points = self.run_multi_stage_optimization()
                    
                    info_text = f"스켈레톤: {len(skeleton)}점\n"
                    info_text += f"최적화된 점: {final_points}개\n"
                    info_text += f"다단계 최적화 완료 ({total_removed}개 제거)"
                    
                    self.result_label.setText(f"처리 완료:\n{info_text}")
                    self.statusBar().showMessage(f"지구계 다단계 최적화 완료 - 총 {final_points}개 점 유지 ({total_removed}개 제거)")
                    
                    # 자동 거리 계산 실행 (지구계 모드)
                    self.auto_calculate_distances()
            except Exception as e:
                if 'progress' in locals():
                    progress.close()
                logger.error(f"지구계 자동화 파이프라인 오류: {e}")
                # 기본 모드로 폴백
                self.canvas_widget.skeleton = skeleton  # type: ignore
                self.canvas_widget.canvas.skeleton = skeleton
                self.canvas_widget.canvas.points = {
                    'intersection': [(float(x), float(y)) for x, y in intersections],
                    'curve': [],
                    'endpoint': []
                }
                self.canvas_widget.canvas.ai_points = {
                    'intersection': [], 'curve': [], 'endpoint': [], 'delete': []
                }
                self.canvas_widget.update_display()
                
                info_text = f"스켈레톤: {len(skeleton)}점\n"
                info_text += f"교차점: {len(intersections)}개"
                self.result_label.setText(f"처리 완료:\n{info_text}")

    def update_polygon_navigation(self):
        """멀티폴리곤 네비게이션 업데이트"""
        if not self.current_polygon_data:
            return
        
        total = self.current_polygon_data['total_polygons']
        current = self.current_polygon_index + 1
        
        self.polygon_info_label.setText(f"폴리곤 {current}/{total}")
        self.prev_polygon_btn.setEnabled(current > 1)
        self.next_polygon_btn.setEnabled(current < total)

    def prev_polygon(self):
        """이전 폴리곤으로 이동"""
        if self.current_polygon_index > 0:
            self.save_current_polygon_session()
            self.current_polygon_index -= 1
            self.load_polygon_at_index(self.current_polygon_index)
            self.update_polygon_navigation()

    def next_polygon(self):
        """다음 폴리곤으로 이동"""
        if self.current_polygon_data is None:
            return
        total = self.current_polygon_data['total_polygons']
        if self.current_polygon_index < total - 1:
            self.save_current_polygon_session()
            self.current_polygon_index += 1
            self.load_polygon_at_index(self.current_polygon_index)
            self.update_polygon_navigation()

    def load_polygon_at_index(self, index):
        """특정 인덱스의 폴리곤 로드"""
        if not self.current_polygon_data or not self.current_polygon_data['polygons']:
            return
        
        self.current_polygon_index = index
        self.load_polygon_result(self.current_polygon_data)

    def save_current_polygon_session(self):
        """현재 폴리곤 작업 저장"""
        if self.current_polygon_data and self.canvas_widget.skeleton and self.current_file:
            base_name = Path(self.current_file).stem
            session_name = get_polygon_session_name(
                base_name,
                self.current_polygon_index + 1,
                self.current_polygon_data['total_polygons']
            )
            
            # 세션 저장 로직
            labels = self.canvas_widget.canvas.points
            skeleton = self.canvas_widget.skeleton
            
            metadata = {
                'file_mode': self.file_mode,
                'polygon_index': self.current_polygon_index + 1,
                'total_polygons': self.current_polygon_data['total_polygons'],
                'target_crs': self.get_target_crs()
            }
            
            save_session(self.current_file, labels, skeleton, metadata)

    def run_multi_stage_optimization(self):
        """다단계 점 최적화 실행"""
        try:
            # 1단계: 지능형 클러스터링
            clustering_result = self.intelligent_clustering_optimization()
            stage1_removed = clustering_result[0] if clustering_result else 0
            
            # 2단계: 가까운 점 클러스터링 삭제
            stage2_removed = self.remove_clustered_points(15.0)
            
            # 3단계: 연결성 기반 커브점 삭제
            stage3_removed = 1 if self.remove_one_curve_point_by_connectivity() else 0
            
            # 4단계: 자동 끝점 정리
            stage4_removed = self.auto_remove_road_endpoints()
            
            # 5단계: 중복점 필터링
            self.filter_overlapping_points()
            
            total_removed = stage1_removed + stage2_removed + stage3_removed + stage4_removed
            final_points = sum(len(self.canvas_widget.canvas.points.get(cat, [])) for cat in ['intersection', 'curve', 'endpoint'])
            
            logger.info(f"🔧 다단계 점 최적화: 1단계({stage1_removed}) + 2단계({stage2_removed}) + 3단계({stage3_removed}) + 4단계({stage4_removed}) = 총 {total_removed}개 제거")
            return total_removed, final_points
            
        except Exception as e:
            logger.error(f"다단계 자동 최적화 실패: {e}")
            return 0, sum(len(self.canvas_widget.canvas.points.get(cat, [])) for cat in ['intersection', 'curve', 'endpoint'])
    
    def intelligent_clustering_optimization(self):
        """20m 반경 지능형 클러스터링 - 중요도 기반 점 선택"""
        if not self.canvas_widget.canvas.points:
            return
        
        all_points = []
        for category in ['intersection', 'curve', 'endpoint']:
            points = self.canvas_widget.canvas.points.get(category, [])
            for point in points:
                all_points.append({
                    'coord': tuple(point),
                    'category': category,
                    'importance': 0.0
                })
        
        if len(all_points) < 2:
            return
        
        skeleton = self.canvas_widget.skeleton
        skeleton_array = np.array(skeleton) if skeleton else np.array([])
        
        for point_data in all_points:
            point_data['importance'] = self.calculate_point_importance(
                point_data['coord'], point_data['category'], skeleton_array)
        
        try:
            from sklearn.cluster import DBSCAN
            
            coords = np.array([p['coord'] for p in all_points])
            clustering = DBSCAN(eps=20.0, min_samples=1).fit(coords)
            
            optimized_points = {'intersection': [], 'curve': [], 'endpoint': []}
            clusters = {}
            
            for i, label in enumerate(clustering.labels_):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(all_points[i])
            
            removed_count = 0
            kept_count = 0
            
            for cluster_points in clusters.values():
                if len(cluster_points) == 1:
                    point = cluster_points[0]
                    optimized_points[point['category']].append(point['coord'])
                    kept_count += 1
                else:
                    best_point = max(cluster_points, key=lambda p: p['importance'])
                    optimized_points[best_point['category']].append(best_point['coord'])
                    kept_count += 1
                    removed_count += len(cluster_points) - 1
            
            for category in ['intersection', 'curve', 'endpoint']:
                self.canvas_widget.canvas.points[category] = optimized_points[category]
            
            self.canvas_widget.canvas.update_display()
            self.update_modification_stats()
            
            return removed_count, kept_count
            
        except ImportError:
            logger.error("scikit-learn이 필요합니다. 기본 중복 제거로 대체")
            self.basic_duplicate_removal()
        except Exception as e:
            logger.error(f"지능형 클러스터링 오류: {e}")
            self.basic_duplicate_removal()
    
    def basic_duplicate_removal(self):
        """기본 중복점 제거 (폴백 방식)"""
        all_points = []
        point_roles = {}
        
        for category in ['intersection', 'curve', 'endpoint']:
            for point in self.canvas_widget.canvas.points.get(category, []):
                all_points.append(tuple(point))
                point_roles[tuple(point)] = category
        
        if len(all_points) < 2:
            return
        
        # 20m 반경 내 중복점 제거
        cleaned_points = {'intersection': [], 'curve': [], 'endpoint': []}
        processed = set()
        
        for point in all_points:
            if point in processed:
                continue
            
            # 20m 반경 내 다른 점들 찾기
            nearby_points = []
            for other_point in all_points:
                if other_point == point or other_point in processed:
                    continue
                    
                distance = np.hypot(point[0] - other_point[0], point[1] - other_point[1])
                if distance <= 20.0:
                    nearby_points.append(other_point)
            
            if not nearby_points:
                # 인근에 다른 점이 없으면 그대로 유지
                category = point_roles[point]
                cleaned_points[category].append(point)
                processed.add(point)
            else:
                # 인근 점들 중에서 우선순위가 높은 점 선택
                all_nearby = [point] + nearby_points
                priority_order = {'intersection': 3, 'endpoint': 2, 'curve': 1}
                
                best_point = max(all_nearby, key=lambda p: priority_order[point_roles[p]])
                category = point_roles[best_point]
                cleaned_points[category].append(best_point)
                
                # 모든 인근 점들을 처리됨으로 표시
                processed.add(point)
                for np_point in nearby_points:
                    processed.add(np_point)
        
        # 결과 적용
        for category in ['intersection', 'curve', 'endpoint']:
            self.canvas_widget.canvas.points[category] = cleaned_points[category]
        
        logger.info("기본 중복점 제거 완료")

    def remove_clustered_points(self, distance_threshold=15.0):
        """가까운 점들 중 하나씩 삭제"""
        deleted_count = 0
        
        all_points = []
        for category in ['intersection', 'curve', 'endpoint']:
            for i, point in enumerate(self.canvas_widget.canvas.points[category]):
                all_points.append({
                    'point': point,
                    'category': category,
                    'index': i
                })
        
        if len(all_points) < 2:
            return 0
        
        points_to_remove = []
        used_indices = set()
        
        for i, p1 in enumerate(all_points):
            if i in used_indices:
                continue
            
            nearby_points = []
            for j, p2 in enumerate(all_points):
                if i != j and j not in used_indices:
                    try:
                        dist = np.sqrt(
                            (float(p1['point'][0]) - float(p2['point'][0]))**2 + 
                            (float(p1['point'][1]) - float(p2['point'][1]))**2
                        )
                        if dist <= distance_threshold:
                            nearby_points.append((j, p2, dist))
                    except:
                        continue
            
            if nearby_points:
                nearby_points.sort(key=lambda x: x[2])
                to_remove_idx, to_remove_point, _ = nearby_points[0]
                points_to_remove.append(to_remove_point)
                used_indices.add(to_remove_idx)
                used_indices.add(i)
        
        for point_info in points_to_remove:
            category = point_info['category']
            point = point_info['point']
            
            if point in self.canvas_widget.canvas.points[category]:
                self.canvas_widget.canvas.points[category].remove(point)
                deleted_count += 1
        
        return deleted_count

    def auto_remove_road_endpoints(self):
        """자동 끝점 정리 - 경계 근처 고립된 끝점 제거"""
        removed_count = 0
        endpoints = self.canvas_widget.canvas.points.get('endpoint', [])
        
        if not endpoints:
            return 0
        
        # 스켈레톤 경계 계산
        skeleton = self.canvas_widget.skeleton
        if not skeleton or not isinstance(skeleton, (list, tuple)):
            return 0
        
        x_coords = [pt[0] for pt in skeleton if hasattr(pt, '__len__') and len(pt) >= 2]
        y_coords = [pt[1] for pt in skeleton if hasattr(pt, '__len__') and len(pt) >= 2]
        
        if not x_coords or not y_coords:
            return 0
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # 경계 근처 고립된 끝점 제거
        endpoints_to_remove = []
        for endpoint in endpoints:
            x, y = float(endpoint[0]), float(endpoint[1])
            
            # 경계와의 거리
            dist_to_boundary = min(x - min_x, max_x - x, y - min_y, max_y - y)
            
            # 경계 20m 이내이고 다른 점들과 50m 이상 떨어진 고립된 끝점 제거
            if dist_to_boundary < 20.0:
                isolated = True
                for category in ['intersection', 'curve']:
                    for other_point in self.canvas_widget.canvas.points.get(category, []):
                        dist = np.sqrt((x - other_point[0])**2 + (y - other_point[1])**2)
                        if dist < 50.0:
                            isolated = False
                            break
                    if not isolated:
                        break
                
                if isolated:
                    endpoints_to_remove.append(endpoint)
        
        # 제거 실행
        for endpoint in endpoints_to_remove:
            if endpoint in self.canvas_widget.canvas.points['endpoint']:
                self.canvas_widget.canvas.points['endpoint'].remove(endpoint)
                removed_count += 1
        
        return removed_count

    def remove_one_curve_point_by_connectivity(self):
        """연결성 검사로 직선상 커브점 1개 삭제"""
        try:
            all_points = []
            for category in ['intersection', 'curve', 'endpoint']:
                for point in self.canvas_widget.canvas.points[category]:
                    all_points.append({
                        'point': point,
                        'category': category,
                        'coords': (float(point[0]), float(point[1]))
                    })
            
            if len(all_points) < 3:
                return False
            
            road_union = self.get_road_union()
            if not road_union:
                return False
            
            deletable_curves = []
            
            for i, p1 in enumerate(all_points):
                for j, p2 in enumerate(all_points):
                    if i >= j:
                        continue
                    
                    try:
                        from shapely.geometry import LineString, Point
                        line = LineString([p1['coords'], p2['coords']])
                        
                        if line.length < 10:
                            continue
                        
                        if road_union.contains(line) or road_union.intersects(line):
                            for k, p3 in enumerate(all_points):
                                if k == i or k == j or p3['category'] != 'curve':
                                    continue
                                
                                point_dist_to_line = line.distance(Point(p3['coords']))
                                
                                if point_dist_to_line < 5.0:
                                    segment_length = np.sqrt(
                                        (p1['coords'][0] - p2['coords'][0])**2 + 
                                        (p1['coords'][1] - p2['coords'][1])**2
                                    )
                                    
                                    deletable_curves.append({
                                        'point_info': p3,
                                        'distance': segment_length,
                                    })
                    except Exception:
                        continue
            
            deletable_curves.sort(key=lambda x: x['distance'])
            
            if deletable_curves:
                to_delete = deletable_curves[0]
                point_info = to_delete['point_info']
                
                if point_info['point'] in self.canvas_widget.canvas.points['curve']:
                    self.canvas_widget.canvas.points['curve'].remove(point_info['point'])
                    self.canvas_widget.canvas.update_display()
                    return True
            
            return False
            
        except Exception:
            return False

    def calculate_point_importance(self, point_coord, category, skeleton_array):
        """점의 중요도 계산 - 스켈레톤 밀도와 카테고리 기반"""
        x, y = point_coord
        importance_score = 0.0
        
        # 1. 카테고리 기반 기본 중요도 (교차점 > 끝점 > 커브점)
        category_weights = {
            'intersection': 10.0,  # 교차점이 가장 중요
            'endpoint': 7.0,       # 끝점이 두 번째 중요
            'curve': 5.0           # 커브점이 세 번째 중요
        }
        importance_score += category_weights.get(category, 0.0)
        
        # 2. 스켈레톤 밀도 기반 중요도 (주변 스켈레톤 점의 개수)
        if len(skeleton_array) > 0:
            # 50m 반경 내 스켈레톤 점 개수
            distances = np.sqrt(np.sum((skeleton_array - np.array([x, y]))**2, axis=1))
            nearby_count_50m = np.sum(distances <= 50.0)
            
            # 30m 반경 내 스켈레톤 점 개수 (더 가중치 높음)
            nearby_count_30m = np.sum(distances <= 30.0)
            
            # 10m 반경 내 스켈레톤 점 개수 (가장 가중치 높음)
            nearby_count_10m = np.sum(distances <= 10.0)
            
            # 밀도 점수 계산
            density_score = (nearby_count_10m * 3.0 + 
                           nearby_count_30m * 2.0 + 
                           nearby_count_50m * 1.0)
            
            importance_score += density_score
        
        # 3. 스켈레톤 중심선과의 거리 (가까울수록 중요)
        if len(skeleton_array) > 0:
            min_distance = np.min(np.sqrt(np.sum((skeleton_array - np.array([x, y]))**2, axis=1)))
            
            # 거리가 가까울수록 높은 점수 (최대 5점)
            if min_distance < 5.0:
                distance_score = 5.0
            elif min_distance < 10.0:
                distance_score = 4.0
            elif min_distance < 15.0:
                distance_score = 3.0
            elif min_distance < 20.0:
                distance_score = 2.0
            else:
                distance_score = 1.0
                
            importance_score += distance_score
        
        # 4. 네트워크 연결성 보너스 (교차점의 경우)
        if category == 'intersection':
            # 교차점은 추가 보너스
            importance_score += 5.0
        
        return importance_score

    def filter_overlapping_points(self):
        """스켈레톤 기반 하이브리드 필터를 사용한 중복 점 필터링"""
        try:
            points = self.canvas_widget.canvas.points
            if not points:
                return
            all_points = []
            point_roles = {}
            for category in ['intersection', 'curve', 'endpoint']:
                for point in points[category]:
                    all_points.append(point)
                    point_roles[point] = category
            if len(all_points) < 2:
                return
            skeleton = self.canvas_widget.skeleton
            if not skeleton:
                logger.warning("스켈레톤 데이터가 없어 기본 필터링 사용")
                skeleton = []
            from shapely.geometry import LineString
            skeleton_line = LineString(skeleton) if skeleton and len(skeleton) > 1 else None
            skeleton_lines = [skeleton_line] if skeleton_line else []
            filtered_points = self.filter_manager.hybrid_filter.filter_by_skeleton_connectivity(
                points=all_points,
                skeleton_lines=skeleton_lines,
                point_roles=point_roles,
                dist_thresh=10.0,
                curve_min_length=20.0
            )
            filtered_by_role = {'intersection': [], 'curve': [], 'endpoint': []}
            for point in filtered_points:
                role = point_roles.get(point, 'curve')
                filtered_by_role[role].append(point)
            removed_count = sum(len(points[cat]) for cat in points) - sum(len(filtered_by_role[cat]) for cat in filtered_by_role)
            self.canvas_widget.canvas.points = filtered_by_role
            logger.info(f"스켈레톤 기반 하이브리드 중복 점 필터링 완료: {removed_count}개 점 제거")
        except Exception as e:
            logger.error(f"스켈레톤 기반 하이브리드 중복 점 필터링 오류: {e}")

    def run_ai_prediction_auto(self, skeleton, file_path):
        """자동화된 AI 예측 실행 (샘플링 제거, 전체 포인트 사용)"""
        try:
            if not self.model_path:
                return None
            
            # DQN 에이전트 생성 및 모델 로드
            agent = create_agent()
            agent.load(self.model_path)
            
            # 전체 스켈레톤 사용 (샘플링 제거)
            skeleton_array = np.array(skeleton)
            logger.info(f"전체 스켈레톤 사용: {len(skeleton_array)}개 점")
            
            # 특징 추출
            features = []
            for i, point in enumerate(skeleton_array):
                feat = self._create_dqn_state_vector(point, skeleton_array, i)
                features.append(feat)
            
            features_array = np.array(features)
            
            # AI 예측 (신뢰도 기반)
            ai_points = {
                'intersection': [],
                'curve': [],
                'endpoint': [],
                'delete': []
            }
            
            confidence_data = []
            
            if hasattr(agent, 'q_network'):
                with torch.no_grad():
                    device = next(agent.q_network.parameters()).device
                    input_tensor = torch.FloatTensor(features_array).to(device)
                    q_values_batch = agent.q_network(input_tensor)
                
                # 각 포인트에 대해 예측 및 신뢰도 계산
                for i, q_values in enumerate(q_values_batch):
                    q_vals = q_values.cpu().numpy()
                    action = np.argmax(q_vals)
                    
                    # 신뢰도 계산 (1등과 2등의 차이)
                    max_q = np.max(q_vals)
                    second_max_q = np.partition(q_vals, -2)[-2] if len(q_vals) > 1 else 0
                    confidence = max_q - second_max_q
                    
                    point = tuple(skeleton_array[i])
                    
                    # 신뢰도 정보 저장
                    confidence_data.append({
                        'point': point,
                        'action': action,
                        'confidence': confidence
                    })
                    
                    # 신뢰도 임계값 체크 - 삭제 전용 모드
                    if confidence >= self.ai_confidence_threshold:
                        if action == 4:  # 삭제만 수행
                            ai_points['delete'].append(point)
                        # 생성 액션들(1,2,3)은 무시
            
            # 긍정적 AI 결과 로그 (삭제 전용 모드)
            total_optimized = len(skeleton_array) - len(ai_points['delete'])
            logger.info(f"AI 스마트 최적화 완료: {total_optimized}개 최적 특징점 검출")
            if len(ai_points['delete']) > 0:
                logger.info(f"AI 품질 개선: {len(ai_points['delete'])}개 불필요한 점 제거 후보")
            
            return {
                'success': True,
                'ai_points': ai_points,
                'confidence_data': confidence_data
            }
            
        except Exception as e:
            logger.error(f"AI 예측 오류: {e}")
            return None

    def calculate_and_display_distances(self):
        """자동 거리 계산 및 표시 - 스마트 연결 로직 사용"""
        try:
            # 스마트 거리 연결 로직 실행
            self.calculate_smart_distance_connections()
            
        except Exception as e:
            logger.error(f"거리 계산 오류: {e}")
            self.distance_label.setText(f"거리 계산 오류: {str(e)}")

    def detect_heuristic_endpoints(self, skeleton, road_bounds=None):
        """휴리스틱 끝점 검출 - 지구계 경계 근처의 도로 끝"""
        if not skeleton:
            return []
        
        endpoints = []
        
        # 스켈레톤 포인트들의 경계 계산
        if skeleton:
            x_coords = [pt[0] for pt in skeleton if len(pt) >= 2]
            y_coords = [pt[1] for pt in skeleton if len(pt) >= 2]
            
            if x_coords and y_coords:
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                # 경계로부터의 거리 임계값 (30m)
                threshold = 30.0
                
                for i, point in enumerate(skeleton):
                    if len(point) < 2:
                        continue
                    
                    x, y = float(point[0]), float(point[1])
                    
                    # 경계와의 거리 계산
                    dist_to_boundary = min(
                        x - min_x, max_x - x,  # 좌우 경계
                        y - min_y, max_y - y   # 상하 경계
                    )
                    
                    # 경계 근처이고 연결된 점이 적으면 끝점
                    if dist_to_boundary < threshold:
                        # 주변 연결점 개수 확인
                        connected_count = 0
                        for other_point in skeleton:
                            if len(other_point) < 2:
                                continue
                            other_x, other_y = float(other_point[0]), float(other_point[1])
                            if (x, y) != (other_x, other_y):
                                dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                                if dist < 50:  # 50m 이내 연결점
                                    connected_count += 1
                        
                        # 연결점이 2개 이하면 끝점으로 판단
                        if connected_count <= 2:
                            endpoints.append((x, y))
        
        logger.info(f"🔚 휴리스틱 끝점 검출: {len(endpoints)}개")
        return endpoints
    
    def apply_deletions(self, delete_points):
        """AI가 예측한 삭제 포인트들을 적용"""
        if not delete_points:
            return
        
        deleted_count = 0
        for delete_x, delete_y in delete_points:
            # 각 카테고리에서 가장 가까운 점 찾아서 제거
            for category in ['intersection', 'curve', 'endpoint']:
                points = self.canvas_widget.canvas.points.get(category, [])
                
                min_dist = float('inf')
                closest_idx = -1
                
                for i, (x, y) in enumerate(points):
                    dist = np.sqrt((x - delete_x)**2 + (y - delete_y)**2)
                    if dist < min_dist and dist < 10:  # 10m 이내만
                        min_dist = dist
                        closest_idx = i
                
                if closest_idx >= 0:
                    del self.canvas_widget.canvas.points[category][closest_idx]
                deleted_count += 1
                break
        
        if deleted_count > 0:
            logger.info(f"AI 스마트 최적화: {deleted_count}개 점 품질 개선 완료")
    
    # 연결성 검사 함수 제거됨 - 3-액션 시스템에서는 불필요

    def run_prediction(self):
        if not self.model_path:
            QMessageBox.warning(self, "경고", "모델을 선택하세요.")
            return
        
        # 파일 모드에 따른 체크
        if self.file_mode == 'road':
            if not self.current_file:
                QMessageBox.warning(self, "경고", "파일을 선택하세요.")
                return
        else:  # district mode
            if not self.canvas_widget.skeleton:
                QMessageBox.warning(self, "경고", "먼저 지구계 파일을 처리하세요.")
                return
        
        progress = QProgressDialog("AI 예측 중...", "취소", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)  # type: ignore
        progress.show()
        
        # 지구계 모드에서는 임시 파일 사용
        if self.file_mode == 'district' and self.current_polygon_data:
            current_polygon = self.current_polygon_data['polygons'][self.current_polygon_index]

            if 'clipped_road' in current_polygon and current_polygon['clipped_road'] is not None:
                # 1) mkdtemp 로 디렉토리 생성 (반환된 경로가 사라지지 않습니다)
                tmp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(tmp_dir, "temp_road.shp")

                # 2) 폴리곤을 to_file 로 저장
                current_polygon['clipped_road'].to_file(temp_path)

                # 3) PredictionWorker 에 전달
                self.prediction_worker = PredictionWorker(temp_path, self.model_path, self.file_mode)
                self.prediction_worker.temp_path = temp_path  # type: ignore
                # (선택) 나중에 tmp_dir 를 지우기 위해 참조를 남겨둡니다
                self._tmp_dir = tmp_dir

            else:
                progress.close()
                QMessageBox.warning(self, "경고", "도로망 데이터가 없습니다.")
                return
        else:
            # 도로망 모드
            self.prediction_worker = PredictionWorker(self.current_file, self.model_path, self.file_mode)
        
        def update_progress(v, m):
            progress.setValue(v)
            progress.setLabelText(m)
        
        def on_completed(r):
            progress.close()
            self.on_prediction_completed(r)
        
        def on_error(e):
            progress.close()
            self.on_prediction_error(e)
        
        self.prediction_worker.progress.connect(update_progress)
        self.prediction_worker.prediction_completed.connect(on_completed)
        self.prediction_worker.error_occurred.connect(on_error)
        self.prediction_worker.start()

    def on_prediction_completed(self, result):
        """예측 완료 시 호출"""
        if result['success']:
            # 예측 결과 저장
            self.original_predictions = result.get('predictions', [])
            self.original_confidence_data = result.get('confidence_data', [])
            
            # 신뢰도 기반 필터링 및 적용
            self.filter_and_apply_predictions()
            
            # 다단계 자동 점 최적화 실행
            total_removed, final_points = self.run_multi_stage_optimization()
            self.statusBar().showMessage(f"AI 예측 + 다단계 최적화 완료 (총 {total_removed}개 제거, {final_points}개 유지)", 5000)
            
            # 수정 통계 업데이트
            self.update_modification_stats()
        else:
            self.result_label.setText("예측 실패!")
            self.statusBar().showMessage("AI 예측 실패", 3000)

    def on_prediction_error(self, error_msg):
        QMessageBox.critical(self, "오류", f"AI 예측 실패:\n{error_msg}")
        
        # 임시 파일 정리
        if hasattr(self.prediction_worker, 'temp_path') and self.prediction_worker.temp_path:
            try:
                Path(self.prediction_worker.temp_path).unlink()
                for ext in ['.shx', '.dbf', '.cpg', '.prj']:
                    Path(self.prediction_worker.temp_path.replace('.shp', ext)).unlink(missing_ok=True)
            except:
                pass

    def toggle_ai_predictions(self, checked):
        self.canvas_widget.canvas.show_ai_predictions = checked
        self.canvas_widget.canvas.update_display()
    
    def toggle_excel_points(self, checked):
        """실제 기준점(Excel 점) 표시/숨김 토글"""
        self.canvas_widget.canvas.show_excel_points = checked
        self.canvas_widget.canvas.update_display()
        logger.info(f"실제 기준점 표시: {'ON' if checked else 'OFF'}")

    def accept_all_predictions(self):
        """AI 예측 수락 + 연결성 검사 기반 점 삭제 기능"""
        # 1단계: AI 기반 삭제 먼저 실행
        ai_delete_count = 0
        ai_delete_attempts = len(self.canvas_widget.canvas.ai_points.get('delete', []))
        
        logger.info(f"AI 기반 삭제 시도: {ai_delete_attempts}개 점")
        
        for i, delete_point in enumerate(self.canvas_widget.canvas.ai_points.get('delete', [])):
            logger.info(f"AI 삭제 시도 {i+1}: 좌표 ({delete_point[0]:.1f}, {delete_point[1]:.1f})")
            
            # 삭제 전 주변 점 개수 확인
            before_counts = {cat: len(pts) for cat, pts in self.canvas_widget.canvas.points.items()}
            
            if self.canvas_widget.canvas.remove_nearest_point(delete_point[0], delete_point[1]):
                ai_delete_count += 1
                # 삭제 후 변화 확인
                after_counts = {cat: len(pts) for cat, pts in self.canvas_widget.canvas.points.items()}
                for cat in before_counts:
                    if before_counts[cat] != after_counts[cat]:
                        logger.info(f"✅ AI 삭제 성공: {cat} 카테고리에서 1개 점 제거")
                        break
            else:
                logger.info(f"❌ AI 삭제 실패: 30m 범위 내에 점이 없음")
        
        # 2단계: 연결성 검사는 지능형 클러스터링에 통합됨
        connectivity_delete_count = 0
        
        # AI 포인트 초기화
        self.canvas_widget.canvas.ai_points = {'intersection': [], 'curve': [], 'endpoint': [], 'delete': []}
        self.canvas_widget.canvas.update_display()
        self.calculate_and_display_distances()
        self.update_modification_stats()
        
        total_deleted = ai_delete_count + connectivity_delete_count
        if total_deleted > 0:
            QMessageBox.information(
                self, "완료", 
                f"삭제 완료!\n"
                f"AI 기반 삭제: {ai_delete_count}개\n"
                f"연결성 검사 삭제: {connectivity_delete_count}개\n"
                f"총 삭제: {total_deleted}개"
            )
        else:
            QMessageBox.information(self, "완료", "삭제할 점이 없습니다.")
    
    # remove_one_curve_point_by_connectivity 메서드 제거됨 - 지능형 클러스터링에 통합
    
    # calculate_accurate_point_distance 메서드 제거됨 - 더 이상 사용되지 않음
    
    def get_road_union(self):
        """도로망 폴리곤 가져오기"""
        try:
            # 1. 기존 road_union이 있으면 사용
            if hasattr(self, 'road_union') and self.road_union:
                return self.road_union
            
            # 2. 스켈레톤에서 도로 폴리곤 생성
            if self.canvas_widget.skeleton:
                from shapely.geometry import LineString, Point
                from shapely.ops import unary_union
                
                skeleton_lines = []
                skeleton_points = []
                
                # 스켈레톤 좌표 수집
                for i, point in enumerate(self.canvas_widget.skeleton):
                    if len(point) >= 2:
                        skeleton_points.append((float(point[0]), float(point[1])))
                
                # 연속된 점들로 LineString 생성
                if len(skeleton_points) > 1:
                    for i in range(len(skeleton_points) - 1):
                        try:
                            line = LineString([skeleton_points[i], skeleton_points[i + 1]])
                            if line.length > 1:  # 너무 짧은 선분 제외
                                skeleton_lines.append(line)
                        except:
                            continue
                
                # 스켈레톤 라인들 통합하고 버퍼 생성
                if skeleton_lines:
                    lines_union = unary_union(skeleton_lines)
                    road_buffer = lines_union.buffer(15)  # 15픽셀 버퍼
                    
                    # 캐시 저장
                    self.road_union = road_buffer
                    logger.info(f"도로망 폴리곤 생성: {len(skeleton_lines)}개 선분")
                    return road_buffer
            
            # 3. 폴백: 모든 점들로 간단한 영역 생성
            all_points = []
            for category in ['intersection', 'curve', 'endpoint']:
                for point in self.canvas_widget.canvas.points[category]:
                    all_points.append((float(point[0]), float(point[1])))
            
            if len(all_points) >= 3:
                from shapely.geometry import MultiPoint
                points_geom = MultiPoint(all_points)
                convex_hull = points_geom.convex_hull.buffer(20)
                
                self.road_union = convex_hull
                logger.info("폴백: 점들의 convex hull로 도로망 생성")
                return convex_hull
            
            logger.warning("도로망 폴리곤 생성 실패")
            return None
            
        except Exception as e:
            logger.error(f"도로망 폴리곤 가져오기 오류: {e}")
            return None
    
    # remove_clustered_points 메서드 제거됨 - 자동 지능형 클러스터링으로 대체

    def clear_user_modifications(self):
        reply = QMessageBox.question(self, "확인", "모든 사용자 수정 사항을 초기화하시겠습니까?", 
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.canvas_widget.canvas.points = {'intersection': [], 'curve': [], 'endpoint': []}
            
            if self.original_predictions:
                self.canvas_widget.canvas.ai_points = self.original_predictions.copy()
            
            self.canvas_widget.canvas.update_display()
            self.update_modification_stats()

    def update_modification_stats(self):
        """실제 vs AI 예측 통계 업데이트"""
        modifications = []
        
        # 각 카테고리별 비교
        for category in ['intersection', 'curve', 'endpoint']:
            ai_count = len(self.canvas_widget.canvas.ai_points.get(category, []))
            user_count = len(self.canvas_widget.canvas.points.get(category, []))
            
            modifications.append({
                'category': category.title(),
                'ai_count': ai_count,
                'user_count': user_count,
                'difference': user_count - ai_count
            })
        
        # 테이블 업데이트 (modification_table이 있는 경우에만)
        if hasattr(self, 'modification_table') and self.modification_table:
            self.modification_table.setRowCount(len(modifications))
            
            # 작은 폰트 설정
            small_font = QFont()
            small_font.setPointSize(8)
            
            for i, mod in enumerate(modifications):
                for j, key in enumerate(['category', 'ai_count', 'user_count', 'difference']):
                    item = QTableWidgetItem(str(mod[key]))
                    item.setFont(small_font)
                    self.modification_table.setItem(i, j, item)
                    
                    # 차이가 있으면 색상 표시
                    if j == 3 and mod['difference'] != 0:
                        item.setBackground(QColor(255, 200, 200))
        else:
            # 테이블이 없으면 로그로만 출력
            logger.info("수정 통계:")
            for mod in modifications:
                logger.info(f"  {mod['category']}: AI {mod['ai_count']}개 → 사용자 {mod['user_count']}개 (차이: {mod['difference']:+d})")

    def save_modified_session(self):
        if not self.current_file and self.file_mode == 'road':
            QMessageBox.warning(self, "경고", "저장할 파일이 없습니다.")
            return
        
        if self.file_mode == 'district' and not self.canvas_widget.skeleton:
            QMessageBox.warning(self, "경고", "저장할 데이터가 없습니다.")
            return
        
        try:
            labels = self.canvas_widget.canvas.points
            skeleton = self.canvas_widget.skeleton
            
            total_user = sum(len(pts) for pts in labels.values())
            total_ai = sum(len(pts) for pts in self.original_predictions.values()) if self.original_predictions else 0
            
            metadata = {
                'process': 'ai_correction',
                'model_used': self.model_path,
                'total_points': len(skeleton) if skeleton is not None else 0,
                'ai_predictions': total_ai,
                'user_corrections': total_user,
                'modification_rate': (total_user - total_ai) / total_ai if total_ai > 0 else 0,
                'file_mode': self.file_mode,
                'target_crs': self.get_target_crs() if self.file_mode == 'district' else None
            }
            
            # 멀티폴리곤 정보 추가
            polygon_info = None
            if self.file_mode == 'district' and self.current_polygon_data:
                polygon_info = {
                    'index': self.current_polygon_index + 1,
                    'total': self.current_polygon_data['total_polygons']
                }
            
            user_actions = []
            session_path = save_session(
                self.current_file, 
                labels, 
                skeleton, 
                metadata, 
                user_actions,
                polygon_info=polygon_info
            )
            
            if session_path:
                self.modified_sessions.append(session_path)
                self.session_count_label.setText(f"수정된 세션: {len(self.modified_sessions)}개")
                
                QMessageBox.information(
                    self, "성공",
                    f"수정된 세션이 저장되었습니다.\n수정률: {metadata['modification_rate']:.1%}"
                )
                
        except Exception as e:
            QMessageBox.critical(self, "오류", f"세션 저장 실패:\n{str(e)}")



    def start_batch_processing(self):
        folder = QFileDialog.getExistingDirectory(self, "Shapefile 폴더 선택")
        
        if folder:
            from process3_batch import BatchInferenceDialog
            dialog = BatchInferenceDialog(folder, self.model_path, self)
            dialog.exec_()

    # update_confidence 메서드 제거됨 - 고정 신뢰도 0.7 사용

    def upload_excel(self):
        """엑셀 파일에서 좌표 불러오기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "엑셀 파일 선택", "", "Excel Files (*.xlsx *.xls)")
            
        if file_path:
            try:
                import pandas as pd
                df = pd.read_excel(file_path, header=None)
                
                # Y(1열), X(2열) 좌표 추출
                self.excel_points = []
                for _, row in df.iterrows():
                    try:
                        y = float(row[0])  # 1열 (Y)
                        x = float(row[1])  # 2열 (X)
                        self.excel_points.append((x, y))  # 내부적으로는 (x, y) 순서
                    except (ValueError, IndexError):
                        continue
                
                logger.info(f"엑셀에서 {len(self.excel_points)}개 좌표 로드")
                
                # Canvas에 전달
                self.canvas_widget.canvas.excel_points = self.excel_points
                self.canvas_widget.canvas.update_display()
                
                # 점 개수 테이블 자동 업데이트
                self.update_point_count_table()
                
                QMessageBox.information(
                    self, "완료", 
                    f"엑셀에서 {len(self.excel_points)}개 좌표를 불러왔습니다."
                )
                
            except Exception as e:
                logger.error(f"엑셀 파일 로드 오류: {e}")
                QMessageBox.critical(self, "오류", f"엑셀 파일 로드 실패:\n{str(e)}")

    def update_point_count_table(self):
        """점 개수 비교 테이블 업데이트 (작은 글씨)"""
        # AI 검출점 개수 계산
        ai_count = 0
        if hasattr(self.canvas_widget.canvas, 'points'):
            for point_type in ['intersection', 'curve', 'endpoint']:
                ai_count += len(self.canvas_widget.canvas.points.get(point_type, []))
        
        # 실제 기준점 개수
        excel_count = len(self.excel_points)
        
        # 차이 계산
        difference = ai_count - excel_count
        
        # 작은 폰트 설정
        small_font = QFont()
        small_font.setPointSize(8)  # 작은 글씨 크기
        
        # 테이블 업데이트
        ai_item = QTableWidgetItem(str(ai_count))
        ai_item.setFont(small_font)
        self.point_count_table.setItem(0, 1, ai_item)
        
        excel_item = QTableWidgetItem(str(excel_count))
        excel_item.setFont(small_font)
        self.point_count_table.setItem(1, 1, excel_item)
        
        diff_item = QTableWidgetItem(f"{difference:+d}")
        diff_item.setFont(small_font)
        if difference > 0:
            diff_item.setForeground(QColor(255, 0, 0))  # 빨간색 (AI가 더 많음)
        elif difference < 0:
            diff_item.setForeground(QColor(0, 0, 255))  # 파란색 (실제가 더 많음)
        else:
            diff_item.setForeground(QColor(0, 128, 0))  # 초록색 (같음)
        
        self.point_count_table.setItem(2, 1, diff_item)
    
    def on_point_changed(self, category, x, y):
        """Canvas에서 점이 변경될 때 호출되는 콜백 - 점 개수 테이블 자동 업데이트"""
        self.update_point_count_table()
        logger.debug(f"점 변경 감지: {category} at ({x:.1f}, {y:.1f}) - 테이블 업데이트 완료")

    def filter_and_apply_predictions(self):
        """도로 경계선 기반 커브점 검출 및 적용"""
        if not hasattr(self, 'original_confidence_data') or not self.original_confidence_data:
            return
        
        # 스켈레톤 가져오기
        skeleton = getattr(self.canvas_widget, 'skeleton', None)
        if not skeleton:
            logger.warning("스켈레톤 데이터가 없어 기본 휴리스틱 사용")
            return
        
        # 1. 휴리스틱 교차점 검출
        intersections = getattr(self.canvas_widget, 'intersections', [])
        heuristic_intersections = intersections if intersections else []
        
        # 2. 휴리스틱 끝점 검출
        heuristic_endpoints = self.detect_heuristic_endpoints(skeleton)
        
        # 3. 🔄 도로 경계선 기반 커브점 검출 (최적 설정값)
        boundary_curves = self.detect_boundary_based_curves(
            skeleton, 
            sample_distance=15.0,      # 샘플링 거리
            curvature_threshold=0.20,  # 곡률 임계값
            road_buffer=3.0,           # 도로 버퍼
            cluster_radius=20.0        # 군집 반경
        )
        
        # 4. 교차점 근처 커브점 제거 (10m 이내)
        filtered_curves = self.remove_curves_near_intersections(
            boundary_curves, heuristic_intersections, threshold=10.0
        )
        
        # 5. 결과 설정
        self.canvas_widget.canvas.points = {
            'intersection': heuristic_intersections,
            'curve': filtered_curves,
            'endpoint': heuristic_endpoints
        }
        
        # 6. AI 삭제 포인트 처리 (신뢰도 기반)
        ai_delete_points = []
        threshold = self.ai_confidence_threshold
        
        for data in self.original_confidence_data:
            if data['confidence'] >= threshold and data['action'] == 4:  # 삭제 액션
                ai_delete_points.append(data['point'])
        
        # AI 삭제 포인트를 캔버스에 표시
        self.canvas_widget.canvas.ai_points = {
            'intersection': [],
            'curve': [],
            'endpoint': [],
            'delete': ai_delete_points
        }
        
        # 7. 화면 업데이트
        self.canvas_widget.canvas.update_display()
        self.calculate_and_display_distances()
        self.update_point_count_table()
        
        # 8. 결과 표시 (AI 스마트 분석 완료)
        total_optimized = len(heuristic_intersections) + len(filtered_curves) + len(heuristic_endpoints)
        self.result_label.setText(
            f"🤖 AI 스마트 분석 완료!\n"
            f"최적 특징점: {total_optimized}개 검출\n"
            f"교차점: {len(heuristic_intersections)}개\n"
            f"커브: {len(filtered_curves)}개\n"
            f"끝점: {len(heuristic_endpoints)}개\n"
            f"AI 품질 개선: {len(ai_delete_points)}개 후보"
        )

    def closeEvent(self, event):
        if self.canvas_widget.canvas and any(
            len(pts) > 0 for pts in self.canvas_widget.canvas.points.values()
        ):
            reply = QMessageBox.question(
                self, "종료 확인",
                "저장하지 않은 수정 사항이 있습니다.\n종료하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        event.accept()

    def run_ai_analysis(self):
        """AI 분석 실행 (지구계/도로망 모드 통합)"""
        if not self.current_file:
            QMessageBox.warning(self, "경고", "파일을 먼저 선택하세요.")
            return
        
        if not self.model_path:
            QMessageBox.warning(self, "경고", "AI 모델을 먼저 선택하세요.")
            return
        
        if self.file_mode == 'district':
            # 지구계 모드: 처리 + AI 분석 한번에
            self.process_district_file(self.current_file)
        else:
            # 도로망 모드: 이미 처리된 상태이므로 AI 분석만
            # (select_file에서 이미 process_road_file 호출됨)
            pass

    def resample_skeleton(self, skeleton, sample_distance=15.0):
        """스켈레톤을 일정 간격으로 리샘플링하여 노이즈 제거"""
        if len(skeleton) < 2:
            return skeleton
        
        resampled = [skeleton[0]]  # 첫 점은 항상 포함
        accumulated_dist = 0
        last_added_point = skeleton[0]
        
        for i in range(1, len(skeleton)):
            current_point = skeleton[i]
            
            # 이전 추가된 점과의 거리 계산
            dist = np.sqrt((current_point[0] - last_added_point[0])**2 + 
                          (current_point[1] - last_added_point[1])**2)
            accumulated_dist += dist
            
            # 샘플 거리 이상이면 점 추가
            if accumulated_dist >= sample_distance:
                resampled.append(current_point)
                last_added_point = current_point
                accumulated_dist = 0
        
        # 마지막 점도 포함 (끝점 보존)
        if len(resampled) > 1:
            last_dist = np.sqrt((skeleton[-1][0] - resampled[-1][0])**2 + 
                               (skeleton[-1][1] - resampled[-1][1])**2)
            if last_dist > sample_distance * 0.5:  # 마지막 점이 충분히 멀면 추가
                resampled.append(skeleton[-1])
        
        logger.info(f"스켈레톤 리샘플링: {len(skeleton)}개 → {len(resampled)}개 점 (간격: {sample_distance}m)")
        return resampled

    def detect_heuristic_curves(self, skeleton, curvature_threshold=0.4):
        """휴리스틱 기반 커브점 검출 (리샘플링 + 개선된 곡률 계산)"""
        curves = []
        
        # 스켈레톤 리샘플링으로 노이즈 제거
        resampled_skeleton = self.resample_skeleton(skeleton, sample_distance=20.0)
        
        if len(resampled_skeleton) < 5:  # 최소 5개 점 필요
            logger.info("리샘플링 후 점이 너무 적어 커브 검출 불가")
            return curves
        
        # 원본 스켈레톤과 리샘플링된 점의 매핑
        original_indices = []
        for resampled_point in resampled_skeleton:
            # 가장 가까운 원본 점 찾기
            min_dist = float('inf')
            closest_idx = 0
            for idx, orig_point in enumerate(skeleton):
                dist = np.sqrt((resampled_point[0] - orig_point[0])**2 + 
                             (resampled_point[1] - orig_point[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx
            original_indices.append(closest_idx)
        
        # 리샘플링된 스켈레톤에서 커브 검출
        for i in range(2, len(resampled_skeleton) - 2):
            try:
                # 5개 점 사용 (현재 점 중심으로 앞뒤 2개씩)
                points = resampled_skeleton[i-2:i+3]
                
                # 연속된 각도 변화 계산
                angles = []
                for j in range(len(points) - 2):
                    p1 = np.array(points[j])
                    p2 = np.array(points[j+1])
                    p3 = np.array(points[j+2])
                    
                    v1 = p2 - p1
                    v2 = p3 - p2
                    
                    len1 = np.linalg.norm(v1)
                    len2 = np.linalg.norm(v2)
                    
                    if len1 > 0 and len2 > 0:
                        v1_norm = v1 / len1
                        v2_norm = v2 / len2
                        
                        # 내적으로 각도 계산
                        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        angles.append(angle)
                
                # 평균 각도와 최대 각도 모두 고려
                if angles:
                    max_angle = max(angles)
                    avg_angle = sum(angles) / len(angles)
                    
                    # 최대 각도가 임계값 이상이고, 평균 각도도 일정 수준 이상일 때만 커브로 판정
                    if max_angle > curvature_threshold and avg_angle > curvature_threshold * 0.5:
                        # 원본 스켈레톤의 해당 구간에서 가장 곡률이 큰 점만 선택
                        start_idx = original_indices[i-1] if i > 0 else original_indices[i]
                        end_idx = original_indices[i+1] if i < len(original_indices)-1 else original_indices[i]
                        
                        # 중간 지점 하나만 커브점으로 추가
                        mid_idx = (start_idx + end_idx) // 2
                        if 0 <= mid_idx < len(skeleton):
                            curves.append(tuple(skeleton[mid_idx][:2]))
                        
            except Exception as e:
                logger.debug(f"커브 검출 오류 at {i}: {e}")
                continue
        
        # 중복 제거
        curves = list(set(curves))
        
        logger.info(f"🔄 휴리스틱 커브 검출: {len(curves)}개 (리샘플링 적용)")
        return curves

    def detect_boundary_based_curves(self, skeleton, sample_distance=15.0, curvature_threshold=0.20, 
                                   road_buffer=3.0, cluster_radius=20.0):
        """도로 경계선 기반 커브점 검출 (전체 도로망 통합)"""
        if not skeleton or len(skeleton) < 5:
            logger.info("스켈레톤이 너무 짧아 경계선 기반 커브 검출 불가")
            return []
        
        try:
            from shapely.geometry import LineString, Point, MultiLineString
            
            # 🔄 전체 스켈레톤을 하나의 도로망으로 통합
            skeleton_coords = [(float(p[0]), float(p[1])) for p in skeleton if len(p) >= 2]
            if len(skeleton_coords) < 2:
                return []
            
            # 연속된 좌표들을 LineString으로 변환
            skeleton_line = LineString(skeleton_coords)
            
            # 🔄 전체 도로망을 하나로 합치기 (unary_union)
            # 실제 도로 모양 그대로 유지하면서 통합
            unified_road = skeleton_line  # 이미 하나의 연속된 라인
            
            # 🔄 통합된 도로에 버퍼 적용 (실제 도로 모양 유지)
            road_polygon = unified_road.buffer(road_buffer)
            
            # 복잡한 도로 형태 처리 (Polygon 또는 MultiPolygon)
            if road_polygon.geom_type == 'Polygon':
                boundaries = [road_polygon.exterior]
            elif road_polygon.geom_type == 'MultiPolygon':
                boundaries = []
                if hasattr(road_polygon, 'geoms'):
                    for poly in road_polygon.geoms:  # type: ignore
                        if hasattr(poly, 'exterior'):
                            boundaries.append(poly.exterior)
            else:
                logger.warning(f"예상치 못한 geometry 타입: {road_polygon.geom_type}")
                return []
            
            logger.info(f"도로망 통합 완료: {len(boundaries)}개 경계선")
            
            # 🔄 모든 경계선에서 커브점 검출
            all_curvature_points = []
            
            for boundary in boundaries:
                total_length = boundary.length
                if total_length < sample_distance:
                    continue
            
                # 각 경계선을 따라 샘플링
                num_samples = max(10, int(total_length / sample_distance))
                
                for i in range(num_samples):
                    distance = (i * sample_distance) % total_length
                    
                    # 곡률 계산
                    curvature = self.calculate_curvature_at_distance(boundary, distance, sample_distance)
                    
                    if curvature > curvature_threshold:
                        point = boundary.interpolate(distance)
                        all_curvature_points.append({
                            'point': (point.x, point.y),
                            'curvature': curvature
                        })
            
            logger.info(f"통합 경계선 곡률 변화 지점: {len(all_curvature_points)}개 검출")
            
            # 🔄 전체 커브점에 대해 군집화
            if len(all_curvature_points) < 2:
                final_curves = [cp['point'] for cp in all_curvature_points]
            else:
                try:
                    from sklearn.cluster import DBSCAN
                    points = np.array([cp['point'] for cp in all_curvature_points])
                    clustering = DBSCAN(eps=cluster_radius, min_samples=2)
                    labels = clustering.fit_predict(points)
                except ImportError:
                    logger.warning("scikit-learn이 필요합니다. 기본 군집화 사용")
                    final_curves = [cp['point'] for cp in all_curvature_points]
                
                # 군집별 중심점 계산
                final_curves = []
                unique_labels = set(labels)
                
                for label in unique_labels:
                    if label == -1:  # 노이즈 제외
                        continue
                    
                    cluster_mask = labels == label
                    cluster_points = points[cluster_mask]
                    cluster_curvatures = [all_curvature_points[i]['curvature'] 
                                        for i, mask in enumerate(cluster_mask) if mask]
                    
                    # 곡률 가중 평균으로 중심점 계산
                    weights = np.array(cluster_curvatures)
                    center_x = np.average(cluster_points[:, 0], weights=weights)
                    center_y = np.average(cluster_points[:, 1], weights=weights)
                    final_curves.append((center_x, center_y))
            
            # 커브점을 가장 가까운 스켈레톤 점으로 이동
            corrected_curves = []
            for curve_point in final_curves:
                closest_skeleton_point = self.find_closest_skeleton_point(curve_point, skeleton)
                if closest_skeleton_point:
                    corrected_curves.append(closest_skeleton_point)
            
            logger.info(f"통합 도로망 경계선 기반 커브점 검출 완료: {len(corrected_curves)}개")
            return corrected_curves
            
        except Exception as e:
            logger.error(f"경계선 기반 커브점 검출 실패: {e}")
            return []
    
    def calculate_curvature_at_distance(self, boundary, distance, window=20.0):
        """특정 거리에서의 곡률 계산"""
        try:
            # 앞뒤 점들 구하기
            d1 = max(0, distance - window)
            d2 = min(boundary.length, distance + window)
            
            if d2 - d1 < window * 0.5:
                return 0.0
            
            p1 = boundary.interpolate(d1)
            p2 = boundary.interpolate(distance)
            p3 = boundary.interpolate(d2)
            
            # 벡터 계산
            v1 = np.array([p2.x - p1.x, p2.y - p1.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            # 각도 변화 계산
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 > 0 and len2 > 0:
                v1_norm = v1 / len1
                v2_norm = v2 / len2
                
                cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                return angle
            
            return 0.0
            
        except:
            return 0.0
    
    def find_closest_skeleton_point(self, curve_point, skeleton):
        """커브점에서 가장 가까운 스켈레톤 점 찾기"""
        if not skeleton:
            return None
        
        min_dist = float('inf')
        closest_point = None
        
        for skel_point in skeleton:
            if len(skel_point) < 2:
                continue
            
            dist = np.sqrt((curve_point[0] - skel_point[0])**2 + 
                          (curve_point[1] - skel_point[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_point = (float(skel_point[0]), float(skel_point[1]))
        
        return closest_point
    
    def remove_curves_near_intersections(self, curves, intersections, threshold=10.0):
        """교차점 근처 커브점 제거"""
        if not curves or not intersections:
            return curves
        
        filtered_curves = []
        
        for curve in curves:
            near_intersection = False
            
            for intersection in intersections:
                if len(intersection) < 2:
                    continue
                
                dist = np.sqrt((curve[0] - intersection[0])**2 + 
                              (curve[1] - intersection[1])**2)
                
                if dist <= threshold:
                    near_intersection = True
                    break
            
            if not near_intersection:
                filtered_curves.append(curve)
        
        logger.info(f"교차점 근처 커브점 제거: {len(curves)} → {len(filtered_curves)}개")
        return filtered_curves

    # auto_remove_road_endpoints 메서드 제거됨 - 지능형 클러스터링에 통합
    
    def enable_manual_edit_mode(self):
        """수동 점 편집 모드 자동 활성화"""
        self.statusBar().showMessage("✋ 수동 점 편집 활성화 - 좌클릭:커브, 우클릭:끝점, D키:삭제", 0)
        logger.info("수동 점 편집 모드가 자동으로 활성화되었습니다.")

    def export_points_to_shp(self):
        """최종 점들을 SHP 파일로 저장"""
        try:
            # 점이 있는지 확인
            points = self.canvas_widget.canvas.points
            if not any(len(pts) for pts in points.values()):
                QMessageBox.warning(self, "경고", "저장할 점이 없습니다.")
                return
            
            # 파일 저장 다이얼로그
            file_path, _ = QFileDialog.getSaveFileName(
                self, "점 SHP 파일 저장", "", "Shapefile (*.shp)"
            )
            
            if not file_path:
                return
            
            # 점들을 GeoDataFrame으로 변환
            all_points = []
            for category, point_list in points.items():
                for point in point_list:
                    all_points.append({
                        'geometry': Point(float(point[0]), float(point[1])),
                        'category': category,
                        'point_id': len(all_points) + 1,
                        'x_coord': float(point[0]),
                        'y_coord': float(point[1])
                    })
            
            if not all_points:
                QMessageBox.warning(self, "경고", "저장할 점이 없습니다.")
                return
            
            # GeoDataFrame 생성
            gdf = gpd.GeoDataFrame(all_points)
            
            # 좌표계 설정
            if self.file_mode == 'district':
                crs = self.get_target_crs()
            else:
                # 도로망 모드에서는 기본 좌표계 사용
                crs = 'EPSG:5186'  # 또는 원본 파일의 좌표계
            
            gdf.crs = crs
            
            # 파일 저장
            gdf.to_file(file_path, encoding='utf-8')
            
            QMessageBox.information(
                self, "성공", 
                f"점 데이터가 저장되었습니다!\n"
                f"파일: {file_path}\n"
                f"점 개수: {len(all_points)}개\n"
                f"좌표계: {crs}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"SHP 파일 저장 실패:\n{str(e)}")
            logger.error(f"SHP 저장 오류: {e}")

    def export_all_to_gpkg(self):
        """지구계, 도로망, 점을 모두 GPKG 파일로 저장"""
        try:
            # 데이터 확인
            points = self.canvas_widget.canvas.points
            if not any(len(pts) for pts in points.values()):
                QMessageBox.warning(self, "경고", "저장할 점이 없습니다.")
                return
            
            # 파일 저장 다이얼로그
            file_path, _ = QFileDialog.getSaveFileName(
                self, "전체 결과 GPKG 파일 저장", "", "GeoPackage (*.gpkg)"
            )
            
            if not file_path:
                return
            
            # 좌표계 설정
            if self.file_mode == 'district':
                crs = self.get_target_crs()
            else:
                crs = 'EPSG:5186'
            
            # 1. 점 레이어 생성
            point_data = []
            for category, point_list in points.items():
                for point in point_list:
                    point_data.append({
                        'geometry': Point(float(point[0]), float(point[1])),
                        'category': category,
                        'point_id': len(point_data) + 1,
                        'x_coord': float(point[0]),
                        'y_coord': float(point[1])
                    })
            
            if point_data:
                points_gdf = gpd.GeoDataFrame(point_data, crs=crs)
                points_gdf.to_file(file_path, layer='points', driver='GPKG')
            
            # 2. 도로망 레이어 저장 (있는 경우)
            try:
                if hasattr(self.canvas_widget, 'road_data') and self.canvas_widget.road_data is not None:
                    road_gdf = self.canvas_widget.road_data.copy()
                    road_gdf = road_gdf.to_crs(crs)
                    road_gdf.to_file(file_path, layer='roads', driver='GPKG')
                elif self.canvas_widget.skeleton:
                    # 스켈레톤을 LineString으로 변환
                    from shapely.geometry import LineString
                    skeleton_coords = [(float(p[0]), float(p[1])) for p in self.canvas_widget.skeleton if len(p) >= 2]  # type: ignore
                    if len(skeleton_coords) > 1:
                        skeleton_line = LineString(skeleton_coords)
                        skeleton_gdf = gpd.GeoDataFrame(
                            [{'geometry': skeleton_line, 'type': 'skeleton'}], 
                            crs=crs
                        )
                        skeleton_gdf.to_file(file_path, layer='skeleton', driver='GPKG')
            except Exception as e:
                logger.warning(f"도로망 레이어 저장 실패: {e}")
            
            # 3. 지구계 경계 레이어 저장 (지구계 모드인 경우)
            try:
                if self.file_mode == 'district' and hasattr(self.canvas_widget, 'background_data'):
                    if self.canvas_widget.background_data is not None:
                        # 지구계 경계가 Polygon인 경우
                        from shapely.geometry import Polygon, MultiPolygon
                        boundary_geom = self.canvas_widget.background_data
                        
                        if isinstance(boundary_geom, (Polygon, MultiPolygon)):
                            boundary_gdf = gpd.GeoDataFrame(
                                [{'geometry': boundary_geom, 'type': 'district_boundary'}], 
                                crs=crs
                            )
                            boundary_gdf.to_file(file_path, layer='district_boundary', driver='GPKG')
            except Exception as e:
                logger.warning(f"지구계 경계 레이어 저장 실패: {e}")
            
            # 레이어 정보 수집
            saved_layers = ['points']
            layer_info = f"점 레이어: {len(point_data)}개 점"
            
            try:
                # 저장된 레이어 확인
                import fiona
                with fiona.open(file_path, layer='roads') as layer:
                    saved_layers.append('roads')
                    layer_info += f"\n도로망 레이어: {len(layer)}개 도로"
            except:
                pass
                
            try:
                with fiona.open(file_path, layer='skeleton') as layer:
                    saved_layers.append('skeleton')
                    layer_info += f"\n스켈레톤 레이어: 포함"
            except:
                pass
                
            try:
                with fiona.open(file_path, layer='district_boundary') as layer:
                    saved_layers.append('district_boundary')
                    layer_info += f"\n지구계 경계 레이어: 포함"
            except:
                pass
            
            QMessageBox.information(
                self, "성공", 
                f"전체 결과가 GPKG 파일로 저장되었습니다!\n\n"
                f"파일: {file_path}\n"
                f"좌표계: {crs}\n"
                f"저장된 레이어: {', '.join(saved_layers)}\n\n"
                f"{layer_info}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"GPKG 파일 저장 실패:\n{str(e)}")
            logger.error(f"GPKG 저장 오류: {e}")

    def get_current_crs(self):
        """현재 작업 중인 데이터의 좌표계 반환"""
        if self.file_mode == 'district':
            return self.get_target_crs()
        else:
            # 도로망 모드에서는 원본 파일의 좌표계 또는 기본값
            return 'EPSG:5186'
    
    def auto_calculate_distances(self):
        """AI 분석 후 자동으로 거리 계산 및 표시"""
        try:
            # 스마트 거리 연결 로직 실행
            self.calculate_smart_distance_connections()
            
            logger.info("✅ AI 분석 후 자동 거리 계산 완료")
            
        except Exception as e:
            logger.error(f"자동 거리 계산 실패: {e}")
            self.distance_label.setText("점간 거리: 계산 실패")
    
    # recalculate_distances 메서드 제거됨 - 자동 거리 계산으로 대체
    
    def calculate_smart_distance_connections(self):
        """스마트 거리 연결 로직 - 요구사항에 맞는 완전한 구현"""
        # 1. 기본 데이터 검증
        if not self.canvas_widget.canvas.points:
            self.distance_label.setText("점간 거리: 분석할 점이 없습니다")
            return
        
        # 2. 모든 점 수집
        all_points = []
        for category in ['intersection', 'curve', 'endpoint']:
            for point in self.canvas_widget.canvas.points.get(category, []):
                all_points.append({
                    'coord': (float(point[0]), float(point[1])),
                    'category': category
                })
        
        if len(all_points) < 2:
            self.distance_label.setText("점간 거리: 점이 부족합니다 (최소 2개 필요)")
            return
        
        # 3. 스켈레톤 및 도로망 데이터 확인
        skeleton = self.canvas_widget.skeleton
        if not skeleton:
            self.distance_label.setText("점간 거리: 스켈레톤 데이터가 없습니다")
            return
        
        road_union = self.get_road_union()
        if not road_union:
            self.distance_label.setText("점간 거리: 도로망 데이터가 없습니다")
            return
        
        # 4. 스마트 거리 연결 찾기
        valid_connections = self.find_smart_distance_connections(all_points, skeleton, road_union)
        
        # 5. Canvas에 거리 연결선 전달 (점선 + 거리 숫자)
        self.update_canvas_distance_display(valid_connections)
        
        # 6. 통계 표시
        self.update_distance_statistics(valid_connections)
        
        logger.info(f"🔗 스마트 거리 연결 완료: {len(valid_connections)}개 연결")
    
    def find_smart_distance_connections(self, all_points, skeleton, road_union):
        """스마트 거리 연결 찾기 - 모든 조건 적용"""
        from shapely.geometry import LineString, Point
        import numpy as np
        
        valid_connections = []
        skeleton_array = np.array(skeleton) if skeleton else np.array([])
        
        if len(skeleton_array) == 0:
            return valid_connections
        
        # 모든 점 쌍에 대해 연결 가능성 검사
        for i, point1_data in enumerate(all_points):
            for j, point2_data in enumerate(all_points[i+1:], i+1):
                point1 = point1_data['coord']
                point2 = point2_data['coord']
                
                # 조건 1: 유클리드 직선거리 계산 및 범위 확인 (15-300m)
                distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                if not (15.0 <= distance <= 300.0):
                    continue
                
                # 조건 2: 스켈레톤 경로 기반 연결 판단
                skeleton_path = self.find_skeleton_path_between_points(point1, point2, skeleton_array)
                if not skeleton_path:
                    continue
                
                # 조건 3: 중간에 다른 점 끼어있으면 연결 안함
                if self.has_intermediate_points_on_path(skeleton_path, all_points, point1, point2):
                    continue
                
                # 조건 4: 도로망 50cm 버퍼 or 80% 겹침 조건 만족
                if not self.satisfies_road_network_condition(skeleton_path, road_union):
                    continue
                
                # 모든 조건 만족 시 연결 추가
                valid_connections.append({
                    'point1': point1,
                    'point2': point2,
                    'distance': distance,
                    'category1': point1_data['category'],
                    'category2': point2_data['category'],
                    'skeleton_path': skeleton_path
                })
        
        # 거리 순으로 정렬하고 상위 20개만 선택
        valid_connections.sort(key=lambda x: x['distance'])
        return valid_connections[:20]
    
    def has_intermediate_points_on_path(self, skeleton_path, all_points, point1, point2):
        """스켈레톤 경로 중간에 다른 점이 끼어있는지 확인"""
        try:
            from shapely.geometry import LineString, Point
            
            if len(skeleton_path) < 2:
                return False
            
            path_line = LineString(skeleton_path)
            
            # 다른 모든 점들에 대해 확인
            for point_data in all_points:
                other_point = point_data['coord']
                
                # 현재 검사 중인 두 점은 제외
                if (abs(other_point[0] - point1[0]) < 1 and abs(other_point[1] - point1[1]) < 1) or \
                   (abs(other_point[0] - point2[0]) < 1 and abs(other_point[1] - point2[1]) < 1):
                    continue
                
                # 다른 점이 스켈레톤 경로 근처(8m 이내)에 있는지 확인
                other_point_geom = Point(other_point)
                distance_to_path = path_line.distance(other_point_geom)
                
                if distance_to_path <= 8.0:
                    return True  # 중간에 다른 점이 끼어있음
            
            return False
            
        except Exception as e:
            logger.warning(f"중간점 확인 오류: {e}")
            return True  # 오류 시 안전하게 연결 안함
    
    def satisfies_road_network_condition(self, skeleton_path, road_union):
        """도로망 50cm 버퍼 or 80% 겹침 조건 확인"""
        try:
            from shapely.geometry import LineString
            
            if len(skeleton_path) < 2:
                return False
            
            path_line = LineString(skeleton_path)
            
            # 조건 1: 도로망 50cm 버퍼 내에 완전히 포함
            road_buffer_small = road_union.buffer(0.5)
            if road_buffer_small.contains(path_line):
                return True
            
            # 조건 2: 80% 이상 도로망과 겹침 (20m 버퍼 사용)
            road_buffer_large = road_union.buffer(20.0)
            intersection = path_line.intersection(road_buffer_large)
            
            if hasattr(intersection, 'length'):
                overlap_ratio = intersection.length / path_line.length
                return overlap_ratio >= 0.8
            
            return False
            
        except Exception as e:
            logger.warning(f"도로망 조건 확인 오류: {e}")
            return False
    
    def update_canvas_distance_display(self, valid_connections):
        """Canvas에 거리 연결선과 숫자 표시"""
        try:
            # Canvas에 distance_connections 설정
            display_connections = []
            for conn in valid_connections:
                display_connections.append({
                    'point1': conn['point1'],
                    'point2': conn['point2'],
                    'distance': conn['distance']
                })
            
            # Canvas에 전달
            if hasattr(self.canvas_widget.canvas, 'distance_connections'):
                self.canvas_widget.canvas.distance_connections = display_connections
                self.canvas_widget.canvas.show_distance_connections = True
                self.canvas_widget.canvas.update_display()
            else:
                # distance_connections 속성이 없으면 생성
                self.canvas_widget.canvas.distance_connections = display_connections
                self.canvas_widget.canvas.show_distance_connections = True
                self.canvas_widget.canvas.update_display()
            
            logger.info(f"📊 Canvas 거리 표시 업데이트: {len(display_connections)}개 연결")
            
        except Exception as e:
            logger.error(f"Canvas 거리 표시 업데이트 실패: {e}")
    
    def update_distance_statistics(self, valid_connections):
        """거리 통계 표시"""
        try:
            if valid_connections:
                total_connections = len(valid_connections)
                distances = [conn['distance'] for conn in valid_connections]
                avg_distance = sum(distances) / total_connections
                min_distance = min(distances)
                max_distance = max(distances)
                
                self.distance_label.setText(
                    f"🔗 스마트 거리 연결: {total_connections}개\n"
                    f"평균: {avg_distance:.1f}m | 최소: {min_distance:.1f}m | 최대: {max_distance:.1f}m"
                )
            else:
                self.distance_label.setText("점간 거리: 연결 가능한 점이 없습니다")
                
        except Exception as e:
            logger.error(f"거리 통계 업데이트 실패: {e}")
            self.distance_label.setText("점간 거리: 통계 계산 실패")
    

    
    def has_visual_connectivity(self, p1, p2):
        """두 점이 시각적으로 연결되어 보이는지 판단"""
        try:
            # 1. 직선 경로에 다른 점들이 방해하지 않는지 확인
            if self.has_blocking_points_between(p1, p2):
                return False
            
            # 2. 직선이 도로망(스켈레톤)을 따라 자연스럽게 이어지는지 확인
            if not self.line_follows_road_network(p1, p2):
                return False
            
            # 3. 시각적으로 자연스러운 연결인지 확인 (각도, 방향성)
            if not self.is_visually_natural_connection(p1, p2):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"시각적 연결성 체크 오류: {e}")
            return False
    
    def has_blocking_points_between(self, p1, p2):
        """두 점 사이의 직선 경로에 다른 점이 방해하는지 확인"""
        try:
            from shapely.geometry import LineString, Point
            
            line = LineString([p1, p2])
            buffer_zone = line.buffer(8)  # 8m 버퍼
            
            # 모든 다른 점들이 이 버퍼 안에 있는지 확인
            for category in ['intersection', 'curve', 'endpoint']:
                for point in self.canvas_widget.canvas.points.get(category, []):
                    point_geom = Point(point[0], point[1])
                    
                    # 시작점, 끝점 제외
                    if point_geom.distance(Point(p1)) < 5 or point_geom.distance(Point(p2)) < 5:
                        continue
                    
                    # 직선 경로를 방해하는 점이 있으면 False
                    if buffer_zone.contains(point_geom):
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"방해점 체크 오류: {e}")
            return True  # 오류 시 안전하게 방해됨으로 간주
    
    def line_follows_road_network(self, p1, p2):
        """직선이 도로망을 자연스럽게 따라가는지 확인"""
        try:
            if not self.canvas_widget.skeleton:
                return True  # 스켈레톤이 없으면 통과
            
            from shapely.geometry import LineString
            
            line = LineString([p1, p2])
            
            # 스켈레톤 좌표 처리
            skeleton_coords = []
            for point in self.canvas_widget.skeleton:  # type: ignore
                if len(point) >= 2:
                    skeleton_coords.append((float(point[0]), float(point[1])))
            
            if len(skeleton_coords) < 2:
                return True
            
            skeleton_line = LineString(skeleton_coords)
            
            # 직선이 스켈레톤과 얼마나 겹치는지 확인
            buffered_skeleton = skeleton_line.buffer(10)  # 10m 버퍼
            overlap_length = line.intersection(buffered_skeleton).length
            
            # 직선의 70% 이상이 도로망 근처에 있으면 자연스러운 연결
            return (overlap_length / line.length) > 0.7
            
        except Exception as e:
            logger.warning(f"도로망 따라가기 체크 오류: {e}")
            return True  # 오류 시 안전하게 통과
    
    def is_visually_natural_connection(self, p1, p2):
        """시각적으로 자연스러운 연결인지 판단"""
        try:
            # 너무 급격한 각도 변화는 부자연스러움
            dx = abs(p1[0] - p2[0])
            dy = abs(p1[1] - p2[1])
            
            # 직선에 가까우면 자연스러움
            if dx < 5 or dy < 5:  # 거의 수직 또는 수평
                return True
            
            # 적당한 대각선도 자연스러움
            angle_ratio = min(dx, dy) / max(dx, dy)
            return angle_ratio > 0.3  # 너무 급격하지 않은 각도
            
        except Exception as e:
            logger.warning(f"자연스러운 연결 체크 오류: {e}")
            return True  # 오류 시 안전하게 통과

    def find_valid_skeleton_connections(self, all_points, skeleton, road_union):
        """스켈레톤 경로 기반 유효한 점 연결 찾기"""
        from shapely.geometry import LineString, Point
        import numpy as np
        
        valid_connections = []
        
        # 스켈레톤을 numpy 배열로 변환
        skeleton_array = np.array(skeleton) if skeleton else np.array([])
        if len(skeleton_array) == 0:
            return valid_connections
        
        # 모든 점 쌍에 대해 연결 가능성 검사
        for i, point1_data in enumerate(all_points):
            for j, point2_data in enumerate(all_points[i+1:], i+1):
                point1 = point1_data['coord']
                point2 = point2_data['coord']
                
                # 1. 스켈레톤 경로 찾기
                skeleton_path = self.find_skeleton_path_between_points(point1, point2, skeleton_array)
                
                if not skeleton_path:
                    continue
                
                # 2. 중간에 다른 점이 끼어있는지 확인
                if self.has_intermediate_points(skeleton_path, all_points, point1, point2):
                    continue
                
                # 3. 스켈레톤 경로가 도로망 조건을 만족하는지 확인
                if not self.skeleton_path_satisfies_road_condition(skeleton_path, road_union):
                    continue
                
                # 4. 유클리드 직선거리 계산
                distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                
                # 5. 거리 범위 확인 (15-300m)
                if 15.0 <= distance <= 300.0:
                    valid_connections.append({
                        'point1': point1,
                        'point2': point2,
                        'distance': distance,
                        'category1': point1_data['category'],
                        'category2': point2_data['category']
                    })
        
        # 거리 순으로 정렬
        valid_connections.sort(key=lambda x: x['distance'])
        
        # 너무 많으면 상위 20개만 선택
        return valid_connections[:20]
    
    def find_skeleton_path_between_points(self, point1, point2, skeleton_array):
        """두 점 사이의 스켈레톤 경로 찾기"""
        try:
            # 각 점에서 가장 가까운 스켈레톤 점 찾기
            start_idx = self.find_closest_skeleton_index(point1, skeleton_array)
            end_idx = self.find_closest_skeleton_index(point2, skeleton_array)
            
            if start_idx is None or end_idx is None:
                return None
            
            if start_idx == end_idx:
                return None
            
            # 스켈레톤 경로 생성 (연속된 인덱스들)
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            
            # start_idx부터 end_idx까지의 스켈레톤 점들
            path_points = skeleton_array[start_idx:end_idx+1]
            
            return path_points.tolist()
            
        except Exception as e:
            logger.warning(f"스켈레톤 경로 찾기 오류: {e}")
            return None
    
    def find_closest_skeleton_index(self, point, skeleton_array):
        """점에서 가장 가까운 스켈레톤 점의 인덱스 찾기"""
        try:
            distances = np.sqrt(np.sum((skeleton_array - np.array(point))**2, axis=1))
            closest_idx = np.argmin(distances)
            
            # 너무 멀면 (50m 이상) 연결 안함
            if distances[closest_idx] > 50.0:
                return None
                
            return closest_idx
            
        except Exception as e:
            logger.warning(f"가장 가까운 스켈레톤 점 찾기 오류: {e}")
            return None
    
    def has_intermediate_points(self, skeleton_path, all_points, point1, point2):
        """스켈레톤 경로 중간에 다른 점이 끼어있는지 확인"""
        try:
            from shapely.geometry import LineString, Point
            
            # 스켈레톤 경로를 LineString으로 변환
            if len(skeleton_path) < 2:
                return False
            
            path_line = LineString(skeleton_path)
            
            # 다른 모든 점들에 대해 확인
            for point_data in all_points:
                other_point = point_data['coord']
                
                # 현재 검사 중인 두 점은 제외
                if (abs(other_point[0] - point1[0]) < 1 and abs(other_point[1] - point1[1]) < 1) or \
                   (abs(other_point[0] - point2[0]) < 1 and abs(other_point[1] - point2[1]) < 1):
                    continue
                
                # 다른 점이 스켈레톤 경로 근처(10m 이내)에 있는지 확인
                other_point_geom = Point(other_point)
                distance_to_path = path_line.distance(other_point_geom)
                
                if distance_to_path <= 10.0:
                    return True  # 중간에 다른 점이 끼어있음
            
            return False
            
        except Exception as e:
            logger.warning(f"중간점 확인 오류: {e}")
            return True  # 오류 시 안전하게 연결 안함
    
    def skeleton_path_satisfies_road_condition(self, skeleton_path, road_union):
        """스켈레톤 경로가 도로망 조건을 만족하는지 확인"""
        try:
            from shapely.geometry import LineString
            
            if len(skeleton_path) < 2:
                return False
            
            path_line = LineString(skeleton_path)
            
            # 조건 1: 도로망 50cm(0.5m) 버퍼 내에 완전히 포함
            road_buffer_small = road_union.buffer(0.5)
            if road_buffer_small.contains(path_line):
                return True
            
            # 조건 2: 80% 이상 도로망과 겹침 (20m 버퍼 사용)
            road_buffer_large = road_union.buffer(20.0)
            intersection = path_line.intersection(road_buffer_large)
            
            if hasattr(intersection, 'length'):
                overlap_ratio = intersection.length / path_line.length
                return overlap_ratio >= 0.8
            
            return False
            
        except Exception as e:
            logger.warning(f"도로망 조건 확인 오류: {e}")
            return False


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # type: ignore
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # type: ignore
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = InferenceTool()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()