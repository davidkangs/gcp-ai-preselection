"""
파일명: process1_labeling_tool_improved.py
설명: 프로세스 1 - 라벨링 도구 (지구계 자동 클리핑 + 파일 네비게이션)
"""
import sys
import os
import time
import tempfile
from pathlib import Path
import numpy as np
import geopandas as gpd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox, 
    QTextEdit, QProgressDialog, QSplitter, QRadioButton,
    QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

# 프로젝트 모듈 import
from src.core.skeleton_extractor import SkeletonExtractor
from src.core.district_road_clipper import DistrictRoadClipper
from src.ui.canvas_widget import CanvasWidget
from src.utils import save_session, get_polygon_session_name
from enhanced_heuristic_detector_v2 import EnhancedHeuristicDetectorV2
from src.learning import DQNDataCollector


class ProcessingThread(QThread):
    """백그라운드 처리 스레드"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, file_path, file_mode='road', target_crs='EPSG:5186'):
        super().__init__()
        self.file_path = file_path
        self.file_mode = file_mode
        self.target_crs = target_crs
        self.skeleton_extractor = SkeletonExtractor()
    
    def run(self):
        """처리 실행"""
        try:
            if self.file_mode == 'district':
                self.process_district_file()
            else:
                self.process_road_file()
        except Exception as e:
            self.error.emit(str(e))
    
    def process_district_file(self):
        """지구계 파일 처리 - 도로망 자동 클리핑"""
        self.progress.emit(10, "지구계 파일 읽는 중...")
        
        clipper = DistrictRoadClipper()
        results = clipper.process_district_file(
            self.file_path,
            target_crs=self.target_crs,
            auto_find_road=True
        )
        
        if not results['success']:
            if results['error'] == "도로망 파일을 찾을 수 없음":
                self.finished.emit({
                    'success': False,
                    'need_manual_selection': True,
                    'polygons': results['polygons'],
                    'error': results['error']
                })
                return
            else:
                self.error.emit(results['error'])
                return
        
        # 첫 번째 폴리곤만 처리
        if results['polygons']:
            first_polygon = results['polygons'][0]
            road_gdf = first_polygon.get('clipped_road')
            
            if road_gdf is None or road_gdf.empty:
                self.error.emit("클리핑된 도로망이 없습니다")
                return
            
            self.progress.emit(40, "스켈레톤 추출 중...")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, "temp_road.shp")
                road_gdf.to_file(temp_path)
                
                if not os.path.exists(temp_path):
                    self.error.emit("임시 파일 생성 실패")
                    return
                
                skeleton, intersections = self.skeleton_extractor.process_shapefile(temp_path)
            
            self.progress.emit(90, "결과 정리 중...")
            result = {
                'success': True,
                'mode': 'district',
                'district_results': results,
                'road_gdf': road_gdf,
                'skeleton': skeleton,
                'intersections': intersections,
                'curves': [],
                'endpoints': [],
                'processing_time': 0,
                'district_polygon_gdf': first_polygon['geometry']
            }
            
            self.progress.emit(100, "완료!")
            self.finished.emit(result)
    
    def process_road_file(self):
        """기존 도로망 파일 처리"""
        self.progress.emit(10, "Shapefile 읽는 중...")
        
        if os.path.exists(self.file_path):
            gdf = gpd.read_file(self.file_path)
        else:
            data_path = os.path.join('data', os.path.basename(self.file_path))
            if os.path.exists(data_path):
                gdf = gpd.read_file(data_path)
            else:
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.file_path}")
        
        self.progress.emit(40, "스켈레톤 추출 중...")
        start_time = time.time()
        skeleton, intersections = self.skeleton_extractor.process_shapefile(self.file_path)
        processing_time = time.time() - start_time
        
        self.progress.emit(85, "기본 휴리스틱 검출 중...")
        detected = {
            'intersection': intersections,
            'curve': [],
            'endpoint': []
        }
        
        self.progress.emit(90, "결과 정리 중...")
        result = {
            'success': True,
            'mode': 'road',
            'gdf': gdf,
            'skeleton': skeleton,
            'intersections': detected['intersection'],
            'curves': detected['curve'],
            'endpoints': detected['endpoint'],
            'processing_time': processing_time
        }
        
        self.progress.emit(100, "완료!")
        self.finished.emit(result)


class Process1LabelingTool(QMainWindow):
    """프로세스 1 - 라벨링 도구 메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.processing_thread = None
        self.file_mode = 'road'
        self.target_crs = 'EPSG:5186'
        
        # 파일 네비게이션 관리
        self.file_list = []
        self.current_file_index = 0
        
        # 멀티폴리곤 관리
        self.current_polygon_data = None
        self.current_polygon_index = 0
        
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("도로망 라벨링 도구 - Process 1 (파일 네비게이션)")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 왼쪽 패널
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # 오른쪽 캔버스
        self.canvas_widget = CanvasWidget()
        splitter.addWidget(self.canvas_widget)
        
        self.collector = DQNDataCollector()
        self.collector.connect_to_canvas(self.canvas_widget.canvas)
    
        splitter.setSizes([400, 1000])
        
        # 상태바
        self.statusBar().showMessage("준비")
        
    def create_left_panel(self):
        """왼쪽 컨트롤 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 파일 모드 선택
        mode_group = QGroupBox("파일 모드")
        mode_layout = QVBoxLayout()
        
        self.mode_button_group = QButtonGroup()
        
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
        
        # 좌표계 선택
        self.crs_group = QGroupBox("좌표계 선택")
        crs_layout = QVBoxLayout()
        
        self.crs_button_group = QButtonGroup()
        
        self.crs_5186_radio = QRadioButton("EPSG:5186 (중부원점)")
        self.crs_5186_radio.setChecked(True)
        self.crs_button_group.addButton(self.crs_5186_radio)
        crs_layout.addWidget(self.crs_5186_radio)
        
        self.crs_5187_radio = QRadioButton("EPSG:5187 (동부원점)")
        self.crs_button_group.addButton(self.crs_5187_radio)
        crs_layout.addWidget(self.crs_5187_radio)
        
        self.crs_group.setLayout(crs_layout)
        self.crs_group.setVisible(False)
        layout.addWidget(self.crs_group)
        
        # 파일 선택 그룹
        file_group = QGroupBox("1. 파일/폴더 선택")
        file_layout = QVBoxLayout()
        
        # 단일 파일 선택
        file_btn_layout = QHBoxLayout()
        select_btn = QPushButton("파일 선택")
        select_btn.clicked.connect(self.select_file)
        file_btn_layout.addWidget(select_btn)
        
        # 폴더 선택 (네비게이션)
        folder_btn = QPushButton("폴더 선택")
        folder_btn.clicked.connect(self.select_folder)
        folder_btn.setStyleSheet("QPushButton {background-color: #2196F3; color: white; font-weight: bold;}")
        file_btn_layout.addWidget(folder_btn)
        
        file_layout.addLayout(file_btn_layout)
        
        self.file_label = QLabel("파일: 선택 안됨")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        # 파일 네비게이션
        self.file_nav_widget = QWidget()
        nav_layout = QVBoxLayout()
        nav_layout.setContentsMargins(0, 0, 0, 0)
        
        # 파일 정보
        self.file_info_label = QLabel("")
        nav_layout.addWidget(self.file_info_label)
        
        # 파일 네비게이션 버튼
        file_nav_buttons = QHBoxLayout()
        
        self.prev_file_btn = QPushButton("◀ 이전 파일")
        self.prev_file_btn.clicked.connect(self.prev_file)
        self.prev_file_btn.setEnabled(False)
        file_nav_buttons.addWidget(self.prev_file_btn)
        
        self.next_file_btn = QPushButton("다음 파일 ▶")
        self.next_file_btn.clicked.connect(self.next_file)
        self.next_file_btn.setEnabled(False)
        file_nav_buttons.addWidget(self.next_file_btn)
        
        nav_layout.addLayout(file_nav_buttons)
        
        self.file_nav_widget.setLayout(nav_layout)
        self.file_nav_widget.setVisible(False)
        file_layout.addWidget(self.file_nav_widget)
        
        # 멀티폴리곤 네비게이션
        self.polygon_nav_widget = QWidget()
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(0, 0, 0, 0)
        
        self.polygon_info_label = QLabel("")
        nav_layout.addWidget(self.polygon_info_label)
        
        self.prev_polygon_btn = QPushButton("이전")
        self.prev_polygon_btn.clicked.connect(self.prev_polygon)
        self.prev_polygon_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_polygon_btn)
        
        self.next_polygon_btn = QPushButton("다음")
        self.next_polygon_btn.clicked.connect(self.next_polygon)
        self.next_polygon_btn.setEnabled(False)
        nav_layout.addWidget(self.next_polygon_btn)
        
        self.polygon_nav_widget.setLayout(nav_layout)
        self.polygon_nav_widget.setVisible(False)
        file_layout.addWidget(self.polygon_nav_widget)
        
        # 처리 버튼
        process_btn = QPushButton("처리 시작")
        process_btn.clicked.connect(self.process_file)
        process_btn.setStyleSheet("QPushButton {background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;}")
        file_layout.addWidget(process_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 통계 그룹
        stats_group = QGroupBox("2. 라벨링 통계")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        stats_layout.addWidget(self.stats_text)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # 수동 편집 그룹
        edit_group = QGroupBox("3. 수동 편집")
        edit_layout = QVBoxLayout()
        
        edit_info = QLabel(
            "• 좌클릭: 커브 수정 (필요시)\n"
            "• 우클릭: 끝점 추가 (도로의 끝)\n"
            "• Shift+클릭: 제거\n"
            "• D: 가장 가까운 점 삭제\n"
            "• Space: 화면 맞춤\n"
            "\n※ 교차점과 커브는 자동 검출됩니다"
        )
        edit_info.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; }")
        edit_layout.addWidget(edit_info)
        
        edit_group.setLayout(edit_layout)
        layout.addWidget(edit_group)

        # DQN 예측 그룹
        dqn_group = QGroupBox("4. DQN AI 예측")
        dqn_layout = QVBoxLayout()
        
        dqn_predict_btn = QPushButton("DQN 자동 예측")
        dqn_predict_btn.clicked.connect(self.run_dqn_prediction)
        dqn_predict_btn.setStyleSheet("QPushButton {background-color: #9C27B0; color: white; font-weight: bold; padding: 8px;}")
        dqn_layout.addWidget(dqn_predict_btn)
        
        dqn_info = QLabel("학습된 DQN 모델로 포인트 예측\n※ 먼저 프로세스 2에서 학습 필요")
        dqn_info.setStyleSheet("QLabel {background-color: #f8f8f8; padding: 8px; font-size: 10px;}")
        dqn_layout.addWidget(dqn_info)
        
        dqn_group.setLayout(dqn_layout)
        layout.addWidget(dqn_group)

        # 저장 그룹
        save_group = QGroupBox("5. 저장")
        save_layout = QVBoxLayout()
        
        save_btn = QPushButton("세션 저장")
        save_btn.clicked.connect(self.save_session)
        save_layout.addWidget(save_btn)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)

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
    
    def select_file(self):
        """단일 파일 선택"""
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
            self.file_label.setText(f"단일 파일: {Path(file_path).name}")
            self.canvas_widget.clear_all()
            self.update_stats()
            
            # 파일 네비게이션 모드 해제
            self.file_nav_widget.setVisible(False)
            self.file_list = []
            self.current_file_index = 0
    
    def select_folder(self):
        """폴더 선택 (파일 네비게이션)"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Shapefile 폴더 선택", ""
        )
        
        if folder_path:
            # 폴더 내 .shp 파일 스캔
            shp_files = sorted(list(Path(folder_path).glob("*.shp")))
            
            if not shp_files:
                QMessageBox.warning(self, "경고", "선택된 폴더에 Shapefile이 없습니다.")
                return
            
            # 파일 리스트 설정
            self.file_list = [str(f) for f in shp_files]
            self.current_file_index = 0
            
            # 첫 번째 파일 로드
            self.load_file_at_index(0)
            
            # 파일 네비게이션 UI 활성화
            self.file_nav_widget.setVisible(True)
            self.update_file_navigation()
            
            self.canvas_widget.clear_all()
    
    def load_file_at_index(self, index):
        """특정 인덱스의 파일 로드"""
        if 0 <= index < len(self.file_list):
            self.current_file = self.file_list[index]
            filename = Path(self.current_file).name
            self.file_label.setText(f"파일: {filename}")
            
            # 폴리곤 데이터 초기화
            self.current_polygon_data = None
            self.current_polygon_index = 0
            self.polygon_nav_widget.setVisible(False)
            
            self.canvas_widget.clear_all()
            self.update_stats()
    
    def update_file_navigation(self):
        """파일 네비게이션 UI 업데이트"""
        if not self.file_list:
            return
        
        current = self.current_file_index + 1
        total = len(self.file_list)
        
        self.file_info_label.setText(f"파일 {current}/{total}")
        self.prev_file_btn.setEnabled(current > 1)
        self.next_file_btn.setEnabled(current < total)
    
    def prev_file(self):
        """이전 파일로 이동"""
        if self.current_file_index > 0:
            # 현재 작업 저장 (선택사항)
            # self.save_current_work()
            
            self.current_file_index -= 1
            self.load_file_at_index(self.current_file_index)
            self.update_file_navigation()
    
    def next_file(self):
        """다음 파일로 이동"""
        if self.current_file_index < len(self.file_list) - 1:
            # 현재 작업 저장 (선택사항)
            # self.save_current_work()
            
            self.current_file_index += 1
            self.load_file_at_index(self.current_file_index)
            self.update_file_navigation()
    
    def process_file(self):
        """단일 파일 처리"""
        if not self.current_file:
            QMessageBox.warning(self, "경고", "파일을 먼저 선택하세요.")
            return
        
        # 진행 다이얼로그
        self.progress_dialog = QProgressDialog("처리 중...", "취소", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # 처리 스레드 시작
        self.processing_thread = ProcessingThread(
            self.current_file, 
            self.file_mode,
            self.get_target_crs() if self.file_mode == 'district' else 'EPSG:5186'
        )
        self.processing_thread.progress.connect(self.on_progress)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()
    
    def on_progress(self, value, message):
        """진행 상황 업데이트"""
        self.progress_dialog.setValue(value)
        self.progress_dialog.setLabelText(message)
        self.statusBar().showMessage(message)
    
    def on_processing_finished(self, result):
        """처리 완료"""
        self.progress_dialog.close()
        
        if result.get('need_manual_selection'):
            # 수동 폴더 선택 필요
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
                self.process_with_manual_road(result['polygons'], folder)
            return
        
        if result['success']:
            if result['mode'] == 'district':
                # 지구계 모드 처리
                self.current_polygon_data = result['district_results']
                self.load_polygon_result(result)
                
                # 멀티폴리곤 네비게이션 설정
                total_polygons = self.current_polygon_data['total_polygons']
                if total_polygons > 1:
                    self.polygon_nav_widget.setVisible(True)
                    self.update_polygon_navigation()
            else:
                # 기존 도로망 모드 처리
                self.load_road_result(result)
    
    def process_with_manual_road(self, polygons, road_folder):
        """수동 선택한 도로망으로 처리"""
        try:
            clipper = DistrictRoadClipper()
            
            # 첫 번째 폴리곤에 대해 클리핑
            if polygons:
                first_polygon = polygons[0]
                clipped = clipper.clip_with_manual_road(
                    first_polygon['geometry'],
                    road_folder,
                    self.get_target_crs()
                )
                
                if clipped is not None and not clipped.empty:
                    # 임시 디렉토리를 사용하여 스켈레톤 추출
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = os.path.join(temp_dir, "temp_road.shp")
                        
                        clipped.to_file(temp_path)
                        skeleton_extractor = SkeletonExtractor()
                        skeleton, intersections = skeleton_extractor.process_shapefile(temp_path)
                    
                    # 결과 로드
                    result = {
                        'success': True,
                        'road_gdf': clipped,
                        'skeleton': skeleton,
                        'intersections': intersections,
                        'curves': [],
                        'endpoints': []
                    }
                    self.load_road_result(result)
                else:
                    QMessageBox.warning(self, "경고", "도로망 클리핑 결과가 없습니다.")
                    
        except Exception as e:
            QMessageBox.critical(self, "오류", f"수동 도로망 처리 실패:\n{str(e)}")
    
    def load_polygon_result(self, result):
        """지구계 처리 결과 로드"""
        # 현재 폴리곤의 지구계 경계 표시
        if 'district_polygon_gdf' in result:
            self.canvas_widget.set_background_data(result['district_polygon_gdf'])
        
        # 도로망 결과 로드
        self.load_road_result(result)
    
    def load_road_result(self, result):
        """도로망 처리 결과 로드"""
        self.canvas_widget.current_file = self.current_file
        
        # 사용자 행동 추적 시작
        self.canvas_widget.canvas.action_tracker.start_session(self.current_file)
        
        # 도로 데이터 설정
        if 'road_gdf' in result:
            self.canvas_widget.set_road_data(result['road_gdf'])
        elif 'gdf' in result:
            self.canvas_widget.set_road_data(result['gdf'])
        
        self.canvas_widget.skeleton = result['skeleton']
        self.canvas_widget.processing_time = result.get('processing_time', 0)
        
        # 포인트 설정
        self.canvas_widget.canvas.points['intersection'] = [
            (float(x), float(y)) for x, y in result['intersections']
        ]
        self.canvas_widget.canvas.points['curve'] = [
            (float(x), float(y)) for x, y in result.get('curves', [])
        ]
        self.canvas_widget.canvas.points['endpoint'] = [
            (float(x), float(y)) for x, y in result.get('endpoints', [])
        ]
        
        # 디스플레이 업데이트
        self.canvas_widget.update_display()
        
        # 통계 업데이트
        self.update_stats()
        
        self.statusBar().showMessage(
            f"처리 완료 - 처리시간: {result.get('processing_time', 0):.2f}초"
        )
        
        # DQN 데이터 수집기 세션 시작
        if hasattr(self, 'collector') and self.collector:
            skeleton_data = {
                'skeleton': result.get('skeleton', []),
                'transform': None
            }
            detected_points = {
                'intersection': result.get('intersections', []),
                'curve': result.get('curves', []),
                'endpoint': result.get('endpoints', [])
            }
            heuristic_results = {
                'intersection': result.get('intersections', []),
                'curve': result.get('curves', []),
                'endpoint': result.get('endpoints', [])
            }
            
            self.collector.start_session(
                self.current_file,
                skeleton_data=skeleton_data,
                detected_points=detected_points,
                heuristic_results=heuristic_results
            )
    
    def update_polygon_navigation(self):
        """멀티폴리곤 네비게이션 업데이트"""
        if not self.current_polygon_data:
            return
        
        total = self.current_polygon_data['total_polygons']
        current = self.current_polygon_index + 1
        
        self.polygon_info_label.setText(f"폴리곤 {current}/{total}")
        self.prev_polygon_btn.setEnabled(current > 1)
        self.next_polygon_btn.setEnabled(current < total)
        
        # 캔버스에도 정보 전달
        if hasattr(self.canvas_widget, 'set_polygon_info'):
            self.canvas_widget.set_polygon_info(current, total)
    
    def prev_polygon(self):
        """이전 폴리곤으로 이동"""
        if self.current_polygon_index > 0:
            # 현재 작업 저장
            self.save_current_polygon_session()
            
            self.current_polygon_index -= 1
            self.load_polygon_at_index(self.current_polygon_index)
            self.update_polygon_navigation()
    
    def next_polygon(self):
        """다음 폴리곤으로 이동"""
        total = self.current_polygon_data['total_polygons']
        if self.current_polygon_index < total - 1:
            # 현재 작업 저장
            self.save_current_polygon_session()
            
            self.current_polygon_index += 1
            self.load_polygon_at_index(self.current_polygon_index)
            self.update_polygon_navigation()
    
    def load_polygon_at_index(self, index):
        """특정 인덱스의 폴리곤 로드"""
        if not self.current_polygon_data or not self.current_polygon_data['polygons']:
            return
        
        # 현재 인덱스의 폴리곤 가져오기
        current_polygon = self.current_polygon_data['polygons'][index]
        
        # 지구계 경계 업데이트
        if 'geometry' in current_polygon:
            self.canvas_widget.set_background_data(current_polygon['geometry'])
        
        # 해당 폴리곤의 도로망이 이미 클리핑되어 있으면 표시
        if 'clipped_road' in current_polygon and current_polygon['clipped_road'] is not None:
            # 이미 클리핑된 도로망이 있으면 사용
            road_gdf = current_polygon['clipped_road']
            
            # 임시 디렉토리로 스켈레톤 추출
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, "temp_road.shp")
                road_gdf.to_file(temp_path)
                
                skeleton_extractor = SkeletonExtractor()
                skeleton, intersections = skeleton_extractor.process_shapefile(temp_path)
            
            # 결과 로드
            result = {
                'success': True,
                'road_gdf': road_gdf,
                'skeleton': skeleton,
                'intersections': intersections,
                'curves': [],
                'endpoints': [],
                'district_polygon_gdf': current_polygon['geometry'] 
            }
            
            # 화면 업데이트
            self.load_road_result(result)
            
            # 캔버스 초기화 (기존 포인트 제거)
            self.canvas_widget.canvas.points = {'intersection': [], 'curve': [], 'endpoint': []}
            self.canvas_widget.canvas.ai_points = {'intersection': [], 'curve': [], 'endpoint': [], 'delete': []}
            
            # 교차점만 표시
            self.canvas_widget.canvas.points['intersection'] = [
                (float(x), float(y)) for x, y in intersections
            ]
            
            self.canvas_widget.update_display()
            self.update_stats()
        else:
            # 클리핑이 안 되어 있으면 새로 처리 필요
            QMessageBox.information(
                self, "알림",
                f"{index + 1}번째 폴리곤은 도로망 클리핑이 필요합니다.\n"
                "처리 시작 버튼을 다시 눌러주세요."
            )
    
    def save_current_polygon_session(self):
        """현재 폴리곤 작업 저장"""
        if self.current_polygon_data:
            base_name = Path(self.current_file).stem
            session_name = get_polygon_session_name(
                base_name,
                self.current_polygon_index + 1,
                self.current_polygon_data['total_polygons']
            )
            # 저장 로직...
    
    def on_processing_error(self, error_msg):
        """처리 오류"""
        self.progress_dialog.close()
        QMessageBox.critical(self, "오류", f"처리 중 오류 발생:\n{error_msg}")
        self.statusBar().showMessage("오류 발생")
    
    def update_stats(self):
        """통계 업데이트"""
        if not hasattr(self.canvas_widget, 'canvas'):
            self.stats_text.clear()
            return
        
        points = self.canvas_widget.canvas.points
        
        mode_text = "지구계 자동 클리핑" if self.file_mode == 'district' else "도로망 직접 선택"
        
        stats_text = f"""=== 라벨링 통계 ({mode_text}) ===
교차점: {len(points.get('intersection', []))}개
커브: {len(points.get('curve', []))}개
끝점: {len(points.get('endpoint', []))}개
━━━━━━━━━━━━━━━━━━━━━
전체: {sum(len(v) for v in points.values())}개"""
        
        # 파일 네비게이션 정보 추가
        if self.file_list:
            stats_text += f"\n\n파일: {self.current_file_index + 1}/{len(self.file_list)}"
        
        if self.file_mode == 'district' and self.current_polygon_data:
            stats_text += f"\n폴리곤: {self.current_polygon_index + 1}/{self.current_polygon_data['total_polygons']}"
            if 'sido' in self.current_polygon_data:
                stats_text += f"\n지역: {self.current_polygon_data['sido']}/{self.current_polygon_data.get('sigungu', '?')}"
        
        self.stats_text.setText(stats_text)

    def run_dqn_prediction(self):
        """DQN 예측 실행"""
        if not hasattr(self.canvas_widget, 'canvas') or not self.canvas_widget.skeleton:
            QMessageBox.warning(self, "경고", "먼저 파일을 처리하세요.")
            return
        
        try:
            self.canvas_widget.canvas.run_dqn_prediction()
            self.update_stats()
            QMessageBox.information(self, "완료", "DQN 예측이 완료되었습니다!")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"DQN 예측 실패:\n{str(e)}")

    def save_session(self):
        """세션 저장"""
        if not self.current_file:
            QMessageBox.warning(self, "경고", "저장할 데이터가 없습니다.")
            return
        
        try:
            labels = self.canvas_widget.canvas.points
            skeleton = self.canvas_widget.skeleton
            
            metadata = {
                'processing_time': getattr(self.canvas_widget, 'processing_time', 0),
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
            
            user_actions = self.canvas_widget.canvas.action_tracker.get_session_actions()
            
            # DQN 샘플 수집
            dqn_samples = []
            if hasattr(self.collector, 'current_session') and self.collector.current_session:
                dqn_samples = self.collector.current_session.get('samples', [])
            
            session_path = save_session(
                self.current_file, 
                labels, 
                skeleton, 
                metadata, 
                user_actions,
                polygon_info=polygon_info,
                dqn_samples=dqn_samples
            )
            
            if session_path:
                QMessageBox.information(self, "성공", f"세션이 저장되었습니다.\n{session_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "오류", f"저장 실패:\n{str(e)}")


def main():
    """메인 실행"""
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = Process1LabelingTool()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()