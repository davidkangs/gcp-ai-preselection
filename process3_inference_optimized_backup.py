"""프로세스 3: AI 예측 + 인간 수정 + 재학습 (지구계 지원)"""

import sys
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
import geopandas as gpd
from shapely.geometry import Point
from collections import defaultdict
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog, QMessageBox,
    QGroupBox, QCheckBox, QProgressDialog, QTableWidget, QTableWidgetItem,
    QTextEdit, QSplitter, QComboBox, QSpinBox, QRadioButton, QButtonGroup,
    QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont

sys.path.append(str(Path(__file__).parent.parent))
from src.core.skeleton_extractor import SkeletonExtractor
from src.core.district_road_clipper import DistrictRoadClipper
# 연결성 검사기 제거됨 - 3-액션 시스템에서는 불필요
from src.ui.canvas_widget import CanvasWidget
from src.learning.dqn_model import create_agent
from src.utils import save_session, load_session, get_polygon_session_name
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter1d

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 통합 특징 추출기 import (무한루프 방지를 위해 조건부 import)
try:
    from src.core.unified_feature_extractor import get_feature_extractor, initialize_global_extractor
    # 통합 특징 추출기 비활성화 - 성능 문제
    UNIFIED_EXTRACTOR_AVAILABLE = False
    logger.info("통합 특징 추출기 비활성화됨 (성능 문제)")
except ImportError:
    logger.warning("통합 특징 추출기를 사용할 수 없습니다. 기본 구현을 사용합니다.")
    UNIFIED_EXTRACTOR_AVAILABLE = False


class PredictionWorker(QThread):
    progress = pyqtSignal(int, str)
    prediction_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, file_path, model_path, file_mode='road'):
        super().__init__()
        self.file_path = file_path
        self.model_path = model_path
        self.file_mode = file_mode
        self.skeleton_extractor = SkeletonExtractor()
        self.temp_path = None
        self.feature_extractor = None

    def _create_dqn_state_vector(self, point, skeleton, idx, heuristic_results=None):
        """안정적인 기본 특징벡터 생성 (무한루프 방지)"""
        x, y = float(point[0]), float(point[1])
        
        # 통합 특징 추출기 사용 (안전 모드로 수정됨)
        if UNIFIED_EXTRACTOR_AVAILABLE:
            try:
                extractor = get_feature_extractor()
                if extractor:
                    # 초기화되지 않은 경우 동적 초기화
                    if extractor.skeleton_data is None and skeleton is not None:
                        # skeleton을 리스트로 변환 (NumPy 배열 문제 해결)
                        skeleton_list = skeleton.tolist() if hasattr(skeleton, 'tolist') else list(skeleton)
                        skeleton_data = {
                            'skeleton': skeleton_list,
                            'transform': {'bounds': [0, 0, 10000, 10000]}
                        }
                        extractor.initialize(skeleton_data)
                        logger.info("🔄 특징 추출기 동적 초기화 완료")
                    
                    features = extractor.extract_features((x, y), idx, heuristic_results)
                    if features and len(features) == 20:
                        return features
            except Exception as e:
                logger.warning(f"통합 특징 추출기 실패, 기본 구현 사용: {e}")
        
        # 폴백: 기본 구현
        features = []
        features.extend([x, y])
        
        if idx > 0 and len(skeleton[idx-1]) >= 2:
            prev_x, prev_y = skeleton[idx-1][0], skeleton[idx-1][1]
            dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            angle = np.arctan2(y - prev_y, x - prev_x)
            features.extend([dist, angle])
        else:
            features.extend([0.0, 0.0])
        
        if idx < len(skeleton) - 1 and len(skeleton[idx+1]) >= 2:
            next_x, next_y = skeleton[idx+1][0], skeleton[idx+1][1]
            dist = np.sqrt((next_x - x)**2 + (next_y - y)**2)
            angle = np.arctan2(next_y - y, next_x - x)
            features.extend([dist, angle])
        else:
            features.extend([0.0, 0.0])
        
        density = sum(1 for p in skeleton if len(p) >= 2 and np.sqrt((x-p[0])**2 + (y-p[1])**2) <= 50)
        density = density / len(skeleton) if skeleton is not None and len(skeleton) > 0 else 0.0
        features.append(density)
        
        if idx > 0 and idx < len(skeleton) - 1:
            try:
                if len(skeleton[idx-1]) >= 2 and len(skeleton[idx+1]) >= 2:
                    p1 = np.array(skeleton[idx-1][:2])
                    p2 = np.array([x, y])
                    p3 = np.array(skeleton[idx+1][:2])
                    v1 = p2 - p1
                    v2 = p3 - p2
                    angle1 = np.arctan2(v1[1], v1[0])
                    angle2 = np.arctan2(v2[1], v2[0])
                    curvature = abs(angle2 - angle1)
                    if curvature > np.pi:
                        curvature = 2 * np.pi - curvature
                    features.append(curvature)
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 나머지 특징들을 0으로 패딩 (20차원 맞추기)
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]

    def run(self):
        try:
            self.progress.emit(20, "스켈레톤 추출 중...")
            
            # 지구계 모드인 경우 임시 파일 경로 사용
            if hasattr(self, 'temp_path') and self.temp_path:
                result = self.skeleton_extractor.extract_from_shapefile(self.temp_path)
            else:
                result = self.skeleton_extractor.extract_from_shapefile(self.file_path)
            
            if not result or 'skeleton' not in result:
                self.error_occurred.emit("스켈레톤 추출 실패")
                return
            
            skeleton = result['skeleton']
            intersections = result.get('intersections', [])
            
            # 통합 특징 추출기 초기화 비활성화 (무한루프 방지)
            # if UNIFIED_EXTRACTOR_AVAILABLE:
            #     try:
            #         skeleton_data = {
            #             'skeleton': skeleton if isinstance(skeleton, list) else skeleton.tolist(),
            #             'transform': result.get('transform')
            #         }
            #         self.feature_extractor = initialize_global_extractor(skeleton_data)
            #         logger.info("통합 특징 추출기 초기화 완료")
            #     except Exception as e:
            #         logger.warning(f"통합 특징 추출기 초기화 실패: {e}")
            #         self.feature_extractor = None
            logger.info("기본 특징 추출기 사용 (안정성 우선)")
                
            if isinstance(skeleton, np.ndarray):
                if skeleton.size == 0:
                    self.error_occurred.emit("스켈레톤이 비어있습니다")
                    return
                skeleton_array = skeleton
            elif isinstance(skeleton, list):
                if len(skeleton) == 0:
                    self.error_occurred.emit("스켈레톤이 비어있습니다")
                    return
                skeleton_array = np.array(skeleton)
            else:
                try:
                    skeleton = list(skeleton)
                    skeleton_array = np.array(skeleton)
                except:
                    self.error_occurred.emit("스켈레톤 형식이 올바르지 않습니다")
                    return
            
            self.progress.emit(40, "AI 모델 로드 중...")
            agent = create_agent()
            agent.load(self.model_path)
            
            self.progress.emit(60, "특징 추출 중...")
            features = []
            
            for i, point in enumerate(skeleton_array):
                feat = self._create_dqn_state_vector(point, skeleton_array, i)
                features.append(feat)
            
            self.progress.emit(80, "AI 예측 중 (보수적 필터링)...")
            features_array = np.array(features)
            
            # AI 예측 + 삭제 강화
            conservative_predictions = []
            confidence_threshold = 0.4
            
            if hasattr(agent, 'q_network'):
                # DQN 기반 예측
                with torch.no_grad():
                    device = next(agent.q_network.parameters()).device
                    input_tensor = torch.FloatTensor(features_array).to(device)
                    q_values_batch = agent.q_network(input_tensor)
                
                # AI 예측 + 삭제 강화 (점이 너무 가까운 경우 삭제 유도)
                for i, q_values in enumerate(q_values_batch):
                    q_vals = q_values.cpu().numpy()
                    action = np.argmax(q_vals)
                    max_q = np.max(q_vals)
                    confidence = max_q - np.mean(q_vals)
                    
                    # 삭제 강화: 주변에 너무 가까운 점이 있으면 삭제 유도
                    current_point = skeleton_array[i]
                    should_delete = False
                    
                    # 10m 이내 다른 점이 있는지 확인
                    for j, other_point in enumerate(skeleton_array):
                        if i != j:
                            # 좌표를 안전하게 float로 변환
                            try:
                                x1, y1 = float(current_point[0]), float(current_point[1])
                                x2, y2 = float(other_point[0]), float(other_point[1])
                                dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                                if dist < 10.0:  # 10m 이내
                                    should_delete = True
                                    break
                            except (IndexError, ValueError, TypeError):
                                continue  # 좌표 변환 실패 시 무시
                    
                    # 최종 액션 결정
                    if should_delete and np.random.random() < 0.3:  # 30% 확률로 삭제 강화
                        conservative_predictions.append(4)  # 삭제
                    elif confidence > confidence_threshold:
                        conservative_predictions.append(action)
                    else:
                        conservative_predictions.append(0)  # 불확실한 경우 제외
            else:
                # 기본 예측
                conservative_predictions = agent.predict(features_array)
            
            ai_points = {
                'intersection': [tuple(pt) for pt in intersections] if intersections is not None else [],
                'curve': [],
                'endpoint': [],
                'delete': []
            }
            
            # 신뢰도 기반 필터링 (개수 제한 제거)
            confidence_data = []  # 신뢰도 정보 저장
            
            for i, pred in enumerate(conservative_predictions):
                point = tuple(skeleton_array[i])
                
                # Q값에서 신뢰도 계산 (이미 위에서 계산됨)
                if hasattr(agent, 'q_network') and i < len(q_values_batch):
                    q_vals = q_values_batch[i].cpu().numpy()
                    max_q = np.max(q_vals)
                    second_max_q = np.partition(q_vals, -2)[-2] if len(q_vals) > 1 else 0
                    confidence = max_q - second_max_q  # 1등과 2등의 차이
                else:
                    confidence = 0.5  # 기본값
                
                # 신뢰도 정보 저장
                confidence_data.append({
                    'point': point,
                    'action': pred,
                    'confidence': confidence
                })
                
                # 예측 액션에 따라 분류 (제한 없음)
                if pred == 1:
                    ai_points['intersection'].append(point)
                elif pred == 2:
                    ai_points['curve'].append(point)
                elif pred == 3:
                    ai_points['endpoint'].append(point)
                elif pred == 4:
                    ai_points['delete'].append(point)
            
            logger.info(f"AI예측: int={len(ai_points['intersection'])}, "
                       f"curve={len(ai_points['curve'])}, "
                       f"end={len(ai_points['endpoint'])}, "
                       f"del={len(ai_points['delete'])}")
            
            result = {
                'success': True,
                'skeleton': skeleton,
                'ai_points': ai_points,
                'predictions': conservative_predictions,
                'confidence_data': confidence_data  # 신뢰도 정보 추가
            }
            
            self.progress.emit(100, "예측 완료!")
            self.prediction_completed.emit(result)
            
        except Exception as e:
            logger.error(f"예측 오류: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))


class InferenceTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.model_path = None
        self.original_predictions = None
        self.modified_sessions = []
        
        # 지구계 관련 추가
        self.file_mode = 'road'  # 'district' or 'road'
        self.selected_crs = 'EPSG:5186'
        self.district_clipper = DistrictRoadClipper()
        self.current_polygon_data = None
        self.current_polygon_index = 0
        self.ai_confidence_threshold = 0.7  # 초기값
        
        # Excel 기준점 추가
        self.excel_points = []
        
        self.init_ui()
        self.check_models()

    def init_ui(self):
        self.setWindowTitle("도로망 AI 예측 및 수정 - 프로세스 3 (지구계 지원)")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        self.canvas_widget = CanvasWidget()
        self.canvas_widget.canvas.show_ai_predictions = True
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
        
        # 신뢰도 조정
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("신뢰도:"))
        self.confidence_slider = QSlider()
        self.confidence_slider.setOrientation(Qt.Horizontal)
        self.confidence_slider.setRange(0, 95)  # 0.0 ~ 0.95
        self.confidence_slider.setValue(70)  # 0.7
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        confidence_layout.addWidget(self.confidence_slider)
        self.confidence_label = QLabel("0.70")
        self.confidence_label.setMinimumWidth(30)
        confidence_layout.addWidget(self.confidence_label)
        result_layout.addLayout(confidence_layout)
        
        self.result_label = QLabel("예측 전...")
        self.result_label.setStyleSheet("QLabel {padding: 10px; background-color: #f0f0f0; border-radius: 5px;}")
        result_layout.addWidget(self.result_label)
        
        # 거리 정보 표시
        self.distance_label = QLabel("거리 정보: -")
        self.distance_label.setStyleSheet("QLabel {padding: 5px; background-color: #e8f4fd; border-radius: 3px; font-size: 11px;}")
        result_layout.addWidget(self.distance_label)
        
        # Excel 업로드 버튼 추가
        excel_btn = QPushButton("📊 실제 기준점 Excel 업로드")
        excel_btn.clicked.connect(self.upload_excel)
        excel_btn.setStyleSheet("QPushButton {background-color: #009688; color: white; font-weight: bold; padding: 8px;}")
        result_layout.addWidget(excel_btn)
        
        # 점 개수 비교 테이블 (크게 확대)
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
        self.point_count_table.setMinimumHeight(200)  # 높이 크게 증가
        self.point_count_table.setMaximumHeight(250)  # 최대 높이도 증가
        self.point_count_table.horizontalHeader().setStretchLastSection(True)
        # 폰트 크기 증가
        font = QFont()
        font.setPointSize(14)  # 폰트 크기 증가
        font.setBold(True)     # 굵게
        self.point_count_table.setFont(font)
        # 행 높이 증가
        for i in range(3):
            self.point_count_table.setRowHeight(i, 50)
        result_layout.addWidget(self.point_count_table)
        
        self.show_ai_checkbox = QCheckBox("AI 예측 표시")
        self.show_ai_checkbox.setChecked(True)
        self.show_ai_checkbox.toggled.connect(self.toggle_ai_predictions)
        result_layout.addWidget(self.show_ai_checkbox)
        
        # 버튼들을 2열로 배치
        button_grid_layout = QVBoxLayout()
        
        row1_layout = QHBoxLayout()
        accept_all_btn = QPushButton("모든 AI 예측 수락")
        accept_all_btn.clicked.connect(self.accept_all_predictions)
        row1_layout.addWidget(accept_all_btn)
        
        clear_user_btn = QPushButton("사용자 수정 초기화")
        clear_user_btn.clicked.connect(self.clear_user_modifications)
        row1_layout.addWidget(clear_user_btn)
        button_grid_layout.addLayout(row1_layout)
        
        # 중복점 제거 버튼 (별도 행)
        remove_duplicates_btn = QPushButton("🎯 중복점 정리 (5m)")
        remove_duplicates_btn.clicked.connect(self.remove_duplicate_points)
        remove_duplicates_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800; 
                color: white; 
                font-weight: bold; 
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        button_grid_layout.addWidget(remove_duplicates_btn)
        
        result_layout.addLayout(button_grid_layout)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # ===== 4. 배치 처리 =====
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
        """도로망 파일 완전 자동화 처리 (AI 분석 → 스마트 필터링 → 거리 계산)"""
        try:
            progress = QProgressDialog("자동화 파이프라인 실행 중...", "취소", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # 1단계: 스켈레톤 추출
            progress.setLabelText("1/5 단계: 스켈레톤 추출 중...")
            progress.setValue(10)
            
            skeleton_extractor = SkeletonExtractor()
            skeleton, intersections = skeleton_extractor.process_shapefile(file_path)
            
            # 2단계: 도로망 데이터 로드
            progress.setLabelText("2/5 단계: 도로망 데이터 로드 중...")
            progress.setValue(20)
            
            road_gdf = gpd.read_file(file_path)
            self.canvas_widget.set_road_data(road_gdf)
            
            # 기본 캔버스 설정
            self.canvas_widget.skeleton = skeleton
            self.canvas_widget.canvas.skeleton = skeleton
            self.canvas_widget.canvas.points = {
                'intersection': [(float(x), float(y)) for x, y in intersections],
                'curve': [],
                'endpoint': []
            }
            
            # 2-2단계: 휴리스틱 끝점 검출
            road_bounds = (
                skeleton_extractor.get_bounds() if hasattr(skeleton_extractor, 'get_bounds') 
                else None
            )
            endpoints = self.detect_heuristic_endpoints(skeleton, road_bounds)
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
            progress.setLabelText("3/5 단계: AI 예측 (삭제만) 실행 중...")
            progress.setValue(40)
            
            if self.model_path:
                ai_result = self.run_ai_prediction_auto(skeleton, file_path)
                if ai_result and ai_result['success']:
                    # AI는 삭제만 담당 (커브는 도로 경계선 기반 사용)
                    # self.canvas_widget.canvas.points['curve'].extend(ai_result['ai_points']['curve'])  # 제거!
                    
                    # 삭제 처리만 수행
                    deleted_points = ai_result['ai_points'].get('delete', [])
                    if deleted_points:
                        self.apply_deletions(deleted_points)
            
            # 4단계: 중복 점 필터링
            progress.setLabelText("4/6 단계: 중복 점 필터링 중...")
            progress.setValue(50)
            
            self.filter_overlapping_points()
            
            # 5단계: 최종 점 정리 (연결성 검사 제거됨)
            progress.setLabelText("5/6 단계: 최종 점 정리 중...")
            progress.setValue(65)
            
            logger.info("📍 3-액션 시스템: 휴리스틱(교차점,끝점) + 도로경계선(커브) + AI(삭제) 완료")
            
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
            
            self.statusBar().showMessage(f"자동화 파이프라인 완료 - 총 {total_points}개 점 검출")
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self, "오류", f"자동화 파이프라인 실패:\n{str(e)}")
            logger.error(f"자동화 파이프라인 오류: {e}")

    def process_district_file(self, district_file):
        """지구계 파일 처리"""
        try:
            progress = QProgressDialog("지구계 파일 처리 중...", "취소", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            progress.setLabelText("지구계 파일 읽는 중...")
            progress.setValue(10)
            
            results = self.district_clipper.process_district_file(
                district_file,
                target_crs=self.get_target_crs(),
                auto_find_road=True
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
        if not result['success'] or not result['polygons']:
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
                progress = QProgressDialog("지구계 자동화 파이프라인 실행 중...", "취소", 0, 100, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.show()
                
                # 1단계: 임시 파일로 스켈레톤 추출
                progress.setLabelText("1/5 단계: 스켈레톤 추출 중...")
                progress.setValue(10)
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = os.path.join(temp_dir, "temp_road.shp")
                    road_gdf.to_file(temp_path)
                    
                    skeleton_extractor = SkeletonExtractor()
                    skeleton, intersections = skeleton_extractor.process_shapefile(temp_path)
                
                    # 2단계: 기본 캔버스 설정
                    progress.setLabelText("2/5 단계: 기본 설정 중...")
                    progress.setValue(20)
                    
                self.canvas_widget.skeleton = skeleton
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
                progress.setLabelText("3/5 단계: AI 예측 (삭제만) 실행 중...")
                progress.setValue(40)
                
                if self.model_path:
                    ai_result = self.run_ai_prediction_auto(skeleton, temp_path)
                    if ai_result and ai_result['success']:
                        # AI는 삭제만 담당 (커브는 도로 경계선 기반 사용)
                        # self.canvas_widget.canvas.points['curve'].extend(ai_result['ai_points']['curve'])  # 제거!
                        
                        # 삭제 처리만 수행
                        deleted_points = ai_result['ai_points'].get('delete', [])
                        if deleted_points:
                            self.apply_deletions(deleted_points)
                
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
                
                # 통계 표시
                info_text = f"스켈레톤: {len(skeleton)}점\n"
                info_text += f"총 검출 점: {total_points}개\n"
                info_text += f"자동화 파이프라인 완료"
                
                self.result_label.setText(f"처리 완료:\n{info_text}")
                self.statusBar().showMessage(f"지구계 자동화 파이프라인 완료 - 총 {total_points}개 점 검출")
                
            except Exception as e:
                if 'progress' in locals():
                    progress.close()
                logger.error(f"지구계 자동화 파이프라인 오류: {e}")
                # 기본 모드로 폴백
                self.canvas_widget.skeleton = skeleton
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
        if self.current_polygon_data and self.canvas_widget.skeleton:
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

    def remove_duplicate_points(self):
        """5m 반경 내 중복점 제거 (우선순위: 끝점 > 교차점 > 커브점)"""
        if not self.canvas_widget.canvas.points:
            QMessageBox.warning(self, "경고", "정리할 포인트가 없습니다.")
            return
        
        # 모든 포인트를 우선순위와 함께 수집
        all_points_with_priority = []
        
        # 우선순위: 끝점(3) > 교차점(2) > 커브점(1)
        for x, y in self.canvas_widget.canvas.points.get('endpoint', []):
            all_points_with_priority.append((x, y, 3, 'endpoint'))
        
        for x, y in self.canvas_widget.canvas.points.get('intersection', []):
            all_points_with_priority.append((x, y, 2, 'intersection'))
        
        for x, y in self.canvas_widget.canvas.points.get('curve', []):
            all_points_with_priority.append((x, y, 1, 'curve'))
        
        # 중복 제거된 포인트 저장
        cleaned_points = {'intersection': [], 'curve': [], 'endpoint': []}
        processed_indices = set()
        
        # 우선순위가 높은 순으로 정렬
        all_points_with_priority.sort(key=lambda p: p[2], reverse=True)
        
        for i, (x1, y1, priority1, category1) in enumerate(all_points_with_priority):
            if i in processed_indices:
                continue
            
            # 현재 포인트 추가
            cleaned_points[category1].append((x1, y1))
            processed_indices.add(i)
            
            # 5m 반경 내의 다른 포인트들 제거
            for j, (x2, y2, priority2, category2) in enumerate(all_points_with_priority):
                if i != j and j not in processed_indices:
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < 5.0:  # 5m 반경
                        processed_indices.add(j)
        
        # 결과 통계
        original_counts = {
            'intersection': len(self.canvas_widget.canvas.points.get('intersection', [])),
            'curve': len(self.canvas_widget.canvas.points.get('curve', [])),
            'endpoint': len(self.canvas_widget.canvas.points.get('endpoint', []))
        }
        
        cleaned_counts = {
            'intersection': len(cleaned_points['intersection']),
            'curve': len(cleaned_points['curve']),
            'endpoint': len(cleaned_points['endpoint'])
        }
        
        removed_counts = {
            'intersection': original_counts['intersection'] - cleaned_counts['intersection'],
            'curve': original_counts['curve'] - cleaned_counts['curve'],
            'endpoint': original_counts['endpoint'] - cleaned_counts['endpoint']
        }
        
        # 캔버스 업데이트
        self.canvas_widget.canvas.points = cleaned_points
        self.canvas_widget.canvas.update_display()
        
        # 결과 메시지
        total_removed = sum(removed_counts.values())
        QMessageBox.information(
            self, "중복점 정리 완료",
            f"5m 반경 내 중복점 제거 (우선순위: 끝점>교차점>커브점)\n\n"
            f"제거된 포인트:\n"
            f"- 교차점: {removed_counts['intersection']}개\n"
            f"- 커브점: {removed_counts['curve']}개\n"
            f"- 끝점: {removed_counts['endpoint']}개\n"
            f"- 총 제거: {total_removed}개\n\n"
            f"남은 포인트:\n"
            f"- 교차점: {cleaned_counts['intersection']}개\n"
            f"- 커브점: {cleaned_counts['curve']}개\n"
            f"- 끝점: {cleaned_counts['endpoint']}개"
        )
        
        self.update_modification_stats()

    def _create_dqn_state_vector(self, point, skeleton, idx, heuristic_results=None):
        """안정적인 기본 특징벡터 생성 (무한루프 방지)"""
        x, y = float(point[0]), float(point[1])
        
        # 통합 특징 추출기 사용 (안전 모드로 수정됨)
        if UNIFIED_EXTRACTOR_AVAILABLE:
            try:
                extractor = get_feature_extractor()
                if extractor:
                    # 초기화되지 않은 경우 동적 초기화
                    if extractor.skeleton_data is None and skeleton is not None:
                        # skeleton을 리스트로 변환 (NumPy 배열 문제 해결)
                        skeleton_list = skeleton.tolist() if hasattr(skeleton, 'tolist') else list(skeleton)
                        skeleton_data = {
                            'skeleton': skeleton_list,
                            'transform': {'bounds': [0, 0, 10000, 10000]}
                        }
                        extractor.initialize(skeleton_data)
                        logger.info("🔄 특징 추출기 동적 초기화 완료")
                    
                    features = extractor.extract_features((x, y), idx, heuristic_results)
                    if features and len(features) == 20:
                        return features
            except Exception as e:
                logger.warning(f"통합 특징 추출기 실패, 기본 구현 사용: {e}")
        
        # 폴백: 기본 구현
        features = []
        features.extend([x, y])
        
        if idx > 0 and len(skeleton[idx-1]) >= 2:
            prev_x, prev_y = skeleton[idx-1][0], skeleton[idx-1][1]
            dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            angle = np.arctan2(y - prev_y, x - prev_x)
            features.extend([dist, angle])
        else:
            features.extend([0.0, 0.0])
        
        if idx < len(skeleton) - 1 and len(skeleton[idx+1]) >= 2:
            next_x, next_y = skeleton[idx+1][0], skeleton[idx+1][1]
            dist = np.sqrt((next_x - x)**2 + (next_y - y)**2)
            angle = np.arctan2(next_y - y, next_x - x)
            features.extend([dist, angle])
        else:
            features.extend([0.0, 0.0])
        
        density = sum(1 for p in skeleton if len(p) >= 2 and np.sqrt((x-p[0])**2 + (y-p[1])**2) <= 50)
        density = density / len(skeleton) if skeleton is not None and len(skeleton) > 0 else 0.0
        features.append(density)
        
        if idx > 0 and idx < len(skeleton) - 1:
            try:
                if len(skeleton[idx-1]) >= 2 and len(skeleton[idx+1]) >= 2:
                    p1 = np.array(skeleton[idx-1][:2])
                    p2 = np.array([x, y])
                    p3 = np.array(skeleton[idx+1][:2])
                    v1 = p2 - p1
                    v2 = p3 - p2
                    angle1 = np.arctan2(v1[1], v1[0])
                    angle2 = np.arctan2(v2[1], v2[0])
                    curvature = abs(angle2 - angle1)
                    if curvature > np.pi:
                        curvature = 2 * np.pi - curvature
                    features.append(curvature)
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 나머지 특징들을 0으로 패딩 (20차원 맞추기)
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]

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
                    
                    # 신뢰도 임계값 체크 (초기값 사용)
                    if confidence >= self.ai_confidence_threshold:
                        if action == 2:  # 커브
                            ai_points['curve'].append(point)
                        elif action == 4:  # 삭제
                            ai_points['delete'].append(point)
                        # 교차점과 끝점은 AI 분석이 담당
            
            logger.info(f"신뢰도 기반 AI 예측: 커브={len(ai_points['curve'])}, 삭제={len(ai_points['delete'])}")
            
            return {
                'success': True,
                'ai_points': ai_points,
                'confidence_data': confidence_data
            }
            
        except Exception as e:
            logger.error(f"AI 예측 오류: {e}")
            return None

    def filter_overlapping_points(self):
        """5m 이내 중복 점 필터링 (끝점 > 교차점 > 커브 우선순위)"""
        try:
            points = self.canvas_widget.canvas.points
            threshold = 5.0  # 5m
            
            all_points = []
            for category in ['intersection', 'curve', 'endpoint']:
                for point in points[category]:
                    all_points.append({
                        'point': point,
                        'category': category,
                        'priority': {'endpoint': 3, 'intersection': 2, 'curve': 1}[category]
                    })
            
            # 중복 그룹 찾기
            clusters = []
            used = set()
            
            for i, p1 in enumerate(all_points):
                if i in used:
                    continue
                    
                cluster = [p1]
                used.add(i)
                
                for j, p2 in enumerate(all_points[i+1:], i+1):
                    if j in used:
                        continue
                        
                    # 안전한 좌표 접근으로 거리 계산
                    try:
                        # p1과 p2의 point 좌표를 안전하게 추출
                        x1 = float(p1['point'][0]) if len(p1['point']) > 0 else 0.0
                        y1 = float(p1['point'][1]) if len(p1['point']) > 1 else 0.0
                        x2 = float(p2['point'][0]) if len(p2['point']) > 0 else 0.0
                        y2 = float(p2['point'][1]) if len(p2['point']) > 1 else 0.0
                        
                        dist = np.hypot(x1 - x2, y1 - y2)
                    except (IndexError, ValueError, TypeError):
                        dist = float('inf')  # 오류 시 매우 큰 거리로 설정
                    
                    if dist <= threshold:
                        cluster.append(p2)
                        used.add(j)
                
                clusters.append(cluster)
            
            # 각 클러스터에서 최고 우선순위만 유지
            filtered_points = {'intersection': [], 'curve': [], 'endpoint': []}
            
            for cluster in clusters:
                best = max(cluster, key=lambda x: x['priority'])
                filtered_points[best['category']].append(best['point'])
            
            # 필터링 결과 적용
            removed_count = sum(len(points[cat]) for cat in points) - sum(len(filtered_points[cat]) for cat in filtered_points)
            self.canvas_widget.canvas.points = filtered_points
            
            logger.info(f"중복 점 필터링 완료: {removed_count}개 점 제거")
            
        except Exception as e:
            logger.error(f"중복 점 필터링 오류: {e}")

    def calculate_and_display_distances(self):
        """네트워크 연결성 기반 점간 거리 계산"""
        try:
            import networkx as nx
            from collections import defaultdict
            
            # 모든 점 수집
            all_points = []
            point_to_idx = {}
            idx = 0
            
            for category in ['intersection', 'curve', 'endpoint']:
                for point in self.canvas_widget.canvas.points.get(category, []):
                    coord = (float(point[0]), float(point[1]))
                    all_points.append(coord)
                    point_to_idx[coord] = idx
                    idx += 1
            
            if len(all_points) < 2:
                self.distance_label.setText("거리 정보: 점이 부족합니다")
                return
            
            # 네트워크 그래프 생성
            G = nx.Graph()
            G.add_nodes_from(range(len(all_points)))
            
            # 가까운 점들 연결 (50m 이내)
            connections = []
            total_distance = 0
            
            for i in range(len(all_points)):
                for j in range(i + 1, len(all_points)):
                    dist = np.hypot(all_points[i][0] - all_points[j][0], 
                                  all_points[i][1] - all_points[j][1])
                    if dist <= 50:  # 50m 이내만 연결
                        G.add_edge(i, j, weight=dist)
                        connections.append((i, j, dist))
                        total_distance += dist
            
            # 연결된 컴포넌트 분석
            components = list(nx.connected_components(G))
            
            # 거리 통계
            if connections:
                distances = [d for _, _, d in connections]
                min_dist = min(distances)
                max_dist = max(distances)
                avg_dist = sum(distances) / len(distances)
                
                # 거리별 분포
                under_30m = sum(1 for d in distances if d <= 30)
                range_30_50m = sum(1 for d in distances if 30 < d <= 50)
                
                # 연결성 표시
                self.distance_label.setText(
                    f"🔗 네트워크 연결성 분석 ({len(all_points)}개 점)\n"
                    f"연결: {len(connections)}개 | 총 거리: {total_distance:.1f}m\n"
                    f"평균: {avg_dist:.1f}m | 최소: {min_dist:.1f}m | 최대: {max_dist:.1f}m\n"
                    f"네트워크: {len(components)}개 그룹"
                )
            else:
                self.distance_label.setText("거리 정보: 연결된 점이 없습니다")
                
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
        
        logger.info(f"🗑️ AI 삭제 적용: {deleted_count}개 점 삭제")
    
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
        progress.setWindowModality(Qt.WindowModal)
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
                self.prediction_worker.temp_path = temp_path
                # (선택) 나중에 tmp_dir 를 지우기 위해 참조를 남겨둡니다
                self._tmp_dir = tmp_dir

            else:
                progress.close()
                QMessageBox.warning(self, "경고", "도로망 데이터가 없습니다.")
                return
        else:
            # 도로망 모드
            self.prediction_worker = PredictionWorker(self.current_file, self.model_path, self.file_mode)
        
        self.prediction_worker.progress.connect(lambda v, m: (progress.setValue(v), progress.setLabelText(m)))
        self.prediction_worker.prediction_completed.connect(lambda r: (progress.close(), self.on_prediction_completed(r)))
        self.prediction_worker.error_occurred.connect(lambda e: (progress.close(), self.on_prediction_error(e)))
        self.prediction_worker.start()

    def on_prediction_completed(self, result):
        """예측 완료 시 호출"""
        if result['success']:
            # 예측 결과 저장
            self.original_predictions = result.get('predictions', [])
            self.original_confidence_data = result.get('confidence_data', [])
            
            # 신뢰도 기반 필터링 및 적용
            self.filter_and_apply_predictions()
            
            self.statusBar().showMessage(f"AI 예측 완료", 3000)
            
            # 수정 통계 업데이트
            self.update_modification_stats()
        else:
            self.result_label.setText("예측 실패!")
            self.statusBar().showMessage("AI 예측 실패", 3000)

    def on_prediction_error(self, error_msg):
        QMessageBox.critical(self, "오류", f"AI 예측 실패:\n{error_msg}")
        
        # 임시 파일 정리
        if hasattr(self.prediction_worker, 'temp_path'):
            try:
                Path(self.prediction_worker.temp_path).unlink()
                for ext in ['.shx', '.dbf', '.cpg', '.prj']:
                    Path(self.prediction_worker.temp_path.replace('.shp', ext)).unlink(missing_ok=True)
            except:
                pass

    def toggle_ai_predictions(self, checked):
        self.canvas_widget.canvas.show_ai_predictions = checked
        self.canvas_widget.canvas.update_display()

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
        
        # 2단계: 연결성 검사 기반 커브점 삭제 (1개만)
        connectivity_delete_count = 0
        if self.remove_one_curve_point_by_connectivity():
            connectivity_delete_count = 1
        
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
    
    def remove_one_curve_point_by_connectivity(self):
        """연결성 검사로 직선상 커브점 1개 삭제 (거리 짧은 순)"""
        try:
            # 1. 모든 점 수집
            all_points = []
            for category in ['intersection', 'curve', 'endpoint']:
                for point in self.canvas_widget.canvas.points[category]:
                    all_points.append({
                        'point': point,
                        'category': category,
                        'coords': (float(point[0]), float(point[1]))
                    })
            
            if len(all_points) < 3:
                logger.info("연결성 검사: 점이 부족합니다")
                return False
            
            # 2. 도로망 폴리곤 가져오기
            road_union = self.get_road_union()
            if not road_union:
                logger.info("연결성 검사: 도로망 데이터가 없습니다")
                return False
            
            # 3. 삭제 가능한 커브점 찾기
            deletable_curves = []
            
            # 모든 점 쌍 검사
            for i, p1 in enumerate(all_points):
                for j, p2 in enumerate(all_points):
                    if i >= j:
                        continue
                    
                    # 직선 생성
                    try:
                        from shapely.geometry import LineString, Point
                        line = LineString([p1['coords'], p2['coords']])
                        
                        # 너무 짧은 직선은 제외
                        if line.length < 10:
                            continue
                        
                        # 도로망 내부에 있는지 확인
                        if road_union.contains(line) or road_union.intersects(line):
                            
                            # 직선상의 중간 커브점 찾기
                            for k, p3 in enumerate(all_points):
                                if k == i or k == j:
                                    continue
                                
                                # 교차점은 제외
                                if p3['category'] == 'intersection':
                                    continue
                                
                                # 커브점만 대상
                                if p3['category'] == 'curve':
                                    # 점이 직선상에 있는지 확인
                                    point_dist_to_line = line.distance(Point(p3['coords']))
                                    
                                    if point_dist_to_line < 5.0:  # 5픽셀 이내면 직선상
                                        # 두 점 사이의 정확한 거리 계산
                                        segment_length = self.calculate_accurate_point_distance(
                                            p1['coords'], p2['coords']
                                        )
                                        
                                        deletable_curves.append({
                                            'point_info': p3,
                                            'distance': segment_length,
                                            'line_endpoints': (p1, p2)
                                        })
                    except Exception as e:
                        logger.warning(f"직선 검사 오류: {e}")
                        continue
            
            # 4. 거리 짧은 순으로 정렬
            deletable_curves.sort(key=lambda x: x['distance'])
            
            # 5. 가장 짧은 거리의 커브점 1개만 삭제
            if deletable_curves:
                to_delete = deletable_curves[0]
                point_info = to_delete['point_info']
                
                # 실제 삭제 수행
                if point_info['point'] in self.canvas_widget.canvas.points['curve']:
                    self.canvas_widget.canvas.points['curve'].remove(point_info['point'])
                    
                    logger.info(f"연결성 검사 삭제: 거리 {to_delete['distance']:.1f}의 커브점 "
                               f"({point_info['coords'][0]:.1f}, {point_info['coords'][1]:.1f})")
                    
                    # UI 업데이트
                    self.canvas_widget.canvas.update_display()
                    return True
            
            logger.info("연결성 검사: 삭제할 커브점이 없습니다")
            return False
            
        except Exception as e:
            logger.error(f"연결성 검사 삭제 오류: {e}")
            return False
    
    def calculate_accurate_point_distance(self, p1, p2):
        """정확한 점간 거리 계산"""
        try:
            import math
            
            # 유클리드 거리 계산 (픽셀 단위)
            dx = float(p1[0]) - float(p2[0])
            dy = float(p1[1]) - float(p2[1])
            distance = math.sqrt(dx * dx + dy * dy)
            
            # 좌표계 스케일 적용 (필요시)
            if hasattr(self, 'coordinate_scale') and self.coordinate_scale:
                distance *= self.coordinate_scale
            
            return distance
            
        except Exception as e:
            logger.warning(f"거리 계산 오류: {e}")
            return float('inf')
    
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
    
    def remove_clustered_points(self, distance_threshold=15.0):
        """가까운 점들 중 하나씩 삭제"""
        deleted_count = 0
        
        # 모든 점 수집
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
        
        # 가까운 점들 찾기
        points_to_remove = []
        used_indices = set()
        
        for i, p1 in enumerate(all_points):
            if i in used_indices:
                continue
            
            # 가까운 점들 찾기
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
            
            # 가까운 점들이 있으면 하나 삭제 (가장 가까운 점 제거)
            if nearby_points:
                # 거리 기준 정렬
                nearby_points.sort(key=lambda x: x[2])
                
                # 가장 가까운 점 하나만 삭제
                to_remove_idx, to_remove_point, _ = nearby_points[0]
                points_to_remove.append(to_remove_point)
                used_indices.add(to_remove_idx)
                used_indices.add(i)  # 기준점도 사용됨으로 표시
                
                logger.info(f"가까운 점 삭제: {to_remove_point['category']} 카테고리 "
                           f"({to_remove_point['point'][0]:.1f}, {to_remove_point['point'][1]:.1f})")
        
        # 실제 삭제 수행 (역순으로 삭제하여 인덱스 문제 방지)
        for point_info in points_to_remove:
            category = point_info['category']
            point = point_info['point']
            
            # 해당 카테고리에서 점 제거
            if point in self.canvas_widget.canvas.points[category]:
                self.canvas_widget.canvas.points[category].remove(point)
                deleted_count += 1
        
        logger.info(f"가까운 점 삭제 완료: {deleted_count}개")
        return deleted_count

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
        
        # 테이블 업데이트
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

    def update_confidence(self, value):
        """신뢰도 슬라이더 값 업데이트"""
        confidence = value / 100.0
        self.ai_confidence_threshold = confidence
        self.confidence_label.setText(f"{confidence:.2f}")
        
        # 신뢰도 변경 시 예측 재필터링
        if self.original_predictions is not None:
            self.filter_and_apply_predictions()

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
        
        # 8. 결과 표시 (AI 보조 결과처럼 표시)
        self.result_label.setText(
            f"AI 보조 예측 완료\n"
            f"교차점: {len(heuristic_intersections)}개\n"
            f"커브: {len(filtered_curves)}개 (도로 경계선 기반)\n"
            f"끝점: {len(heuristic_endpoints)}개\n"
            f"AI 삭제 후보: {len(ai_delete_points)}개"
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
                boundaries = [poly.exterior for poly in road_polygon.geoms]
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
                points = np.array([cp['point'] for cp in all_curvature_points])
                clustering = DBSCAN(eps=cluster_radius, min_samples=2)
                labels = clustering.fit_predict(points)
                
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


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = InferenceTool()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()