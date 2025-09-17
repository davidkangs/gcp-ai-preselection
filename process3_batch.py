import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import geopandas as gpd
import torch

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal

sys.path.append(str(Path(__file__).parent.parent))
from src.core.skeleton_extractor import SkeletonExtractor
from src.core.district_road_clipper import DistrictRoadClipper
from src.learning.dqn_model import create_agent
from src.utils import save_session

try:
    from enhanced_heuristic_detector_v2 import EnhancedHeuristicDetectorV2
    HEURISTIC_DETECTOR_LOADED = True
except ImportError:
    EnhancedHeuristicDetectorV2 = None
    HEURISTIC_DETECTOR_LOADED = False

import logging
logger = logging.getLogger(__name__)


class BatchInferenceWorker(QThread):
    file_started = pyqtSignal(str)
    file_completed = pyqtSignal(str, dict)
    progress_updated = pyqtSignal(int, str)
    error_occurred = pyqtSignal(str, str)
    
    def __init__(self, file_list, model_path, save_results=True, file_mode='road', target_crs='EPSG:5186'):
        super().__init__()
        self.file_list = file_list
        self.model_path = model_path
        self.save_results = save_results
        self.file_mode = file_mode
        self.target_crs = target_crs
        self.skeleton_extractor = SkeletonExtractor()
        self.district_clipper = DistrictRoadClipper()
        self.heuristic_detector = EnhancedHeuristicDetectorV2() if HEURISTIC_DETECTOR_LOADED else None

    def _create_dqn_state_vector(self, point, skeleton, idx, heuristic_results=None):
        features = []
        x, y = float(point[0]), float(point[1])
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
        density = density / len(skeleton) if skeleton else 0.0
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
        
        # 통합 특징 추출기 사용 시도
        try:
            from src.core.unified_feature_extractor import get_feature_extractor
            extractor = get_feature_extractor()
            
            if extractor and extractor.topology_analyzer:
                unified_features = extractor.extract_features((x, y), idx, heuristic_results)
                return unified_features
        except ImportError:
            pass
        
        # 20차원까지 패딩
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def remove_duplicate_points(self, points_dict):
        """5m 반경 내 중복점 제거 (우선순위: 끝점 > 교차점 > 커브점)"""
        # 모든 포인트를 우선순위와 함께 수집
        all_points_with_priority = []
        
        # 우선순위: 끝점(3) > 교차점(2) > 커브점(1)
        for x, y in points_dict.get('endpoint', []):
            all_points_with_priority.append((x, y, 3, 'endpoint'))
        
        for x, y in points_dict.get('intersection', []):
            all_points_with_priority.append((x, y, 2, 'intersection'))
        
        for x, y in points_dict.get('curve', []):
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
        
        return cleaned_points
    
    def process_district_file(self, file_path):
        """지구계 파일 처리하여 도로망 추출"""
        results = self.district_clipper.process_district_file(
            file_path,
            target_crs=self.target_crs,
            auto_find_road=True
        )
        
        if not results['success']:
            return None, None
        
        # 첫 번째 폴리곤의 클리핑된 도로망 반환
        if results['polygons'] and results['polygons'][0].get('clipped_road') is not None:
            return results['polygons'][0]['clipped_road'], results
        
        return None, results
    
    def run(self):
        try:
            agent = create_agent()
            agent.load(self.model_path)
            agent.q_network.eval()
        except Exception as e:
            self.error_occurred.emit("", f"모델 로드 실패: {str(e)}")
            return
        
        total_files = len(self.file_list)
        results_summary = []
        
        for idx, file_path in enumerate(self.file_list):
            try:
                self.file_started.emit(file_path)
                
                progress = int((idx / total_files) * 100)
                self.progress_updated.emit(
                    progress,
                    f"처리 중 ({idx + 1}/{total_files}): {Path(file_path).name}"
                )
                
                # 파일 타입 확인 및 처리
                if self.file_mode == 'district' or file_path.endswith('.shp') and not file_path.endswith('_road.shp'):
                    # 지구계 파일 처리
                    road_gdf, district_results = self.process_district_file(file_path)
                    
                    if road_gdf is None:
                        self.error_occurred.emit(file_path, "도로망 추출 실패")
                        continue
                    
                    # 임시 파일로 저장하여 스켈레톤 추출
                    with tempfile.NamedTemporaryFile(suffix='.shp', delete=False) as tmp:
                        temp_path = tmp.name
                    
                    road_gdf.to_file(temp_path)
                    skeleton, intersections = self.skeleton_extractor.process_shapefile(temp_path)
                    
                    # 임시 파일 삭제
                    Path(temp_path).unlink()
                    for ext in ['.shx', '.dbf', '.cpg', '.prj']:
                        Path(temp_path.replace('.shp', ext)).unlink(missing_ok=True)
                    
                    gdf = road_gdf
                    
                else:
                    # 일반 도로망 파일
                    skeleton, intersections = self.skeleton_extractor.process_shapefile(file_path)
                    gdf = gpd.read_file(file_path)
                
                if self.heuristic_detector:
                    heuristic = self.heuristic_detector.detect_all(gdf, skeleton, intersections)
                else:
                    heuristic = {
                        'intersection': [tuple(pt) for pt in intersections],
                        'curve': [],
                        'endpoint': []
                    }
                
                features = []
                skeleton_array = np.array(skeleton)
                
                for i, point in enumerate(skeleton_array):
                    feat = self._create_dqn_state_vector(point, skeleton_array, i)
                    features.append(feat)
                
                predictions = agent.predict(np.array(features))
                
                processed = self.postprocess_predictions(
                    skeleton, predictions, heuristic['intersection']
                )
                
                # 중복점 제거 적용
                processed = self.remove_duplicate_points(processed)
                
                if self.save_results:
                    labels = {
                        'intersection': processed['intersection'],
                        'curve': processed['curve'],
                        'endpoint': processed['endpoint']
                    }
                    
                    metadata = {
                        'process': 'batch_inference',
                        'model': Path(self.model_path).name,
                        'auto_generated': True,
                        'heuristic_detector': 'integrated' if self.heuristic_detector else 'basic',
                        'duplicate_removed': True,
                        'file_mode': self.file_mode,
                        'target_crs': self.target_crs if self.file_mode == 'district' else None
                    }
                    
                    user_actions = []
                    save_session(file_path, labels, skeleton, metadata, user_actions)
                
                result = {
                    'success': True,
                    'skeleton_points': len(skeleton),
                    'heuristic_intersection': len(heuristic['intersection']),
                    'heuristic_curve': len(heuristic['curve']),
                    'ai_intersection': len(processed['intersection']) - len(heuristic['intersection']),
                    'ai_curve': len(processed['curve']),
                    'ai_endpoint': len(processed['endpoint']),
                    'total_intersection': len(processed['intersection']),
                    'total_curve': len(processed['curve']),
                    'total_endpoint': len(processed['endpoint']),
                    'file_mode': self.file_mode
                }
                
                results_summary.append({
                    'file': Path(file_path).name,
                    'result': result
                })
                
                self.file_completed.emit(file_path, result)
                
            except Exception as e:
                logger.error(f"파일 처리 오류 {file_path}: {e}")
                self.error_occurred.emit(file_path, str(e))
                
                result = {'success': False, 'error': str(e)}
                self.file_completed.emit(file_path, result)
        
        self.progress_updated.emit(100, "배치 처리 완료")
    
    def postprocess_predictions(self, skeleton, predictions, heuristic_intersections):
        skeleton_array = np.array(skeleton)
        
        result = {
            'intersection': list(heuristic_intersections),
            'curve': [],
            'endpoint': [],
            'delete': []
        }
        
        intersection_indices = []
        curve_indices = []
        endpoint_indices = []
        delete_indices = []
        
        for i, pred in enumerate(predictions):
            if pred == 1:
                intersection_indices.append(i)
            elif pred == 2:
                curve_indices.append(i)
            elif pred == 3:
                endpoint_indices.append(i)
            elif pred == 4:
                delete_indices.append(i)
        
        for idx in intersection_indices:
            point = tuple(skeleton_array[idx])
            
            is_duplicate = False
            for existing_point in result['intersection']:
                if np.linalg.norm(np.array(point) - np.array(existing_point)) < 20:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                result['intersection'].append(point)
        
        if curve_indices:
            segments = []
            current_segment = []
            
            for idx in sorted(curve_indices):
                if not current_segment or idx - current_segment[-1] <= 5:
                    current_segment.append(idx)
                else:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = [idx]
            
            if current_segment:
                segments.append(current_segment)
            
            for segment in segments:
                if len(segment) >= 3:
                    mid_idx = segment[len(segment)//2]
                    result['curve'].append(tuple(skeleton_array[mid_idx]))
        
        skeleton_length = len(skeleton_array)
        
        for idx in endpoint_indices:
            is_near_start = idx < 20
            is_near_end = idx > skeleton_length - 20
            
            if is_near_start or is_near_end:
                point = skeleton_array[idx]
                
                too_close = False
                for existing_point in result['endpoint']:
                    if np.linalg.norm(point - np.array(existing_point)) < 30:
                        too_close = True
                        break
                
                if not too_close:
                    result['endpoint'].append(tuple(point))
        
        for idx in delete_indices:
            result['delete'].append(tuple(skeleton_array[idx]))
        
        return result


class BatchInferenceDialog(QDialog):
    def __init__(self, folder_path, model_path, parent=None):
        super().__init__(parent)
        self.folder_path = folder_path
        self.model_path = model_path
        self.file_list = []
        self.worker = None
        self.file_mode = 'auto'  # 'auto', 'district', 'road'
        self.target_crs = 'EPSG:5186'
        
        self.init_ui()
        self.load_shapefiles()
    
    def init_ui(self):
        self.setWindowTitle("배치 AI 예측 (DQN 호환 + 지구계 지원)")
        self.setGeometry(200, 200, 800, 650)
        self.setModal(True)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 파일 모드 선택
        mode_group = QGroupBox("파일 처리 모드")
        mode_layout = QVBoxLayout()
        
        self.auto_radio = QRadioButton("자동 감지 (파일명으로 구분)")
        self.auto_radio.setChecked(True)
        self.auto_radio.toggled.connect(lambda: self.set_file_mode('auto'))
        mode_layout.addWidget(self.auto_radio)
        
        self.district_radio = QRadioButton("모두 지구계 파일로 처리")
        self.district_radio.toggled.connect(lambda: self.set_file_mode('district'))
        mode_layout.addWidget(self.district_radio)
        
        self.road_radio = QRadioButton("모두 도로망 파일로 처리")
        self.road_radio.toggled.connect(lambda: self.set_file_mode('road'))
        mode_layout.addWidget(self.road_radio)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # 좌표계 선택
        crs_group = QGroupBox("좌표계 (지구계 모드)")
        crs_layout = QHBoxLayout()
        
        self.crs_5186_radio = QRadioButton("EPSG:5186 (중부원점)")
        self.crs_5186_radio.setChecked(True)
        self.crs_5186_radio.toggled.connect(lambda: self.set_target_crs('EPSG:5186'))
        crs_layout.addWidget(self.crs_5186_radio)
        
        self.crs_5187_radio = QRadioButton("EPSG:5187 (동부원점)")
        self.crs_5187_radio.toggled.connect(lambda: self.set_target_crs('EPSG:5187'))
        crs_layout.addWidget(self.crs_5187_radio)
        
        crs_group.setLayout(crs_layout)
        layout.addWidget(crs_group)
        
        # 배치 처리 정보
        info_group = QGroupBox("배치 처리 정보")
        info_layout = QVBoxLayout()
        
        info_text = f"""폴더: {Path(self.folder_path).name}
모델: {Path(self.model_path).name}
검출기: {'통합 휴리스틱' if HEURISTIC_DETECTOR_LOADED else '기본 교차점만'}
DQN 호환: 1=교차점, 2=커브, 3=끝점
중복제거: 5m 반경 (끝점>교차점>커브점)"""
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("QLabel { padding: 10px; background-color: #f0f0f0; }")
        info_layout.addWidget(info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # 파일 목록
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
        
        # 처리 로그
        log_group = QGroupBox("처리 로그")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # 버튼
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("배치 예측 시작")
        self.start_btn.clicked.connect(self.start_inference)
        self.start_btn.setStyleSheet("QPushButton {background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;}")
        button_layout.addWidget(self.start_btn)
        
        self.close_btn = QPushButton("닫기")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def set_file_mode(self, mode):
        self.file_mode = mode
        logger.info(f"파일 모드 변경: {mode}")
    
    def set_target_crs(self, crs):
        self.target_crs = crs
        logger.info(f"좌표계 변경: {crs}")
    
    def load_shapefiles(self):
        self.file_list_widget.clear()
        self.file_list = []
        
        for file_path in Path(self.folder_path).glob("*.shp"):
            self.file_list.append(str(file_path))
            
            # 파일 타입 표시
            file_name = file_path.name
            if self.file_mode == 'auto':
                if file_name.endswith('_road.shp'):
                    display_name = f"[도로망] {file_name}"
                else:
                    display_name = f"[지구계] {file_name}"
            else:
                display_name = file_name
            
            self.file_list_widget.addItem(display_name)
        
        self.status_label.setText(f"{len(self.file_list)}개의 shapefile 발견")
        
        if not self.file_list:
            self.start_btn.setEnabled(False)
    
    def start_inference(self):
        if not self.file_list:
            QMessageBox.warning(self, "경고", "처리할 파일이 없습니다.")
            return
        
        reply = QMessageBox.question(
            self, "확인",
            f"{len(self.file_list)}개 파일을 배치 처리하시겠습니까?\n"
            f"모델: {Path(self.model_path).name}\n"
            f"모드: {self.file_mode}",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        self.start_btn.setEnabled(False)
        self.log_text.clear()
        self.log_text.append(f"=== 배치 추론 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        self.log_text.append(f"모델: {Path(self.model_path).name}")
        self.log_text.append(f"파일 수: {len(self.file_list)}개")
        self.log_text.append(f"파일 모드: {self.file_mode}")
        self.log_text.append(f"좌표계: {self.target_crs}\n")
        
        self.worker = BatchInferenceWorker(
            self.file_list, 
            self.model_path,
            save_results=True,
            file_mode=self.file_mode,
            target_crs=self.target_crs
        )
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_completed.connect(self.on_file_completed)
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.error_occurred.connect(self.on_error_occurred)
        
        self.worker.start()
    
    def on_file_started(self, file_path):
        self.log_text.append(f"[시작] {Path(file_path).name}")
        self.log_text.moveCursor(self.log_text.textCursor().End)
    
    def on_file_completed(self, file_path, result):
        filename = Path(file_path).name
        
        if result['success']:
            file_type = "지구계" if result.get('file_mode') == 'district' else "도로망"
            self.log_text.append(
                f"[완료] {filename} ({file_type})\n"
                f"  → 교차점: {result['total_intersection']}개 (휴리스틱: {result['heuristic_intersection']}, AI: {result['ai_intersection']})\n"
                f"  → 커브: {result['total_curve']}개, 끝점: {result['total_endpoint']}개\n"
                f"  → 중복 제거 완료 (5m 반경)\n"
            )
        else:
            self.log_text.append(f"[오류] {filename} - {result.get('error', '알 수 없는 오류')}\n")
        
        self.log_text.moveCursor(self.log_text.textCursor().End)
    
    def on_progress_updated(self, progress, message):
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
        
        if progress == 100:
            self.start_btn.setEnabled(True)
            self.log_text.append(f"=== 배치 추론 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            self.log_text.moveCursor(self.log_text.textCursor().End)
            
            QMessageBox.information(
                self, "완료",
                f"배치 처리가 완료되었습니다.\n"
                f"처리된 파일: {len(self.file_list)}개\n"
                f"중복점 정리 완료 (5m 반경)"
            )
    
    def on_error_occurred(self, file_path, error_msg):
        if file_path:
            self.log_text.append(f"[오류] {Path(file_path).name}: {error_msg}")
        else:
            self.log_text.append(f"[시스템 오류] {error_msg}")
        
        self.log_text.moveCursor(self.log_text.textCursor().End)