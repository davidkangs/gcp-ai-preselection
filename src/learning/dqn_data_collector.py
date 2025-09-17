import json
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

try:
    from ..core.unified_feature_extractor import get_feature_extractor, initialize_global_extractor
except ImportError:
    logger.warning("통합 특징 추출기를 가져올 수 없습니다. 기본 구현을 사용합니다.")
    get_feature_extractor = None
    initialize_global_extractor = None

class DQNDataCollector:
    def __init__(self, data_dir="data/training_samples"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session = {
            'session_id': None,
            'start_time': None,
            'file_path': None,
            'samples': []
        }
        
        self.canvas_widget = None
        self.skeleton_data = None
        self.detected_points = None
        self.boundary_polygon = None
        self.feature_extractor = None
        self.heuristic_results = None
        
        logger.info(f"DQN 데이터 수집기 초기화 - 저장 경로: {self.data_dir}")
    
    def connect_to_canvas(self, canvas_widget):
        self.canvas_widget = canvas_widget
        
        if hasattr(canvas_widget, 'point_added'):
            canvas_widget.point_added.connect(self.on_point_added)
        if hasattr(canvas_widget, 'point_removed'):
            canvas_widget.point_removed.connect(self.on_point_removed)
        
        logger.info("캔버스 위젯에 데이터 수집기 연결 완료")
    
    def start_session(self, file_path, skeleton_data=None, detected_points=None, 
                     boundary_polygon=None, heuristic_results=None):
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.current_session = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'file_path': str(file_path),
            'samples': []
        }
        
        self.skeleton_data = skeleton_data
        self.detected_points = detected_points
        self.boundary_polygon = boundary_polygon
        self.heuristic_results = heuristic_results
        
        # 통합 특징 추출기 초기화
        if get_feature_extractor and skeleton_data:
            try:
                self.feature_extractor = initialize_global_extractor(
                    skeleton_data, boundary_polygon, skeleton_data.get('transform')
                )
                logger.info("통합 특징 추출기 초기화 완료")
            except Exception as e:
                logger.warning(f"통합 특징 추출기 초기화 실패: {e}")
                self.feature_extractor = None
        
        logger.info(f"DQN 세션 시작: {session_id} - {Path(file_path).name if file_path else 'Unknown'}")
    
    def on_point_added(self, category, x, y):
        if not self.current_session['session_id']:
            return
            
        try:
            state_vector = self.extract_state_vector(x, y, category)
            
            # 3-액션 시스템: 0=keep, 1=add_curve, 2=delete  
            # 교차점과 끝점은 휴리스틱이 담당하므로 AI가 추가하지 않음
            # AI는 커브만 추가
            if category == 'curve':
                action = 1  # add_curve
            else:
                # 교차점이나 끝점을 수동 추가하는 경우는 keep으로 처리
                action = 0  # keep (휴리스틱이 놓친 경우)
            
            sample = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'category': category,
                'position': [float(x), float(y)],
                'state_vector': state_vector.tolist(),
                'reward': 1.0,
                'context': self.get_context_info()
            }
            
            self.current_session['samples'].append(sample)
            logger.debug(f"포인트 추가 수집: {category} at ({x:.1f}, {y:.1f}) -> action={action}")
            
        except Exception as e:
            logger.error(f"포인트 추가 데이터 수집 실패: {e}")
    
    def on_point_removed(self, category, x, y):
        if not self.current_session['session_id']:
            return
            
        try:
            state_vector = self.extract_state_vector(x, y, category)
            
            # 3-액션 시스템: 모든 삭제는 action=2 (delete)
            sample = {
                'timestamp': datetime.now().isoformat(),
                'action': 2,  # delete
                'category': category,
                'position': [float(x), float(y)],
                'state_vector': state_vector.tolist(),
                'reward': 1.0,
                'context': self.get_context_info()
            }
            
            self.current_session['samples'].append(sample)
            logger.debug(f"포인트 삭제 수집: {category} at ({x:.1f}, {y:.1f}) -> action=2")
            
        except Exception as e:
            logger.error(f"포인트 삭제 데이터 수집 실패: {e}")
    
    def extract_state_vector(self, x, y, category):
        """통합 특징 추출기를 사용한 20차원 상태 벡터 추출"""
        try:
            # 통합 특징 추출기 사용
            if self.feature_extractor:
                skeleton_index = self._find_skeleton_index(x, y)
                features = self.feature_extractor.extract_features(
                    (x, y), skeleton_index, self.heuristic_results
                )
                return np.array(features)
            
            # 폴백: 기본 구현
            logger.warning("통합 특징 추출기 없음. 기본 구현 사용")
            return self._extract_basic_features(x, y, category)
            
        except Exception as e:
            logger.error(f"상태 벡터 추출 실패: {e}")
            return np.zeros(20)
    
    def _find_skeleton_index(self, x, y) -> Optional[int]:
        """스켈레톤에서 가장 가까운 점의 인덱스 찾기"""
        if not self.skeleton_data or 'skeleton' not in self.skeleton_data:
            return None
        
        skeleton = self.skeleton_data['skeleton']
        if not skeleton:
            return None
        
        min_dist = float('inf')
        best_index = None
        
        for i, point in enumerate(skeleton):
            if len(point) >= 2:
                dist = np.linalg.norm([x - point[0], y - point[1]])
                if dist < min_dist:
                    min_dist = dist
                    best_index = i
        
        # 너무 멀면 None 반환
        return best_index if min_dist < 50 else None
    
    def _extract_basic_features(self, x, y, category):
        """기본 특징 추출 (폴백용)"""
        features = [float(x), float(y)]
        
        # 실제 계산 시도
        try:
            road_count, avg_road_length = self.analyze_roads_around_point(x, y)
            features.extend([road_count, avg_road_length])
            
            distances = self.get_nearest_point_distances(x, y, category)
            features.extend(distances)
            
            density_info = self.calculate_local_density(x, y)
            features.extend(density_info)
            
            geometric_features = self.get_geometric_features(x, y)
            features.extend(geometric_features)
            
            heuristic_confidence = self.get_heuristic_confidence(x, y, category)
            features.append(heuristic_confidence)
            
        except Exception as e:
            logger.warning(f"기본 특징 추출 실패, 더미 값 사용: {e}")
            # 더미 값으로 채우기
            features.extend([0.0] * 18)
        
        # 20차원 맞추기
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20])
    
    def analyze_roads_around_point(self, x, y, radius=50):
        """실제 주변 도로 분석"""
        if not self.skeleton_data or 'skeleton' not in self.skeleton_data:
            return 0, 0.0
        
        skeleton = self.skeleton_data['skeleton']
        if not skeleton:
            return 0, 0.0
        
        # 반경 내 스켈레톤 점들 찾기
        nearby_points = []
        for point in skeleton:
            if len(point) >= 2:
                dist = np.linalg.norm([x - point[0], y - point[1]])
                if dist <= radius:
                    nearby_points.append(point)
        
        if not nearby_points:
            return 0, 0.0
        
        # 연결된 도로 세그먼트 길이 계산
        total_length = 0.0
        for i in range(len(nearby_points) - 1):
            p1, p2 = nearby_points[i], nearby_points[i + 1]
            if len(p1) >= 2 and len(p2) >= 2:
                segment_length = np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1]])
                total_length += segment_length
        
        road_count = max(1, len(nearby_points) // 3)  # 대략적인 도로 수
        avg_length = total_length / road_count if road_count > 0 else 0.0
        
        return road_count, avg_length
    
    def get_nearest_point_distances(self, x, y, category, max_points=5):
        """실제 가장 가까운 점들까지의 거리"""
        if not self.detected_points:
            return [1000.0] * max_points
        
        distances = []
        
        # 모든 카테고리의 점들 수집
        all_points = []
        for cat, points in self.detected_points.items():
            for point in points:
                if len(point) >= 2:
                    dist = np.linalg.norm([x - point[0], y - point[1]])
                    all_points.append(dist)
        
        # 거리순 정렬
        all_points.sort()
        
        # 상위 max_points개 선택
        for i in range(max_points):
            if i < len(all_points):
                distances.append(all_points[i])
            else:
                distances.append(1000.0)
        
        return distances
    
    def calculate_local_density(self, x, y, radii=[25, 50, 100]):
        """실제 지역 밀도 계산"""
        if not self.skeleton_data or 'skeleton' not in self.skeleton_data:
            return [0.0] * len(radii)
        
        skeleton = self.skeleton_data['skeleton']
        densities = []
        
        for radius in radii:
            count = 0
            for point in skeleton:
                if len(point) >= 2:
                    dist = np.linalg.norm([x - point[0], y - point[1]])
                    if dist <= radius:
                        count += 1
            
            # 밀도 = 개수 / 면적
            area = np.pi * radius * radius
            density = count / area if area > 0 else 0.0
            densities.append(density * 10000)  # 스케일 조정
        
        return densities
    
    def get_geometric_features(self, x, y):
        """기하학적 특징 계산"""
        features = []
        
        # 좌표 기반 특징
        features.append(x % 100)
        features.append(y % 100)
        features.append((x + y) % 50)
        features.append(abs(x - y) % 30)
        
        return features
    
    def get_heuristic_confidence(self, x, y, category):
        """휴리스틱 신뢰도 계산"""
        if not self.heuristic_results:
            return 0.5
        
        # 해당 카테고리의 휴리스틱 결과와의 거리 확인
        if category in self.heuristic_results:
            min_dist = float('inf')
            for hpoint in self.heuristic_results[category]:
                if len(hpoint) >= 2:
                    dist = np.linalg.norm([x - hpoint[0], y - hpoint[1]])
                    min_dist = min(min_dist, dist)
            
            # 거리에 따른 신뢰도 (가까울수록 높음)
            if min_dist < 20:
                return 0.9
            elif min_dist < 50:
                return 0.7
            else:
                return 0.3
        
        return 0.5
    
    def get_context_info(self):
        if not self.current_session['start_time']:
            return {}
        
        start_time = datetime.fromisoformat(self.current_session['start_time'])
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            'session_duration': duration,
            'total_edits': len(self.current_session['samples']),
            'edit_rate': len(self.current_session['samples']) / max(duration, 1) * 60
        }
    
    def end_session(self):
        if not self.current_session['session_id']:
            return None
        
        try:
            self.current_session['end_time'] = datetime.now().isoformat()
            self.current_session['total_samples'] = len(self.current_session['samples'])
            
            filename = f"session_{self.current_session['session_id']}.json"
            filepath = self.data_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.current_session, f, indent=2, ensure_ascii=False)
            
            logger.info(f"세션 데이터 저장: {filepath} ({len(self.current_session['samples'])}개 샘플)")
            
            session_id = self.current_session['session_id']
            self.current_session = {
                'session_id': None,
                'start_time': None,
                'file_path': None,
                'samples': []
            }
            
            return filepath
            
        except Exception as e:
            logger.error(f"세션 데이터 저장 실패: {e}")
            return None
    
    def get_collected_data_summary(self):
        try:
            data_files = list(self.data_dir.glob("session_*.json"))
            total_sessions = len(data_files)
            total_samples = 0
            
            for filepath in data_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                        total_samples += len(session_data.get('samples', []))
                except Exception:
                    continue
            
            return {
                'total_sessions': total_sessions,
                'total_samples': total_samples,
                'data_directory': str(self.data_dir),
                'average_samples_per_session': total_samples / max(total_sessions, 1)
            }
            
        except Exception as e:
            logger.error(f"데이터 요약 실패: {e}")
            return {'error': str(e)}

def create_data_collector(data_dir="data/training_samples"):
    return DQNDataCollector(data_dir)

def load_training_data(data_dir="data/training_samples"):
    data_path = Path(data_dir)
    all_samples = []
    
    for session_file in data_path.glob("session_*.json"):
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                all_samples.extend(session_data.get('samples', []))
        except Exception as e:
            logger.error(f"세션 파일 로드 실패 {session_file}: {e}")
    
    return all_samples