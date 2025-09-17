"""
파이프라인 관리 모듈 - 전체 자동화 파이프라인 관리
"""

import os
import tempfile
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import numpy as np

from .data_processor import DataProcessor
from .filter_manager import FilterManager
from .ai_predictor import AIPredictor
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class PipelineManager:
    """파이프라인 관리 클래스"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 session_dir: str = "sessions",
                 filter_config: Optional[Dict[str, float]] = None):
        """
        Args:
            model_path: AI 모델 경로
            session_dir: 세션 저장 디렉토리
            filter_config: 필터 설정 {'dbscan_eps': 20.0, 'network_max_dist': 50.0, 'road_buffer': 2.0}
        """
        # 모듈 초기화
        self.data_processor = DataProcessor()
        
        # 필터 설정
        filter_config = filter_config or {}
        self.filter_manager = FilterManager(
            dbscan_eps=filter_config.get('dbscan_eps', 20.0),
            network_max_dist=filter_config.get('network_max_dist', 50.0),
            road_buffer=filter_config.get('road_buffer', 2.0)
        )
        
        self.ai_predictor = AIPredictor(model_path)
        self.session_manager = SessionManager(session_dir)
        
        # 파이프라인 설정
        self.pipeline_config = {
            'auto_filter': True,
            'ai_confidence_threshold': 0.7,
            'curve_detection_method': 'boundary_based',  # 'boundary_based' or 'heuristic'
            'enable_distance_calculation': True,
            'save_intermediate_results': False
        }
        
        # 콜백 함수들
        self.progress_callback: Optional[Callable[[int, str], None]] = None
        self.result_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        
        logger.info("PipelineManager 초기화 완료")
    
    def set_progress_callback(self, callback: Callable[[int, str], None]):
        """진행률 콜백 설정"""
        self.progress_callback = callback
    
    def set_result_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """결과 콜백 설정"""
        self.result_callback = callback
    
    def _emit_progress(self, progress: int, message: str):
        """진행률 전송"""
        if self.progress_callback:
            self.progress_callback(progress, message)
        logger.info(f"진행률 {progress}%: {message}")
    
    def _emit_result(self, result: Dict[str, Any]):
        """결과 전송"""
        if self.result_callback:
            self.result_callback(result)
    
    def run_road_pipeline(self, file_path: str, 
                         enable_ai: bool = True,
                         save_session: bool = True,
                         target_crs: str = 'EPSG:5186') -> Dict[str, Any]:
        """
        도로망 파일 자동화 파이프라인
        
        Args:
            file_path: 도로망 파일 경로
            enable_ai: AI 예측 활성화 여부
            save_session: 세션 저장 여부
            target_crs: 대상 좌표계 (기본값: EPSG:5186)
        
        Returns:
            파이프라인 결과
        """
        try:
            self._emit_progress(0, "도로망 파이프라인 시작")
            
            # 1단계: 데이터 처리
            self._emit_progress(10, "🔍 AI가 도로 구조 분석 중...")
            data_result = self.data_processor.process_road_file(file_path, target_crs)
            
            if not data_result['success']:
                return {'success': False, 'error': data_result['error']}
            
            skeleton = data_result['skeleton']
            intersections = data_result['intersections']
            road_gdf = data_result['road_gdf']
            
            # 2단계: 휴리스틱 분석
            self._emit_progress(20, "🎯 AI가 도로 끝점 검출 중...")
            endpoints = self.data_processor.detect_heuristic_endpoints(skeleton)
            
            # 3단계: 커브 검출
            self._emit_progress(30, "🔄 AI가 커브점 검출 중...")
            if self.pipeline_config['curve_detection_method'] == 'boundary_based':
                curves = self.data_processor.detect_boundary_based_curves(
                    skeleton,
                    sample_distance=15.0,
                    curvature_threshold=0.20,
                    road_buffer=3.0,
                    cluster_radius=20.0
                )
                # 교차점 근처 커브점 제거
                curves = self.data_processor.remove_curves_near_intersections(
                    curves, intersections, threshold=10.0
                )
            else:
                curves = []  # 다른 방법이 필요하면 여기에 추가
            
            # 4단계: 초기 점 구성
            self._emit_progress(40, "📍 AI 분석 결과 정리 중...")
            initial_points = {
                'intersection': [(float(x), float(y)) for x, y in intersections],
                'curve': curves,
                'endpoint': endpoints
            }
            
            # 5단계: AI 예측 (선택사항)
            ai_result = None
            if enable_ai and self.ai_predictor.is_model_loaded():
                self._emit_progress(50, "🤖 AI 스마트 예측 실행 중...")
                ai_result = self.ai_predictor.predict_points(
                    skeleton, 
                    confidence_threshold=self.pipeline_config['ai_confidence_threshold']
                )
                
                if ai_result and ai_result['success']:
                    # AI 삭제 포인트 적용
                    delete_points = ai_result['ai_points'].get('delete', [])
                    if delete_points:
                        initial_points = self.ai_predictor.apply_deletions(
                            initial_points, delete_points
                        )
            
            # 6단계: 필터링
            self._emit_progress(60, "🔧 AI 최적화 필터링 중...")
            if self.pipeline_config['auto_filter']:
                filtered_points = self.filter_manager.remove_duplicate_points(
                    initial_points, skeleton
                )
            else:
                filtered_points = initial_points
            
            # 7단계: 거리 계산
            distances_info = {}
            if self.pipeline_config['enable_distance_calculation']:
                self._emit_progress(70, "📏 AI 거리 계산 중...")
                distances_info = self._calculate_distances(filtered_points)
            
            # 8단계: 세션 저장
            session_path = None
            if save_session:
                self._emit_progress(80, "💾 AI 분석 결과 저장 중...")
                session_path = self.session_manager.save_session(
                    file_path=file_path,
                    points=filtered_points,
                    skeleton=skeleton,
                    metadata={
                        'file_mode': 'road',
                        'pipeline_config': self.pipeline_config,
                        'ai_enabled': enable_ai,
                        'ai_result': ai_result is not None
                    }
                )
            
            # 결과 구성
            result = {
                'success': True,
                'skeleton': skeleton,
                'points': filtered_points,
                'road_gdf': road_gdf,
                'ai_result': ai_result,
                'distances': distances_info,
                'session_path': session_path,
                'stats': {
                    'total_skeleton_points': len(skeleton),
                    'detected_intersections': len(intersections),
                    'detected_curves': len(curves),
                    'detected_endpoints': len(endpoints),
                    'final_points': {
                        category: len(points) for category, points in filtered_points.items()
                    }
                }
            }
            
            self._emit_progress(100, "✅ AI 도로 분석 완료!")
            self._emit_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"도로망 파이프라인 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_district_pipeline(self, file_path: str, 
                            target_crs: str = 'EPSG:5186',
                            enable_ai: bool = True,
                            save_session: bool = True) -> Dict[str, Any]:
        """
        지구계 파일 자동화 파이프라인
        
        Args:
            file_path: 지구계 파일 경로
            target_crs: 목표 좌표계
            enable_ai: AI 예측 활성화 여부
            save_session: 세션 저장 여부
        
        Returns:
            파이프라인 결과
        """
        try:
            self._emit_progress(0, "🌍 AI 지구계 분석 시작")
            
            # 1단계: 지구계 파일 처리
            self._emit_progress(10, "🔍 AI가 지구계 도로망 분석 중...")
            district_result = self.data_processor.process_district_file(file_path, target_crs)
            
            if not district_result['success']:
                return {'success': False, 'error': district_result['error']}
            
            polygons = district_result['polygons']
            total_polygons = district_result['total_polygons']
            
            # 2단계: 각 폴리곤 처리
            self._emit_progress(20, f"🗺️ AI 지구 분석 시작 ({total_polygons}개 구역)")
            polygon_results = []
            
            for i, polygon_data in enumerate(polygons):
                self._emit_progress(
                    20 + (i * 60 // total_polygons), 
                    f"🤖 AI 구역 {i+1}/{total_polygons} 분석 중..."
                )
                
                # 폴리곤에 도로망 데이터가 있는지 확인
                if 'clipped_road' not in polygon_data or polygon_data['clipped_road'] is None:
                    continue
                
                # 임시 파일 생성하여 처리
                temp_path = self.data_processor.create_temporary_file(polygon_data['clipped_road'])
                
                try:
                    # 도로망 파이프라인 실행
                    polygon_result = self.run_road_pipeline(
                        temp_path, 
                        enable_ai=enable_ai, 
                        save_session=False,  # 개별 저장은 나중에
                        target_crs=target_crs  # 좌표계 전달
                    )
                    
                    if polygon_result['success']:
                        # 폴리곤 정보 추가
                        # geometry_gdf가 있으면 사용, 없으면 geometry 사용
                        poly_geometry = polygon_data.get('geometry')
                        if 'geometry_gdf' in polygon_data and polygon_data['geometry_gdf'] is not None:
                            # GeoDataFrame에서 geometry 추출
                            poly_gdf = polygon_data['geometry_gdf']
                            if not poly_gdf.empty:
                                poly_geometry = poly_gdf.geometry.iloc[0]
                        
                        polygon_result['polygon_info'] = {
                            'index': i + 1,
                            'total': total_polygons,
                            'geometry': poly_geometry
                        }
                        
                        # 폴리곤 세션 저장
                        if save_session:
                            session_path = self.session_manager.save_polygon_session(
                                file_path=file_path,
                                points=polygon_result['points'],
                                skeleton=polygon_result['skeleton'],
                                polygon_index=i + 1,
                                total_polygons=total_polygons,
                                target_crs=target_crs
                            )
                            polygon_result['session_path'] = session_path
                        
                        polygon_results.append(polygon_result)
                    
                finally:
                    # 임시 파일 정리
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            
            # 결과 통합
            result = {
                'success': True,
                'file_mode': 'district',
                'target_crs': target_crs,
                'total_polygons': total_polygons,
                'processed_polygons': len(polygon_results),
                'polygon_results': polygon_results,
                'summary_stats': self._calculate_summary_stats(polygon_results)
            }
            
            self._emit_progress(100, "✅ AI 지구계 분석 완료!")
            self._emit_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"지구계 파이프라인 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_batch_pipeline(self, file_paths: List[str], 
                          file_mode: str = 'road',
                          enable_ai: bool = True,
                          save_sessions: bool = True,
                          target_crs: str = 'EPSG:5186') -> Dict[str, Any]:
        """
        배치 파이프라인
        
        Args:
            file_paths: 파일 경로 리스트
            file_mode: 파일 모드 ('road' or 'district')
            enable_ai: AI 예측 활성화 여부
            save_sessions: 세션 저장 여부
        
        Returns:
            배치 처리 결과
        """
        try:
            self._emit_progress(0, f"🚀 AI 배치 분석 시작 ({len(file_paths)}개 파일)")
            
            batch_results = []
            successful_count = 0
            failed_count = 0
            
            for i, file_path in enumerate(file_paths):
                self._emit_progress(
                    (i * 90 // len(file_paths)), 
                    f"🤖 AI 파일 {i+1}/{len(file_paths)} 분석 중..."
                )
                
                try:
                    if file_mode == 'district':
                        result = self.run_district_pipeline(
                            file_path, 
                            target_crs=target_crs,  # 좌표계 전달
                            enable_ai=enable_ai, 
                            save_session=save_sessions
                        )
                    else:
                        result = self.run_road_pipeline(
                            file_path, 
                            enable_ai=enable_ai, 
                            save_session=save_sessions,
                            target_crs=target_crs  # 좌표계 전달
                        )
                    
                    if result['success']:
                        successful_count += 1
                    else:
                        failed_count += 1
                    
                    batch_results.append({
                        'file_path': file_path,
                        'result': result
                    })
                    
                except Exception as e:
                    logger.error(f"파일 처리 오류 {file_path}: {e}")
                    failed_count += 1
                    batch_results.append({
                        'file_path': file_path,
                        'result': {'success': False, 'error': str(e)}
                    })
            
            # 배치 결과 통합
            batch_result = {
                'success': True,
                'total_files': len(file_paths),
                'successful_files': successful_count,
                'failed_files': failed_count,
                'file_mode': file_mode,
                'batch_results': batch_results,
                'summary_stats': self._calculate_batch_summary_stats(batch_results)
            }
            
            self._emit_progress(100, f"✅ AI 배치 분석 완료! ({successful_count}개 성공, {failed_count}개 실패)")
            self._emit_result(batch_result)
            
            return batch_result
            
        except Exception as e:
            logger.error(f"배치 파이프라인 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_distances(self, points: Dict[str, List[Tuple[float, float]]]) -> Dict[str, Any]:
        """거리 계산"""
        try:
            import networkx as nx
            
            # 모든 점 수집
            all_points = []
            for category, point_list in points.items():
                all_points.extend(point_list)
            
            if len(all_points) < 2:
                return {'error': '점이 부족합니다'}
            
            # 네트워크 그래프 생성
            G = nx.Graph()
            G.add_nodes_from(range(len(all_points)))
            
            # 가까운 점들 연결 (50m 이내)
            total_distance = 0
            connections = []
            
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
                
                return {
                    'total_points': len(all_points),
                    'total_connections': len(connections),
                    'total_distance': total_distance,
                    'min_distance': min_dist,
                    'max_distance': max_dist,
                    'avg_distance': avg_dist,
                    'network_components': len(components)
                }
            else:
                return {'error': '연결된 점이 없습니다'}
                
        except Exception as e:
            logger.error(f"거리 계산 오류: {e}")
            return {'error': str(e)}
    
    def _calculate_summary_stats(self, polygon_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """폴리곤 결과 요약 통계"""
        if not polygon_results:
            return {}
        
        try:
            total_points = {'intersection': 0, 'curve': 0, 'endpoint': 0}
            total_skeleton_points = 0
            
            for result in polygon_results:
                if result['success']:
                    for category, points in result['points'].items():
                        total_points[category] += len(points)
                    total_skeleton_points += len(result['skeleton'])
            
            return {
                'total_polygons_processed': len(polygon_results),
                'total_skeleton_points': total_skeleton_points,
                'total_detected_points': total_points,
                'avg_points_per_polygon': {
                    category: count / len(polygon_results) 
                    for category, count in total_points.items()
                }
            }
            
        except Exception as e:
            logger.error(f"요약 통계 계산 오류: {e}")
            return {'error': str(e)}
    
    def _calculate_batch_summary_stats(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """배치 결과 요약 통계"""
        if not batch_results:
            return {}
        
        try:
            successful_results = [r for r in batch_results if r['result']['success']]
            
            if not successful_results:
                return {'error': '성공한 결과가 없습니다'}
            
            total_points = {'intersection': 0, 'curve': 0, 'endpoint': 0}
            total_skeleton_points = 0
            
            for batch_item in successful_results:
                result = batch_item['result']
                
                if 'points' in result:
                    for category, points in result['points'].items():
                        total_points[category] += len(points)
                    total_skeleton_points += len(result.get('skeleton', []))
                elif 'polygon_results' in result:
                    # 지구계 모드
                    for poly_result in result['polygon_results']:
                        if poly_result['success']:
                            for category, points in poly_result['points'].items():
                                total_points[category] += len(points)
                            total_skeleton_points += len(poly_result['skeleton'])
            
            return {
                'total_successful_files': len(successful_results),
                'total_skeleton_points': total_skeleton_points,
                'total_detected_points': total_points,
                'avg_points_per_file': {
                    category: count / len(successful_results) 
                    for category, count in total_points.items()
                }
            }
            
        except Exception as e:
            logger.error(f"배치 요약 통계 계산 오류: {e}")
            return {'error': str(e)}
    
    def update_config(self, config: Dict[str, Any]):
        """파이프라인 설정 업데이트"""
        self.pipeline_config.update(config)
        logger.info(f"파이프라인 설정 업데이트: {config}")
    
    def get_config(self) -> Dict[str, Any]:
        """현재 파이프라인 설정 반환"""
        return self.pipeline_config.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """파이프라인 상태 정보"""
        return {
            'ai_model_loaded': self.ai_predictor.is_model_loaded(),
            'ai_model_path': self.ai_predictor.model_path,
            'session_dir': str(self.session_manager.session_dir),
            'pipeline_config': self.pipeline_config,
            'session_stats': self.session_manager.get_session_stats()
        } 