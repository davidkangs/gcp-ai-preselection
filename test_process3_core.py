#!/usr/bin/env python3
"""
Process3 핵심 기능 독립 실행 스크립트
스켈레톤 연결성 기반 거리 분석 시스템 테스트
"""

import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from sklearn.neighbors import KDTree
import logging
import json
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Process3Core:
    def __init__(self):
        self.skeleton_coords = []
        self.analysis_points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        self._road_union_cache = None
        self._graph_cache = {
            'skeleton_id': None,
            'graph': None,
            'kdtree': None,
            'coords': None
        }
        
    def load_test_data(self):
        """테스트용 데이터 로드"""
        logger.info("🔄 테스트용 데이터 생성 중...")
        
        # 실제 도로망 스켈레톤 시뮬레이션 (더 복잡한 구조)
        self.skeleton_coords = [
            # 메인 도로 (수평)
            (0, 50), (20, 50), (40, 50), (60, 50), (80, 50), (100, 50),
            # 교차로 1에서 분기 (수직)
            (20, 50), (20, 30), (20, 10), (20, 0),
            (20, 50), (20, 70), (20, 90), (20, 100),
            # 교차로 2에서 분기 (수직)
            (60, 50), (60, 30), (60, 10), (60, 0),
            (60, 50), (60, 70), (60, 90), (60, 100),
            # 커브 도로 (대각선)
            (80, 50), (85, 45), (90, 40), (95, 35), (100, 30),
            (80, 50), (85, 55), (90, 60), (95, 65), (100, 70),
            # 연결 도로들
            (20, 30), (40, 30), (60, 30),
            (20, 70), (40, 70), (60, 70),
        ]
        
        # 분석 포인트들 (실제 도로망 분석 시나리오)
        self.analysis_points = {
            'intersection': [
                (20, 50),   # 교차로 1
                (60, 50),   # 교차로 2
                (20, 30),   # 소교차로 1
                (60, 30),   # 소교차로 2
                (20, 70),   # 소교차로 3
                (60, 70),   # 소교차로 4
            ],
            'curve': [
                (85, 45),   # 커브 1
                (90, 40),   # 커브 2
                (85, 55),   # 커브 3
                (90, 60),   # 커브 4
            ],
            'endpoint': [
                (0, 50),    # 시작점
                (100, 50),  # 끝점 1
                (20, 0),    # 끝점 2
                (60, 0),    # 끝점 3
                (20, 100),  # 끝점 4
                (60, 100),  # 끝점 5
                (100, 30),  # 끝점 6
                (100, 70),  # 끝점 7
            ]
        }
        
        logger.info(f"✅ 테스트 데이터 로드 완료:")
        logger.info(f"  - 스켈레톤 포인트: {len(self.skeleton_coords)}개")
        logger.info(f"  - 교차점: {len(self.analysis_points['intersection'])}개")
        logger.info(f"  - 커브점: {len(self.analysis_points['curve'])}개")
        logger.info(f"  - 끝점: {len(self.analysis_points['endpoint'])}개")
        
    def get_road_union(self):
        """고도화된 도로망 폴리곤 생성"""
        if self._road_union_cache is not None:
            return self._road_union_cache
        
        if not self.skeleton_coords:
            logger.error("스켈레톤 데이터가 없습니다")
            return None
        
        try:
            # 스켈레톤으로부터 LineString 생성
            skeleton_line = LineString(self.skeleton_coords)
            
            # 도로 폭을 고려한 버퍼 적용
            road_buffer = skeleton_line.buffer(8.0)  # 8m 버퍼
            
            # unary_union으로 도로 폴리곤 생성
            self._road_union_cache = unary_union([road_buffer])
            
            logger.info(f"✅ 도로망 폴리곤 생성 완료: {self._road_union_cache.geom_type}")
            return self._road_union_cache
            
        except Exception as e:
            logger.error(f"도로망 폴리곤 생성 실패: {e}")
            return None
    
    def _ensure_skeleton_graph(self):
        """스켈레톤 그래프 생성/캐시"""
        if not self.skeleton_coords:
            return None, None, None
        
        skeleton_id = id(self.skeleton_coords)
        if self._graph_cache['skeleton_id'] == skeleton_id:
            return (self._graph_cache['graph'],
                    self._graph_cache['kdtree'],
                    self._graph_cache['coords'])
        
        try:
            # NetworkX 그래프 생성
            G = nx.Graph()
            coords_array = np.array(self.skeleton_coords)
            
            # 노드 추가
            for i, coord in enumerate(self.skeleton_coords):
                G.add_node(i, pos=coord)
            
            # KDTree 생성
            kdtree = KDTree(coords_array)
            
            # 엣지 추가 (20m 반지름)
            for i, coord in enumerate(self.skeleton_coords):
                indices = kdtree.query_radius([coord], r=20.0)[0]
                for j in indices:
                    if i != j:
                        dist = np.linalg.norm(np.array(coord) - np.array(self.skeleton_coords[j]))
                        G.add_edge(i, j, weight=dist)
            
            # 캐시 저장
            self._graph_cache = {
                'skeleton_id': skeleton_id,
                'graph': G,
                'kdtree': kdtree,
                'coords': coords_array
            }
            
            logger.info(f"✅ 스켈레톤 그래프 생성: {G.number_of_nodes()}개 노드, {G.number_of_edges()}개 엣지")
            return G, kdtree, coords_array
            
        except Exception as e:
            logger.error(f"스켈레톤 그래프 생성 실패: {e}")
            return None, None, None
    
    def calculate_and_display_distances(self):
        """고도화된 스켈레톤 연결성 기반 거리 계산"""
        logger.info("🔄 스켈레톤 연결성 기반 거리 분석 시작...")
        
        # 모든 분석 포인트 수집
        all_points = []
        for category in ['intersection', 'curve', 'endpoint']:
            for point in self.analysis_points[category]:
                all_points.append((float(point[0]), float(point[1])))
        
        if len(all_points) < 2:
            logger.warning("분석할 포인트가 부족합니다")
            return
        
        logger.info(f"📊 분석 포인트 총 {len(all_points)}개")
        
        # 스켈레톤 그래프 생성
        graph, kdtree, skeleton_coords = self._ensure_skeleton_graph()
        
        if graph is None:
            logger.error("스켈레톤 그래프 생성 실패")
            return
        
        # 도로망 폴리곤 생성
        road_union_buf = self.get_road_union()
        
        # 각 분석 포인트를 가장 가까운 스켈레톤 노드에 매핑
        point_to_skeleton = {}
        for i, point in enumerate(all_points):
            distances, indices = kdtree.query([point], k=1)
            if distances[0][0] < 25.0:  # 25m 이내
                point_to_skeleton[i] = indices[0][0]
        
        logger.info(f"🔗 스켈레톤 매핑: {len(point_to_skeleton)}개 포인트")
        
        # 연결성 검사 및 거리 계산
        network_connections = []
        connection_distances = []
        
        for i in range(len(all_points)):
            for j in range(i + 1, len(all_points)):
                if i not in point_to_skeleton or j not in point_to_skeleton:
                    continue
                
                # 유클리드 거리 계산
                euclidean_dist = np.linalg.norm(
                    np.array(all_points[i]) - np.array(all_points[j])
                )
                
                # 거리 필터링 (15m~300m)
                if euclidean_dist < 15.0 or euclidean_dist > 300.0:
                    continue
                
                try:
                    # 스켈레톤 그래프에서 경로 찾기
                    path = nx.shortest_path(
                        graph, 
                        point_to_skeleton[i], 
                        point_to_skeleton[j], 
                        weight='weight'
                    )
                    
                    # 중간에 다른 분석 포인트가 있는지 확인
                    path_has_other_points = False
                    if len(path) > 2:
                        path_line = LineString([skeleton_coords[node] for node in path])
                        
                        for k in range(len(all_points)):
                            if k != i and k != j:
                                other_point = all_points[k]
                                point_geom = Point(other_point)
                                distance_to_path = path_line.distance(point_geom)
                                
                                if distance_to_path < 20.0:
                                    dist_to_start = np.linalg.norm(np.array(other_point) - np.array(all_points[i]))
                                    dist_to_end = np.linalg.norm(np.array(other_point) - np.array(all_points[j]))
                                    
                                    if dist_to_start > 30.0 and dist_to_end > 30.0:
                                        path_has_other_points = True
                                        break
                    
                    if path_has_other_points:
                        continue
                    
                    # 스켈레톤 경로 길이 계산
                    skeleton_path_length = sum(
                        graph[path[k]][path[k+1]]['weight'] 
                        for k in range(len(path) - 1)
                    )
                    
                    # 경로 길이가 너무 우회하는지 확인
                    if skeleton_path_length > euclidean_dist * 2.0:
                        continue
                    
                    # 🔍 고도화된 도로망 폴리곤 검증
                    road_validation_passed = True
                    inside_ratio = 1.0
                    
                    if road_union_buf is not None:
                        line = LineString([all_points[i], all_points[j]])
                        total_len = line.length
                        
                        if total_len > 0:
                            try:
                                intersection_geom = line.intersection(road_union_buf)
                                
                                if hasattr(intersection_geom, 'length'):
                                    inside_len = intersection_geom.length
                                elif hasattr(intersection_geom, 'geoms'):
                                    inside_len = sum(geom.length for geom in intersection_geom.geoms if hasattr(geom, 'length'))
                                else:
                                    inside_len = 0.0
                                
                                inside_ratio = inside_len / total_len
                                
                                if inside_ratio < 0.5:
                                    road_validation_passed = False
                                    
                            except Exception as e:
                                logger.debug(f"도로망 교집합 계산 실패: {e}")
                                road_validation_passed = False
                    
                    if not road_validation_passed:
                        continue
                    
                    # 모든 조건을 만족하면 연결 추가
                    connection_info = {
                        'point_indices': (i, j),
                        'distance': euclidean_dist,
                        'skeleton_path_length': skeleton_path_length,
                        'road_ratio': inside_ratio
                    }
                    network_connections.append((i, j, euclidean_dist, connection_info))
                    connection_distances.append(euclidean_dist)
                    
                except nx.NetworkXNoPath:
                    continue
        
        # 📊 결과 분석 및 표시
        logger.info(f"🔗 스켈레톤 연결성 검사 완료: {len(network_connections)}개 연결 생성")
        
        if connection_distances:
            min_d = min(connection_distances)
            max_d = max(connection_distances)
            avg_d = sum(connection_distances) / len(connection_distances)
            
            # 거리별 분포
            under_50m = sum(1 for d in connection_distances if d <= 50.0)
            range_50_100m = sum(1 for d in connection_distances if 50.0 < d <= 100.0)
            over_100m = sum(1 for d in connection_distances if d > 100.0)
            
            # 도로망 검증 통계
            road_ratios = []
            for conn in network_connections:
                if len(conn) >= 4 and 'road_ratio' in conn[3]:
                    road_ratios.append(conn[3]['road_ratio'])
            
            avg_road_ratio = sum(road_ratios) / len(road_ratios) if road_ratios else 1.0
            min_road_ratio = min(road_ratios) if road_ratios else 1.0
            
            logger.info("📊 === 거리 분석 결과 ===")
            logger.info(f"🔗 연결 개수: {len(network_connections)}개")
            logger.info(f"📏 거리 범위: {min_d:.1f}m ~ {max_d:.1f}m (평균: {avg_d:.1f}m)")
            logger.info(f"📊 거리 분포:")
            logger.info(f"  - 50m 이하: {under_50m}개")
            logger.info(f"  - 50-100m: {range_50_100m}개")
            logger.info(f"  - 100m 초과: {over_100m}개")
            logger.info(f"🛣️ 도로망 검증:")
            logger.info(f"  - 평균 도로망 비율: {avg_road_ratio:.1%}")
            logger.info(f"  - 최소 도로망 비율: {min_road_ratio:.1%}")
            logger.info(f"  - 검증 임계값: 50%")
            
            # 상세 연결 정보
            logger.info("\n📋 === 상세 연결 정보 ===")
            for i, (idx1, idx2, dist, info) in enumerate(network_connections, 1):
                p1 = all_points[idx1]
                p2 = all_points[idx2]
                skeleton_dist = info.get('skeleton_path_length', dist)
                road_ratio = info.get('road_ratio', 1.0)
                
                logger.info(f"{i:2d}. ({p1[0]:5.1f},{p1[1]:5.1f}) → ({p2[0]:5.1f},{p2[1]:5.1f})")
                logger.info(f"     거리: {dist:.1f}m, 스켈레톤: {skeleton_dist:.1f}m, 도로망: {road_ratio:.1%}")
        
        else:
            logger.info("🔗 표시할 연결이 없습니다 (15-300m 범위)")
        
        return network_connections, connection_distances
    
    def run_comprehensive_analysis(self):
        """종합 분석 실행"""
        logger.info("🚀 Process3 핵심 기능 종합 분석 시작")
        logger.info("=" * 60)
        
        # 1. 데이터 로드
        self.load_test_data()
        
        # 2. 도로망 폴리곤 생성
        road_union = self.get_road_union()
        if road_union is None:
            logger.error("도로망 폴리곤 생성 실패")
            return False
        
        # 3. 스켈레톤 그래프 생성
        graph, kdtree, coords = self._ensure_skeleton_graph()
        if graph is None:
            logger.error("스켈레톤 그래프 생성 실패")
            return False
        
        # 4. 거리 분석 실행
        connections, distances = self.calculate_and_display_distances()
        
        # 5. 결과 저장
        self.save_analysis_results(connections, distances)
        
        logger.info("=" * 60)
        logger.info("✅ Process3 핵심 기능 종합 분석 완료")
        
        return True
    
    def save_analysis_results(self, connections, distances):
        """분석 결과 저장"""
        try:
            results = {
                'timestamp': str(np.datetime64('now')),
                'total_connections': len(connections),
                'distance_stats': {
                    'min': float(min(distances)) if distances else 0,
                    'max': float(max(distances)) if distances else 0,
                    'avg': float(sum(distances) / len(distances)) if distances else 0,
                    'count': len(distances)
                },
                'connections': []
            }
            
            for i, (idx1, idx2, dist, info) in enumerate(connections):
                results['connections'].append({
                    'id': i + 1,
                    'point1_index': idx1,
                    'point2_index': idx2,
                    'distance': float(dist),
                    'skeleton_path_length': float(info.get('skeleton_path_length', dist)),
                    'road_ratio': float(info.get('road_ratio', 1.0))
                })
            
            # JSON 파일로 저장
            output_file = Path('process3_analysis_results.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 분석 결과 저장: {output_file}")
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")

def main():
    """메인 함수"""
    print("🔍 Process3 핵심 기능 독립 실행")
    print("스켈레톤 연결성 기반 거리 분석 시스템")
    print("=" * 50)
    
    # Process3 핵심 기능 인스턴스 생성
    analyzer = Process3Core()
    
    # 종합 분석 실행
    success = analyzer.run_comprehensive_analysis()
    
    if success:
        print("\n🎉 분석 완료! 결과는 process3_analysis_results.json 파일에 저장되었습니다.")
    else:
        print("\n❌ 분석 실패!")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 