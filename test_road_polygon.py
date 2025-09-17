#!/usr/bin/env python3
"""
도로망 폴리곤 검증 시스템 테스트 스크립트
고도화된 unary_union 기반 도로망 생성과 50% 임계값 검증 테스트
"""

import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class RoadPolygonTester:
    def __init__(self):
        self.skeleton_coords = None
        self.analysis_points = []
        self.road_union = None
        self.graph = None
        self.kdtree = None
        
    def create_test_data(self):
        """테스트용 스켈레톤과 분석 포인트 생성"""
        # 🛣️ 테스트용 스켈레톤 (도로 형태)
        self.skeleton_coords = [
            (0, 0), (10, 0), (20, 0), (30, 0), (40, 0),      # 수평 도로
            (40, 0), (40, 10), (40, 20), (40, 30), (40, 40), # 수직 도로  
            (40, 40), (30, 40), (20, 40), (10, 40), (0, 40), # 수평 도로 (위쪽)
            (0, 40), (0, 30), (0, 20), (0, 10), (0, 0)       # 수직 도로 (왼쪽)
        ]
        
        # 🔍 분석 포인트들 (교차점, 커브 등)
        self.analysis_points = [
            (0, 0),    # 교차점 1
            (40, 0),   # 교차점 2  
            (40, 40),  # 교차점 3
            (0, 40),   # 교차점 4
            (20, 0),   # 중간점 1
            (20, 40),  # 중간점 2
            (0, 20),   # 중간점 3
            (40, 20),  # 중간점 4
        ]
        
        logger.info(f"테스트 데이터 생성: {len(self.skeleton_coords)}개 스켈레톤 포인트, {len(self.analysis_points)}개 분석 포인트")
        
    def create_road_union(self):
        """고도화된 도로망 폴리곤 생성 (unary_union 사용)"""
        if not self.skeleton_coords:
            logger.error("스켈레톤 데이터가 없습니다")
            return None
        
        # 스켈레톤으로부터 LineString 생성
        skeleton_line = LineString(self.skeleton_coords)
        
        # 도로 폭을 고려한 버퍼 적용
        road_buffer = skeleton_line.buffer(5.0)  # 5m 버퍼
        
        # ✅ unary_union으로 도로 폴리곤 생성 (빈 공간 보존)
        self.road_union = unary_union([road_buffer])
        
        logger.info(f"✅ 도로망 폴리곤 생성 완료: {self.road_union.geom_type}")
        return self.road_union
    
    def create_skeleton_graph(self):
        """스켈레톤 그래프 생성"""
        if not self.skeleton_coords:
            return None, None
        
        # NetworkX 그래프 생성
        G = nx.Graph()
        
        # 노드 추가
        for i, coord in enumerate(self.skeleton_coords):
            G.add_node(i, pos=coord)
        
        # 연결된 노드들 간 엣지 추가 (20m 반지름)
        coords_array = np.array(self.skeleton_coords)
        kdtree = KDTree(coords_array)
        
        for i, coord in enumerate(self.skeleton_coords):
            # 20m 반지름 내 이웃 찾기
            indices = kdtree.query_radius([coord], r=20.0)[0]
            for j in indices:
                if i != j:
                    dist = np.linalg.norm(np.array(coord) - np.array(self.skeleton_coords[j]))
                    G.add_edge(i, j, weight=dist)
        
        self.graph = G
        self.kdtree = kdtree
        
        logger.info(f"스켈레톤 그래프 생성: {G.number_of_nodes()}개 노드, {G.number_of_edges()}개 엣지")
        return G, kdtree
    
    def test_road_polygon_validation(self, point1, point2):
        """도로망 폴리곤 검증 테스트"""
        logger.info(f"\n🔍 도로망 검증 테스트: {point1} → {point2}")
        
        # 직선 생성
        line = LineString([point1, point2])
        total_length = line.length
        
        logger.info(f"  직선 길이: {total_length:.1f}m")
        
        if self.road_union is None:
            logger.error("  도로망 폴리곤이 없습니다")
            return False
        
        # 교집합 계산
        try:
            intersection_geom = line.intersection(self.road_union)
            
            # 교집합 결과 타입에 따라 길이 계산
            if hasattr(intersection_geom, 'length'):
                inside_length = intersection_geom.length
            elif hasattr(intersection_geom, 'geoms'):
                inside_length = sum(geom.length for geom in intersection_geom.geoms if hasattr(geom, 'length'))
            else:
                inside_length = 0.0
            
            inside_ratio = inside_length / total_length
            
            logger.info(f"  도로망 내부 길이: {inside_length:.1f}m")
            logger.info(f"  내부 비율: {inside_ratio:.1%}")
            
            # 50% 임계값 검사
            if inside_ratio >= 0.5:
                logger.info(f"  ✅ 검증 통과: {inside_ratio:.1%} ≥ 50%")
                return True
            else:
                logger.info(f"  ❌ 검증 실패: {inside_ratio:.1%} < 50%")
                return False
                
        except Exception as e:
            logger.error(f"  교집합 계산 실패: {e}")
            return False
    
    def run_comprehensive_test(self):
        """종합 테스트 실행"""
        logger.info("🚀 도로망 폴리곤 검증 시스템 종합 테스트 시작")
        
        # 1. 테스트 데이터 생성
        self.create_test_data()
        
        # 2. 도로망 폴리곤 생성
        self.create_road_union()
        
        # 3. 스켈레톤 그래프 생성
        self.create_skeleton_graph()
        
        # 4. 다양한 연결 시나리오 테스트
        test_cases = [
            # (point1, point2, expected_result, description)
            ((0, 0), (20, 0), True, "도로 위 직선 연결"),
            ((0, 0), (40, 0), True, "도로 위 긴 직선 연결"),
            ((0, 0), (40, 40), False, "대각선 연결 (도로 벗어남)"),
            ((0, 0), (20, 20), False, "도로 밖 지역 통과"),
            ((20, 0), (20, 40), True, "수직 도로 연결"),
            ((10, 0), (30, 0), True, "같은 도로 내 연결"),
            ((0, 20), (40, 20), False, "도로 없는 지역 통과"),
        ]
        
        passed_tests = 0
        total_tests = len(test_cases)
        
        for i, (p1, p2, expected, description) in enumerate(test_cases, 1):
            logger.info(f"\n📋 테스트 케이스 {i}/{total_tests}: {description}")
            result = self.test_road_polygon_validation(p1, p2)
            
            if result == expected:
                logger.info(f"  ✅ 테스트 통과")
                passed_tests += 1
            else:
                logger.info(f"  ❌ 테스트 실패 (예상: {expected}, 실제: {result})")
        
        # 5. 결과 요약
        logger.info(f"\n📊 테스트 결과 요약:")
        logger.info(f"  통과: {passed_tests}/{total_tests} ({passed_tests/total_tests:.1%})")
        
        if passed_tests == total_tests:
            logger.info("  🎉 모든 테스트 통과!")
        else:
            logger.info("  ⚠️ 일부 테스트 실패")
        
        return passed_tests, total_tests
    
    def visualize_test_results(self):
        """테스트 결과 시각화"""
        if not self.skeleton_coords or not self.analysis_points:
            logger.error("시각화할 데이터가 없습니다")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 스켈레톤 그리기
        skeleton_x = [p[0] for p in self.skeleton_coords]
        skeleton_y = [p[1] for p in self.skeleton_coords]
        plt.plot(skeleton_x, skeleton_y, 'b-', linewidth=2, label='스켈레톤')
        
        # 도로망 폴리곤 그리기
        if self.road_union:
            if hasattr(self.road_union, 'exterior'):
                x, y = self.road_union.exterior.coords.xy
                plt.fill(x, y, alpha=0.3, color='lightblue', label='도로망 폴리곤')
        
        # 분석 포인트 그리기
        analysis_x = [p[0] for p in self.analysis_points]
        analysis_y = [p[1] for p in self.analysis_points]
        plt.scatter(analysis_x, analysis_y, c='red', s=100, marker='o', label='분석 포인트')
        
        # 테스트 연결선 그리기
        test_connections = [
            ((0, 0), (20, 0), 'green'),      # 통과
            ((0, 0), (40, 40), 'red'),       # 실패
            ((20, 0), (20, 40), 'green'),    # 통과
        ]
        
        for p1, p2, color in test_connections:
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                    color=color, linewidth=2, alpha=0.7)
        
        plt.xlabel('X 좌표')
        plt.ylabel('Y 좌표')
        plt.title('도로망 폴리곤 검증 시스템 테스트 결과')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 그래프 저장
        plt.savefig('road_polygon_test_results.png', dpi=300, bbox_inches='tight')
        logger.info("시각화 결과 저장: road_polygon_test_results.png")
        
        plt.show()

def main():
    """메인 함수"""
    print("🔍 도로망 폴리곤 검증 시스템 테스트")
    print("=" * 50)
    
    # 테스터 인스턴스 생성
    tester = RoadPolygonTester()
    
    # 종합 테스트 실행
    passed, total = tester.run_comprehensive_test()
    
    # 시각화 (선택적)
    try:
        tester.visualize_test_results()
    except ImportError:
        logger.warning("matplotlib을 사용할 수 없어 시각화를 건너뜁니다")
    except Exception as e:
        logger.warning(f"시각화 실패: {e}")
    
    print("\n" + "=" * 50)
    print(f"최종 결과: {passed}/{total} 테스트 통과")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 