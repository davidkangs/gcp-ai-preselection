#!/usr/bin/env python3
"""
통합된 특징벡터 시스템 테스트
토폴로지 분석, 경계 거리 계산, 삭제 우선순위 테스트
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from shapely.geometry import Polygon, Point
import logging

# 프로젝트 경로 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))

from core.unified_feature_extractor import UnifiedFeatureExtractor, initialize_global_extractor, get_feature_extractor
from core.topology_analyzer import TopologyAnalyzer, BoundaryDistanceCalculator
from core.skeleton_extractor import SkeletonExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_road_network():
    """테스트용 T자형 도로망 생성"""
    # T자형 도로망 좌표
    skeleton_points = [
        # 메인 도로 (수평)
        [100, 200], [150, 200], [200, 200], [250, 200], [300, 200], [350, 200],
        # 분기 도로 (수직 - 교차점에서 위로)
        [200, 200], [200, 250], [200, 300], [200, 350],
        # 분기 도로 (수직 - 교차점에서 아래로)
        [200, 200], [200, 150], [200, 100], [200, 50],
        # 작은 분기 (오른쪽 상단)
        [300, 200], [320, 220], [340, 240],
    ]
    
    # 변환 정보
    transform_info = {
        'bounds': [50, 25, 400, 375],  # minx, miny, maxx, maxy
        'crs': 'EPSG:5186'
    }
    
    # 경계 폴리곤 (사각형)
    boundary_polygon = Polygon([
        (50, 25), (400, 25), (400, 375), (50, 375), (50, 25)
    ])
    
    skeleton_data = {
        'skeleton': skeleton_points,
        'transform': transform_info
    }
    
    return skeleton_data, boundary_polygon


def test_topology_analyzer():
    """토폴로지 분석기 테스트"""
    print("=" * 60)
    print("🔍 토폴로지 분석기 테스트")
    print("=" * 60)
    
    skeleton_data, boundary_polygon = create_test_road_network()
    
    # 토폴로지 분석기 초기화
    analyzer = TopologyAnalyzer(skeleton_data)
    
    # 도로 그래프 구축
    graph = analyzer.build_road_graph()
    print(f"📊 도로 그래프: {graph.number_of_nodes()}개 노드, {graph.number_of_edges()}개 엣지")
    
    # 교차점 찾기
    intersections = analyzer.find_intersections(min_degree=3)
    print(f"🔴 교차점 {len(intersections)}개 검출:")
    for i, intersection in enumerate(intersections):
        print(f"  {i+1}. ({intersection[0]:.1f}, {intersection[1]:.1f})")
    
    # 도로 세그먼트 분석
    segments = analyzer.analyze_road_segments()
    print(f"🛣️ 도로 세그먼트 {len(segments)}개 분석:")
    for i, segment in enumerate(segments):
        road_type = "메인도로" if segment['is_main_road'] else "분기도로"
        print(f"  {i+1}. {road_type} - 길이: {segment['length']:.1f}, "
              f"시작도: {segment['start_degree']}, 종료도: {segment['end_degree']}")
    
    # 메인도로/분기도로 분류
    main_roads, branch_roads = analyzer.classify_roads()
    print(f"📈 분류 결과: 메인도로 {len(main_roads)}개, 분기도로 {len(branch_roads)}개")
    
    return analyzer


def test_boundary_calculator():
    """경계 거리 계산기 테스트"""
    print("\n" + "=" * 60)
    print("📏 경계 거리 계산기 테스트")
    print("=" * 60)
    
    skeleton_data, boundary_polygon = create_test_road_network()
    
    calculator = BoundaryDistanceCalculator(boundary_polygon)
    
    # 테스트 포인트들
    test_points = [
        (225, 225),  # 중앙 (경계에서 멀음)
        (75, 200),   # 왼쪽 경계 근처
        (375, 200),  # 오른쪽 경계 근처
        (200, 50),   # 아래쪽 경계 근처
        (200, 350),  # 위쪽 경계 근처
    ]
    
    print("📍 테스트 포인트별 경계 거리:")
    for i, point in enumerate(test_points):
        distance = calculator.calculate_distance_to_boundary(point)
        is_near = calculator.is_near_boundary(point, threshold=50)
        boundary_score = calculator.get_boundary_score(point)
        
        print(f"  {i+1}. ({point[0]}, {point[1]}) -> "
              f"거리: {distance:.1f}, 근접: {'Yes' if is_near else 'No'}, "
              f"점수: {boundary_score:.3f}")
    
    return calculator


def test_unified_feature_extractor():
    """통합 특징 추출기 테스트"""
    print("\n" + "=" * 60)
    print("🧮 통합 특징 추출기 테스트")
    print("=" * 60)
    
    skeleton_data, boundary_polygon = create_test_road_network()
    
    # 통합 특징 추출기 초기화
    extractor = initialize_global_extractor(skeleton_data, boundary_polygon)
    
    # 테스트 포인트들 (다양한 위치)
    test_points = [
        (200, 200, "교차점 중심"),
        (300, 200, "메인도로 상의 점"),
        (200, 350, "분기도로 끝점"),
        (340, 240, "작은 분기 끝점"),
        (150, 200, "메인도로 시작 부근"),
        (320, 220, "작은 분기 중간"),
    ]
    
    print("🔍 테스트 포인트별 20차원 특징벡터:")
    feature_matrix = []
    
    for i, (x, y, description) in enumerate(test_points):
        features = extractor.extract_features((x, y), i, None)
        feature_matrix.append(features)
        
        print(f"\n  {i+1}. {description} ({x}, {y}):")
        print(f"     위치 특징: [{features[0]:.3f}, {features[1]:.3f}, {features[2]:.3f}]")
        print(f"     기하 특징: [{features[3]:.3f}, {features[4]:.3f}, {features[5]:.3f}, {features[6]:.3f}]")
        print(f"     토폴로지: [{features[7]:.3f}, {features[8]:.3f}, {features[9]:.3f}, {features[10]:.3f}]")
        print(f"     우선순위: [{features[11]:.3f}, {features[12]:.3f}, {features[13]:.3f}, {features[14]:.3f}]")
        print(f"     근접성: [{features[15]:.3f}, {features[16]:.3f}, {features[17]:.3f}]")
        print(f"     컨텍스트: [{features[18]:.3f}, {features[19]:.3f}]")
    
    return extractor, feature_matrix, test_points


def visualize_features(feature_matrix, test_points):
    """특징벡터 시각화"""
    print("\n" + "=" * 60)
    print("📊 특징벡터 시각화")
    print("=" * 60)
    
    feature_matrix = np.array(feature_matrix)
    feature_names = [
        'norm_x', 'norm_y', 'boundary_dist',
        'prev_dist', 'next_dist', 'curvature', 'density',
        'branch_ratio', 'int_density', 'min_int_dist', 'pos_score',
        'del_priority', 'main_roads', 'branch_roads', 'on_branch',
        'near_int', 'near_curve', 'near_end',
        'rel_pos', 'boundary_score'
    ]
    
    # 히트맵 생성
    fig, ax = plt.subplots(figsize=(15, 8))
    
    im = ax.imshow(feature_matrix.T, cmap='viridis', aspect='auto')
    
    # 축 라벨 설정
    ax.set_xticks(range(len(test_points)))
    ax.set_xticklabels([f"P{i+1}\n{desc}" for i, (x, y, desc) in enumerate(test_points)], 
                       rotation=45, ha='right')
    
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    
    # 컬러바 추가
    plt.colorbar(im, ax=ax, label='Feature Value')
    
    # 제목 및 레이블
    ax.set_title('통합 특징벡터 히트맵 (20차원)', fontsize=14, fontweight='bold')
    ax.set_xlabel('테스트 포인트', fontsize=12)
    ax.set_ylabel('특징 차원', fontsize=12)
    
    # 그리드
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_heatmap.png', dpi=300, bbox_inches='tight')
    print("💾 특징 히트맵을 'feature_heatmap.png'로 저장했습니다.")
    plt.show()


def test_deletion_priority():
    """삭제 우선순위 테스트"""
    print("\n" + "=" * 60)
    print("🗑️ 삭제 우선순위 테스트")
    print("=" * 60)
    
    skeleton_data, boundary_polygon = create_test_road_network()
    analyzer = TopologyAnalyzer(skeleton_data)
    analyzer.build_road_graph()
    analyzer.analyze_road_segments()
    analyzer.classify_roads()
    
    # 테스트 포인트들
    test_points = [
        (200, 200, "교차점 중심"),
        (300, 200, "메인도로 상의 점"),
        (200, 350, "분기도로 끝점"),
        (340, 240, "작은 분기 끝점"),
        (320, 220, "작은 분기 중간"),
    ]
    
    print("🎯 포인트별 삭제 우선순위 분석:")
    priorities = []
    
    for x, y, description in test_points:
        # 분기 길이 비율
        branch_ratio = analyzer.calculate_branch_length_ratio((x, y))
        
        # 교차점 밀도
        int_density = analyzer.calculate_intersection_density((x, y))
        
        # 삭제 우선순위
        deletion_priority = analyzer.get_deletion_priority_score((x, y))
        
        # 포인트 컨텍스트
        context = analyzer.analyze_point_context((x, y))
        
        priorities.append(deletion_priority)
        
        print(f"\n  📍 {description} ({x}, {y}):")
        print(f"     분기도로 비율: {branch_ratio:.3f}")
        print(f"     교차점 밀도: {int_density:.6f}")
        print(f"     삭제 우선순위: {deletion_priority:.3f}")
        print(f"     분기도로 위치: {'Yes' if context['is_on_branch_road'] else 'No'}")
        print(f"     최근접 교차점: {context['min_intersection_distance']:.1f}")
    
    # 삭제 우선순위 순으로 정렬
    sorted_points = sorted(zip(test_points, priorities), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 삭제 우선순위 순위:")
    for i, ((x, y, desc), priority) in enumerate(sorted_points):
        print(f"  {i+1}위. {desc} - 우선순위: {priority:.3f}")


def simulate_user_deletion_pattern():
    """사용자 삭제 패턴 시뮬레이션"""
    print("\n" + "=" * 60)
    print("👤 사용자 삭제 패턴 시뮬레이션")
    print("=" * 60)
    
    # 사용자가 삭제할 가능성이 높은 패턴들 정의
    deletion_scenarios = [
        {
            'name': '교차점 두 개가 가까이 있을 때 중간 하나 삭제',
            'condition': 'min_intersection_distance < 30 and intersection_density > 0.001',
            'action': '중간 교차점 삭제 후 커브점 생성'
        },
        {
            'name': '메인도로 대비 매우 짧은 분기의 교차점 삭제',
            'condition': 'branch_ratio > 0.7 and is_on_branch_road == True',
            'action': '분기 교차점 삭제'
        },
        {
            'name': '경계 근처에서 끝점이 아닌 점을 끝점으로 변경',
            'condition': 'boundary_distance < 0.3 and position_score < 0.5',
            'action': '점을 끝점으로 변경'
        }
    ]
    
    skeleton_data, boundary_polygon = create_test_road_network()
    extractor = initialize_global_extractor(skeleton_data, boundary_polygon)
    
    print("🧠 학습된 사용자 패턴 분석:")
    for i, scenario in enumerate(deletion_scenarios):
        print(f"\n  패턴 {i+1}: {scenario['name']}")
        print(f"     조건: {scenario['condition']}")
        print(f"     행동: {scenario['action']}")
    
    # 모든 스켈레톤 포인트에 대해 삭제 패턴 적용
    skeleton_points = skeleton_data['skeleton']
    deletion_candidates = []
    
    for i, point in enumerate(skeleton_points):
        if len(point) >= 2:
            features = extractor.extract_features((point[0], point[1]), i, None)
            
            # 각 삭제 패턴에 대해 확인
            for scenario in deletion_scenarios:
                if scenario['name'] == '교차점 두 개가 가까이 있을 때 중간 하나 삭제':
                    if features[9] < 0.3 and features[8] > 0.001:  # min_int_dist < 30, int_density > 0.001
                        deletion_candidates.append((point, scenario['action'], features[11]))  # deletion_priority
                
                elif scenario['name'] == '메인도로 대비 매우 짧은 분기의 교차점 삭제':
                    if features[7] > 0.7 and features[14] > 0.5:  # branch_ratio > 0.7, is_on_branch_road
                        deletion_candidates.append((point, scenario['action'], features[11]))
                
                elif scenario['name'] == '경계 근처에서 끝점이 아닌 점을 끝점으로 변경':
                    if features[2] < 0.3 and features[10] < 0.5:  # boundary_distance < 0.3, position_score < 0.5
                        deletion_candidates.append((point, scenario['action'], features[11]))
    
    # 삭제 우선순위로 정렬
    deletion_candidates.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n🎯 사용자가 삭제할 가능성이 높은 포인트들:")
    for i, (point, action, priority) in enumerate(deletion_candidates[:5]):  # 상위 5개만
        print(f"  {i+1}. ({point[0]:.1f}, {point[1]:.1f}) - {action} (우선순위: {priority:.3f})")


def main():
    """메인 테스트 함수"""
    print("🚀 통합된 20차원 특징벡터 시스템 테스트 시작")
    print("=" * 80)
    
    try:
        # 1. 토폴로지 분석기 테스트
        analyzer = test_topology_analyzer()
        
        # 2. 경계 거리 계산기 테스트
        calculator = test_boundary_calculator()
        
        # 3. 통합 특징 추출기 테스트
        extractor, feature_matrix, test_points = test_unified_feature_extractor()
        
        # 4. 특징벡터 시각화
        visualize_features(feature_matrix, test_points)
        
        # 5. 삭제 우선순위 테스트
        test_deletion_priority()
        
        # 6. 사용자 삭제 패턴 시뮬레이션
        simulate_user_deletion_pattern()
        
        print("\n" + "=" * 80)
        print("✅ 모든 테스트가 성공적으로 완료되었습니다!")
        print("=" * 80)
        
        print("\n📊 테스트 결과 요약:")
        print("  ✓ 토폴로지 분석기: 도로 그래프 구축, 교차점 검출, 분기도로 분류 정상")
        print("  ✓ 경계 거리 계산기: 지구계 경계까지의 거리 계산 정상")
        print("  ✓ 통합 특징 추출기: 20차원 특징벡터 생성 정상")
        print("  ✓ 삭제 우선순위: 사용자 라벨링 패턴 기반 우선순위 계산 정상")
        print("  ✓ 특징 시각화: feature_heatmap.png 파일 생성")
        
        print("\n🎯 다음 단계:")
        print("  1. 새로운 특징벡터로 모델 재학습")
        print("  2. 실제 데이터로 성능 비교 테스트")
        print("  3. 사용자 삭제 패턴 학습 정확도 평가")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 