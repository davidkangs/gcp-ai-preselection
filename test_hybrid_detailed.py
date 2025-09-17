"""
하이브리드 필터 상세 테스트 - 실제 중복점 제거 확인
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.process3 import FilterManager

def test_hybrid_filter_detailed():
    """하이브리드 필터 상세 테스트"""
    print("🔍 하이브리드 필터 상세 테스트 시작!")
    
    filter_manager = FilterManager(
        dbscan_eps=10.0,    # 10m 클러스터링 (더 작게)
        network_max_dist=30.0,
        road_buffer=2.0
    )
    
    # 실제 중복점이 있는 테스트 데이터
    test_points = [
        (100.0, 100.0),
        (102.0, 102.0),  # 2.8m 거리 (매우 가까움)
        (200.0, 200.0),
        (203.0, 203.0),  # 4.2m 거리 (가까움)
        (300.0, 300.0),
        (500.0, 500.0),  # 충분히 멀리
    ]
    
    # 간단한 직선 스켈레톤
    test_skeleton = [
        [50, 50], [100, 100], [150, 150], [200, 200], [250, 250], [300, 300], [450, 450], [500, 500]
    ]
    
    point_roles = {pt: 'curve' for pt in test_points}
    
    print(f"원본 점: {len(test_points)}개")
    for i, pt in enumerate(test_points):
        print(f"  점 {i+1}: {pt}")
    
    # 하이브리드 필터 직접 호출
    print("\n하이브리드 필터 적용 중...")
    filtered_points = filter_manager.apply_hybrid_filter(
        points=test_points,
        skeleton=test_skeleton,
        point_roles=point_roles
    )
    
    print(f"\n필터링 후: {len(filtered_points)}개")
    for i, pt in enumerate(filtered_points):
        print(f"  점 {i+1}: {pt}")
    
    removed_count = len(test_points) - len(filtered_points)
    print(f"\n제거된 점: {removed_count}개")
    
    # 거리 기반 필터도 테스트
    print("\n거리 기반 필터 테스트...")
    distance_filtered = filter_manager.filter_by_distance(test_points, min_distance=5.0)
    print(f"거리 필터 후: {len(distance_filtered)}개")
    
    return removed_count > 0

def test_remove_duplicate_points():
    """중복점 제거 전체 워크플로우 테스트"""
    print("\n🎯 중복점 제거 워크플로우 테스트")
    
    filter_manager = FilterManager()
    
    # 카테고리별 점 데이터
    points_by_category = {
        'intersection': [
            (100.0, 100.0),
            (102.0, 102.0),  # 중복
        ],
        'curve': [
            (200.0, 200.0),
            (203.0, 203.0),  # 중복
            (400.0, 400.0),
        ],
        'endpoint': [
            (300.0, 300.0),
            (600.0, 600.0),
        ]
    }
    
    test_skeleton = [[i*50, i*50] for i in range(15)]
    
    print("원본 카테고리별 점 개수:")
    for category, points in points_by_category.items():
        print(f"  {category}: {len(points)}개")
    
    # 중복점 제거 실행
    filtered_by_category = filter_manager.remove_duplicate_points(
        points_by_category, 
        test_skeleton,
        distance_threshold=5.0
    )
    
    print("\n필터링 후 카테고리별 점 개수:")
    total_before = sum(len(pts) for pts in points_by_category.values())
    total_after = sum(len(pts) for pts in filtered_by_category.values())
    
    for category, points in filtered_by_category.items():
        print(f"  {category}: {len(points)}개")
    
    print(f"\n전체: {total_before} → {total_after} (제거: {total_before - total_after}개)")
    
    return total_after < total_before

if __name__ == "__main__":
    print("🚀 하이브리드 필터 상세 테스트!\n")
    
    test1_result = test_hybrid_filter_detailed()
    test2_result = test_remove_duplicate_points()
    
    print("\n" + "="*50)
    print("📋 상세 테스트 결과")
    print("="*50)
    print(f"하이브리드 필터 직접 테스트: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"중복점 제거 워크플로우: {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 하이브리드 필터가 완벽하게 통합되었습니다!")
    else:
        print("\n⚠️ 일부 기능에 문제가 있을 수 있습니다.")
    
    print("="*50) 