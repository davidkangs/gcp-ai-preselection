#!/usr/bin/env python3
"""
고도화된 거리 분석 시스템 테스트
point_sample 방식을 활용한 종합 테스트
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from process3_inference import InferenceTool
from src.distance_analysis import AdvancedDistanceCalculator

def create_test_data():
    """테스트용 데이터 생성"""
    # 테스트용 점 데이터
    points_data = {
        'intersection': [
            (100, 100),
            (200, 200),
            (300, 150),
            (150, 250)
        ],
        'curve': [
            (130, 120),
            (180, 180),
            (250, 170),
            (170, 220)
        ],
        'endpoint': [
            (80, 90),
            (320, 160),
            (190, 270)
        ]
    }
    
    # 테스트용 스켈레톤 점들 (도로 중심선)
    skeleton_points = []
    for i in range(0, 400, 10):
        x = i
        y = 100 + 50 * np.sin(i / 50)  # 곡선 형태의 도로
        skeleton_points.append((x, y))
    
    # 추가 스켈레톤 (교차로)
    for i in range(100, 200, 10):
        skeleton_points.append((150, i))
    
    return points_data, skeleton_points

def test_advanced_distance_calculator():
    """고도화된 거리 계산기 테스트"""
    print("=== 고도화된 거리 계산기 테스트 ===")
    
    # 테스트 데이터 생성
    points_data, skeleton_points = create_test_data()
    
    # 거리 계산기 초기화
    calculator = AdvancedDistanceCalculator(
        skeleton_points=skeleton_points,
        road_polygons=[],  # 폴리곤 없이 테스트
        max_distance=300.0,
        min_distance=15.0
    )
    
    # 분석 실행
    result = calculator.calculate_optimal_distances(points_data)
    
    # 결과 출력
    print(f"총 연결: {len(result['connections'])}개")
    print(f"통계: {result['statistics']}")
    
    # 상위 5개 연결 출력
    print("\n=== 상위 5개 연결 ===")
    for i, conn in enumerate(result['connections'][:5]):
        print(f"{i+1}. {conn['category1']} -> {conn['category2']}: "
              f"{conn['distance']:.1f}m (우선순위: {conn['priority']:.3f})")
    
    # Canvas 표시용 데이터 테스트
    display_data = calculator.get_canvas_display_data()
    print(f"\nCanvas 표시 데이터: {len(display_data)}개")
    
    # 통계 텍스트 테스트
    stats_text = calculator.get_statistics_text()
    print(f"\n통계 텍스트:\n{stats_text}")
    
    return result

def test_with_inference_tool():
    """InferenceTool과 통합 테스트"""
    print("\n=== InferenceTool 통합 테스트 ===")
    
    app = QApplication(sys.argv)
    
    # 인퍼런스 도구 실행
    tool = InferenceTool()
    
    # 테스트용 점들 설정
    points_data, skeleton_points = create_test_data()
    
    # Canvas에 점들 설정
    tool.canvas_widget.canvas.points = points_data
    tool.canvas_widget.canvas.skeleton = skeleton_points
    
    # 거리 분석 모드 활성화
    tool.distance_analysis_active = True
    
    # 거리 분석 실행
    try:
        tool.calculate_and_update_visual_distances()
        print("거리 분석 성공!")
        
        # 연결 정보 확인
        if hasattr(tool.canvas_widget.canvas, 'distance_connections'):
            connections = tool.canvas_widget.canvas.distance_connections
            print(f"Canvas 연결: {len(connections)}개")
            
            # 상위 3개 연결 출력
            for i, conn in enumerate(connections[:3]):
                print(f"  {i+1}. {conn['distance']:.1f}m ({conn['categories']})")
        
    except Exception as e:
        print(f"거리 분석 실패: {e}")
        import traceback
        traceback.print_exc()
    
    return tool

def test_network_connectivity():
    """네트워크 연결성 분석 테스트"""
    print("\n=== 네트워크 연결성 분석 테스트 ===")
    
    from src.distance_analysis.network_connectivity import NetworkConnectivityAnalyzer
    
    points_data, skeleton_points = create_test_data()
    
    # 점 메타데이터 준비
    points_with_metadata = []
    for category, points in points_data.items():
        for idx, (x, y) in enumerate(points):
            points_with_metadata.append((x, y, category, idx))
    
    # 네트워크 분석기 초기화
    analyzer = NetworkConnectivityAnalyzer(
        skeleton_points=skeleton_points,
        road_polygons=[],
        max_distance=300.0,
        min_distance=15.0
    )
    
    # 그래프 생성
    graph = analyzer.build_road_graph(points_with_metadata)
    
    # 연결 정보 확인
    connections = analyzer.get_connected_pairs()
    print(f"네트워크 연결: {len(connections)}개")
    
    # 통계 정보
    stats = analyzer.get_network_statistics()
    print(f"네트워크 통계: {stats}")
    
    # 중심성 계산
    centrality = analyzer.get_node_centrality()
    print(f"중심성 계산 완료: {len(centrality.get('degree', {}))}개 노드")
    
    return analyzer

def test_visual_connectivity():
    """시각적 연결성 분석 테스트"""
    print("\n=== 시각적 연결성 분석 테스트 ===")
    
    from src.distance_analysis.visual_connectivity import VisualConnectivityChecker
    
    points_data, skeleton_points = create_test_data()
    
    # 점 메타데이터 준비
    points_with_metadata = []
    for category, points in points_data.items():
        for idx, (x, y) in enumerate(points):
            points_with_metadata.append((x, y, category, idx))
    
    # 시각적 연결성 체커 초기화
    checker = VisualConnectivityChecker(
        skeleton_points=skeleton_points,
        road_polygons=[],
        max_distance=300.0,
        min_distance=15.0
    )
    
    # 시각적 연결 확인
    visual_connections = checker.get_visual_connections(points_with_metadata)
    print(f"시각적 연결: {len(visual_connections)}개")
    
    # 몇 개 연결의 시각적 점수 확인
    for i, (idx1, idx2, distance) in enumerate(visual_connections[:3]):
        p1 = points_with_metadata[idx1]
        p2 = points_with_metadata[idx2]
        
        score = checker.get_visibility_score((p1[0], p1[1]), (p2[0], p2[1]))
        print(f"  {i+1}. {p1[2]} -> {p2[2]}: {distance:.1f}m (점수: {score:.3f})")
    
    return checker

def test_importance_scorer():
    """중요도 점수 계산 테스트"""
    print("\n=== 중요도 점수 계산 테스트 ===")
    
    from src.distance_analysis.importance_scorer import ImportanceScorer
    
    points_data, skeleton_points = create_test_data()
    
    # 중요도 계산기 초기화
    scorer = ImportanceScorer(
        skeleton_points=skeleton_points,
        road_polygons=[]
    )
    
    # 각 점의 중요도 계산
    print("점별 중요도:")
    for category, points in points_data.items():
        for idx, (x, y) in enumerate(points):
            importance = scorer.calculate_point_importance((x, y), category)
            print(f"  {category}[{idx}]: {importance:.3f}")
    
    # 연결 우선순위 계산
    test_connections = [
        (0, 1, 100, (100, 100), (200, 200), 'intersection', 'intersection'),
        (0, 2, 150, (100, 100), (130, 120), 'intersection', 'curve'),
        (1, 3, 200, (200, 200), (80, 90), 'intersection', 'endpoint'),
    ]
    
    ranked_connections = scorer.rank_connections(test_connections)
    
    print("\n연결 우선순위:")
    for i, conn in enumerate(ranked_connections):
        print(f"  {i+1}. {conn['category1']} -> {conn['category2']}: "
              f"{conn['distance']:.1f}m (우선순위: {conn['priority']:.3f})")
    
    return scorer

def main():
    """메인 테스트 실행"""
    print("고도화된 거리 분석 시스템 종합 테스트")
    print("=" * 50)
    
    # 1. 고도화된 거리 계산기 테스트
    result = test_advanced_distance_calculator()
    
    # 2. 네트워크 연결성 테스트
    analyzer = test_network_connectivity()
    
    # 3. 시각적 연결성 테스트
    checker = test_visual_connectivity()
    
    # 4. 중요도 점수 테스트
    scorer = test_importance_scorer()
    
    # 5. InferenceTool 통합 테스트
    tool = test_with_inference_tool()
    
    print("\n" + "=" * 50)
    print("모든 테스트 완료!")
    
    # GUI 실행 (선택사항)
    if '--gui' in sys.argv:
        print("GUI 모드로 실행...")
        tool.show()
        sys.exit(QApplication.instance().exec_())

if __name__ == '__main__':
    main() 