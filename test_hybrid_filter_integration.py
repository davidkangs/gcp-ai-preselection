"""
하이브리드 필터 통합 테스트 스크립트
리팩토링된 process3 모듈들이 제대로 작동하는지 확인
"""

import sys
from pathlib import Path
import numpy as np

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

from src.process3 import PipelineManager, DataProcessor, FilterManager, AIPredictor, SessionManager

def test_filter_manager():
    """FilterManager 테스트"""
    print("🔍 FilterManager 테스트...")
    
    # 필터 매니저 생성
    filter_manager = FilterManager(
        dbscan_eps=20.0,
        network_max_dist=50.0,
        road_buffer=2.0
    )
    
    # 테스트 데이터 생성
    test_points = [
        (100.0, 100.0),
        (105.0, 105.0),  # 가까운 점 (중복으로 제거될 수 있음)
        (200.0, 200.0),
        (300.0, 300.0),
        (305.0, 305.0),  # 가까운 점
    ]
    
    test_skeleton = [
        [90, 90], [100, 100], [110, 110], [200, 200], [300, 300], [310, 310]
    ]
    
    point_roles = {
        (100.0, 100.0): 'intersection',
        (105.0, 105.0): 'intersection',
        (200.0, 200.0): 'curve',
        (300.0, 300.0): 'endpoint',
        (305.0, 305.0): 'endpoint'
    }
    
    print(f"   원본 점 개수: {len(test_points)}")
    print(f"   스켈레톤 점 개수: {len(test_skeleton)}")
    
    # 하이브리드 필터 적용
    filtered_points = filter_manager.apply_hybrid_filter(
        points=test_points,
        skeleton=test_skeleton,
        point_roles=point_roles
    )
    
    print(f"   필터링 후 점 개수: {len(filtered_points)}")
    print(f"   제거된 점 개수: {len(test_points) - len(filtered_points)}")
    
    # 통계 확인
    stats = filter_manager.get_filter_stats(test_points, filtered_points)
    print(f"   제거율: {stats['removal_rate']:.1%}")
    print(f"   유지율: {stats['retention_rate']:.1%}")
    
    return len(filtered_points) < len(test_points)  # 일부 점이 제거되었는지 확인

def test_data_processor():
    """DataProcessor 테스트"""
    print("\n📊 DataProcessor 테스트...")
    
    data_processor = DataProcessor()
    
    # 테스트 스켈레톤 데이터
    test_skeleton = []
    for i in range(0, 1000, 20):  # 직선 도로 시뮬레이션
        test_skeleton.append([float(i), float(i * 0.5)])
    
    print(f"   테스트 스켈레톤: {len(test_skeleton)}개 점")
    
    # 끝점 검출 테스트
    endpoints = data_processor.detect_heuristic_endpoints(test_skeleton)
    print(f"   검출된 끝점: {len(endpoints)}개")
    
    # 커브 검출 테스트
    curves = data_processor.detect_boundary_based_curves(
        test_skeleton,
        sample_distance=15.0,
        curvature_threshold=0.20,
        road_buffer=3.0,
        cluster_radius=20.0
    )
    print(f"   검출된 커브: {len(curves)}개")
    
    return True

def test_pipeline_manager():
    """PipelineManager 테스트"""
    print("\n🎯 PipelineManager 테스트...")
    
    # 파이프라인 매니저 생성 (모델 경로 없이)
    pipeline_manager = PipelineManager(
        model_path=None,  # AI 없이 테스트
        filter_config={
            'dbscan_eps': 15.0,
            'network_max_dist': 40.0,
            'road_buffer': 2.5
        }
    )
    
    print("   파이프라인 매니저 초기화 완료")
    
    # 설정 확인
    config = pipeline_manager.get_config()
    print(f"   설정: {config}")
    
    # 상태 확인
    status = pipeline_manager.get_status()
    print(f"   AI 모델 로드 상태: {status['ai_model_loaded']}")
    print(f"   세션 디렉토리: {status['session_dir']}")
    
    return True

def test_session_manager():
    """SessionManager 테스트"""
    print("\n💾 SessionManager 테스트...")
    
    session_manager = SessionManager()
    
    # 세션 통계 확인
    stats = session_manager.get_session_stats()
    print(f"   총 세션 수: {stats['total_sessions']}")
    print(f"   수정된 세션 수: {stats['modified_sessions']}")
    
    # 세션 목록 확인
    sessions = session_manager.list_sessions()
    print(f"   세션 목록: {len(sessions)}개")
    
    return True

def main():
    """메인 테스트 실행"""
    print("🚀 하이브리드 필터 통합 테스트 시작!\n")
    
    test_results = []
    
    try:
        # 각 모듈 테스트
        test_results.append(("FilterManager", test_filter_manager()))
        test_results.append(("DataProcessor", test_data_processor()))
        test_results.append(("PipelineManager", test_pipeline_manager()))
        test_results.append(("SessionManager", test_session_manager()))
        
        # 결과 요약
        print("\n" + "="*50)
        print("📋 테스트 결과 요약")
        print("="*50)
        
        all_passed = True
        for module_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {module_name}: {status}")
            if not result:
                all_passed = False
        
        print("\n" + "="*50)
        if all_passed:
            print("🎉 모든 테스트 통과! 하이브리드 필터 통합 성공!")
        else:
            print("⚠️  일부 테스트 실패")
        print("="*50)
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 