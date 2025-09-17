#!/usr/bin/env python3
"""
Process3 하이브리드 실행 - 휴리스틱 중심 + AI 보조 (5개 제한 패치 적용)
실행: python run_process3_hybrid_fixed.py
"""

import sys
from pathlib import Path
import numpy as np

# 경로 설정
sys.path.append(str(Path(__file__).parent))

# 필수 디렉토리 생성
print("🔧 디렉토리 초기화...")
for folder in ['sessions', 'models', 'results', 'logs']:
    Path(folder).mkdir(exist_ok=True)
    print(f"  ✅ {folder}/")

print("\n🎯 휴리스틱 중심 + AI 보조 모드 시작...")
print("📊 특징:")
print("  🔹 휴리스틱이 메인 검출 담당")
print("  🔹 AI는 최대 5개만 보조")
print("  🔹 신뢰도 순 정렬로 상위 5개 선택")
print("  🔹 거리 체크로 중복 방지")

# 🚨 5개 제한 패치 적용
def apply_total_5_patch():
    try:
        from heuristic_centered_inference import HybridPredictionWorker
        
        def new_merge_results(self, skeleton_array, heuristic_results, ai_filtered):
            """휴리스틱 100% 보존 + AI 보조 전체 5개만"""
            
            print("🚨 [DEBUG] 5개 제한 패치 실행 중...")
            
            # 1. 휴리스틱 결과 100% 보존
            final_results = {
                'intersection': list(heuristic_results['intersection']),
                'curve': list(heuristic_results['curve']),                
                'endpoint': list(heuristic_results['endpoint']),
                'delete': []
            }
            
            print(f"\n🔧 휴리스틱 100% 보존:")
            print(f"  교차점: {len(final_results['intersection'])}개")
            print(f"  커브: {len(final_results['curve'])}개") 
            print(f"  끝점: {len(final_results['endpoint'])}개")
            
            # 2. 기존 모든 점들 수집
            all_existing_points = []
            all_existing_points.extend(heuristic_results['intersection'])
            all_existing_points.extend(heuristic_results['curve'])
            all_existing_points.extend(heuristic_results['endpoint'])
            
            print(f"기존 점 총 개수: {len(all_existing_points)}개")
            
            # 3. 모든 AI 예측을 하나의 리스트로 통합
            all_ai_candidates = []
            for i, (idx, pred, conf) in enumerate(zip(
                ai_filtered['indices'], 
                ai_filtered['predictions'], 
                ai_filtered['confidences']
            )):
                if idx >= len(skeleton_array) or conf < 0.85:
                    continue
                    
                point = tuple(skeleton_array[idx])
                all_ai_candidates.append((point, pred, conf))
            
            print(f"필터링 후 AI 후보: {len(all_ai_candidates)}개")
            
            # 4. 신뢰도 순으로 정렬하고 **강제로** 상위 5개만 선택
            all_ai_candidates.sort(key=lambda x: x[2], reverse=True)
            
            # 🚨 강제 제한: 무조건 5개만
            MAX_AI_ADDITIONS = 5
            top_candidates = all_ai_candidates[:MAX_AI_ADDITIONS]
            
            print(f"\n🤖 전체 AI 후보 {len(all_ai_candidates)}개 중 강제로 상위 {len(top_candidates)}개만 선택")
            
            # 5. 거리 임계값을 좀 더 관대하게 설정
            min_distance_threshold = 20.0  # 20m 이내는 제외
            ai_added = {'intersection': 0, 'curve': 0, 'endpoint': 0, 'delete': 0}
            actually_added = 0
            skipped_by_distance = 0
            
            for i, (point, pred, conf) in enumerate(top_candidates):
                # 기존 점들과의 최소 거리 계산
                min_distance = float('inf')
                if all_existing_points:  # 기존 점이 있을 때만 거리 체크
                    for existing_point in all_existing_points:
                        dist = np.linalg.norm(np.array(point) - np.array(existing_point))
                        min_distance = min(min_distance, dist)
                else:
                    min_distance = min_distance_threshold + 1  # 기존 점이 없으면 통과
                
                action_name = {1: '교차점', 2: '커브', 3: '끝점', 4: '삭제'}[pred]
                print(f"  {i+1}. {action_name} - 신뢰도: {conf:.3f}, 최근접거리: {min_distance:.1f}m", end="")
                
                if min_distance < min_distance_threshold:
                    print(f" ❌ 거리 {min_distance_threshold}m 미만으로 스킵")
                    skipped_by_distance += 1
                    continue
                
                # 추가
                if pred == 1:
                    final_results['intersection'].append(point)
                    ai_added['intersection'] += 1
                elif pred == 2:
                    final_results['curve'].append(point)
                    ai_added['curve'] += 1
                elif pred == 3:
                    final_results['endpoint'].append(point)
                    ai_added['endpoint'] += 1
                elif pred == 4:
                    final_results['delete'].append(point)
                    ai_added['delete'] += 1
                
                all_existing_points.append(point)  # 다음 거리 계산용
                actually_added += 1
                print(f" ✅ 추가됨")
                
                # 🚨 이중 체크: 혹시라도 5개 초과하면 강제 중단
                if actually_added >= MAX_AI_ADDITIONS:
                    print(f"⚠️ 최대 추가 개수 {MAX_AI_ADDITIONS}개 도달, 중단")
                    break
            
            print(f"\n🎯 최종 결과:")
            print(f"  처리한 후보: {len(top_candidates)}개")
            print(f"  거리로 스킵: {skipped_by_distance}개") 
            print(f"  실제 AI 보조 추가: {actually_added}개 (최대 {MAX_AI_ADDITIONS}개)")
            print(f"  교차점: +{ai_added['intersection']}개")
            print(f"  커브: +{ai_added['curve']}개")
            print(f"  끝점: +{ai_added['endpoint']}개")
            print(f"  삭제: +{ai_added['delete']}개")
            
            # 최종 점 개수 확인
            total_final = len(final_results['intersection']) + len(final_results['curve']) + len(final_results['endpoint'])
            print(f"\n📊 최종 전체 점 개수: {total_final}개")
            
            return final_results
        
        # 기존 함수를 새 함수로 교체
        HybridPredictionWorker.merge_results = new_merge_results
        print("\n✅ 5개 제한 패치 적용 완료!")
        
    except ImportError as e:
        print(f"❌ 패치 적용 실패: {e}")

# 패치 적용 후 실행
try:
    apply_total_5_patch()
    from heuristic_centered_inference import main
    main()
except ImportError as e:
    print(f"❌ 모듈 import 오류: {e}")
except Exception as e:
    print(f"❌ 실행 오류: {e}")
    import traceback
    traceback.print_exc()