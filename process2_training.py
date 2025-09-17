import os
import re
from pathlib import Path
import json
from collections import defaultdict, Counter

def complete_analysis_all_files(base_path="./data/재조사지구340개"):
    """전체 340개 파일 완전 분석 (샘플 제한 없음)"""
    
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"❌ 경로가 존재하지 않습니다: {base_path}")
        return
    
    print("🔍 전체 340개 파일 완전 분석 중...")
    print("="*80)
    
    # 파일 수집
    shp_files = list(base_path.glob("*.shp"))
    total_files = len(shp_files)
    
    print(f"📄 총 {total_files}개 shp 파일 발견")
    print("="*80)
    
    # 모든 파일 완전 분석
    all_patterns = {}
    sido_extraction_results = {}
    sigungu_extraction_results = {}
    
    # 정규식 패턴들 (JSON 결과 기반으로 수정)
    extraction_patterns = [
        # 패턴 1: '25년_전북_남원시_강석지구_현황_폴리곤 (작은따옴표 + 축약형)
        (r"^'?(?:\d+년?)_([가-힣]+)_([가-힣]+[시군구])_([가-힣0-9,]+지구)_", "작은따옴표_축약형"),
        
        # 패턴 2: '25년도 전북특별자치도 김제시 만경1,2지구_현황_폴리곤 (공백)
        (r"^'?(?:\d+년도?)\s+([가-힣]+(?:특별자치도|특별시|광역시|도))\s+([가-힣]+[시군구])\s+([가-힣0-9,]+지구)_", "공백_정식명"),
        
        # 패턴 3: 25년_경남남도_고성군_석지지구_현황_폴리곤 (오타형)
        (r"^'?(?:\d+년?)_([가-힣]+(?:남도|특별자치도|특별시|광역시|도))_([가-힣]+[시군구])_([가-힣0-9,]+지구)_", "오타_포함형"),
        
        # 패턴 4: '25년도_인천_남동구_고잔5지구_현황_폴리곤 (광역시_구)
        (r"^'?(?:\d+년도?)_([가-힣]+)_([가-힣]+[구])_([가-힣0-9,]+지구)_", "광역시_구형"),
        
        # 패턴 5: 25년_용인시_기흥구_언남2지구_현황_폴리곤 (시_구)
        (r"^'?(?:\d+년?)_([가-힣]+시)_([가-힣]+구)_([가-힣0-9,]+지구)_", "시_구형"),
        
        # 패턴 6: 25년_광주시_검복1지구_현황_폴리곤 (시만)
        (r"^'?(?:\d+년?)_([가-힣]+시)_([가-힣0-9,]+지구)_", "시_단독형"),
        
        # 패턴 7: 25년_오산시 고현동_고현2지구_현황_폴리곤 (동 포함)
        (r"^'?(?:\d+년?)_([가-힣]+시)\s+([가-힣]+동)_([가-힣0-9,]+지구)_", "동_포함형"),
    ]
    
    print("📋 전체 파일 상세 분석:")
    print("-" * 80)
    
    successful_extractions = []
    failed_extractions = []
    
    for i, shp_file in enumerate(shp_files):
        filename = shp_file.name
        basename = shp_file.stem
        
        # 작은따옴표, 백틱 제거
        clean_basename = basename.lstrip("'`").replace("`", "")
        
        print(f"{i+1:3d}. {filename}")
        
        # 패턴 매칭 시도
        matched = False
        for pattern, pattern_name in extraction_patterns:
            match = re.search(pattern, clean_basename)
            if match:
                groups = match.groups()
                
                if len(groups) >= 2:
                    if len(groups) == 3:
                        sido, sigungu, district = groups[0], groups[1], groups[2]
                    elif len(groups) == 2:
                        sido, sigungu, district = groups[0], groups[1], "미확인"
                    else:
                        sido, sigungu, district = groups[0], "미확인", "미확인"
                    
                    extraction_info = {
                        'filename': filename,
                        'pattern': pattern_name,
                        'sido': sido,
                        'sigungu': sigungu,
                        'district': district,
                        'groups': groups
                    }
                    
                    successful_extractions.append(extraction_info)
                    print(f"     ✅ {pattern_name}: {sido} / {sigungu} / {district}")
                    matched = True
                    break
        
        if not matched:
            failed_extractions.append(filename)
            print(f"     ❌ 패턴 매칭 실패")
    
    # 통계 분석
    print(f"\n📊 전체 분석 결과:")
    print(f"✅ 성공적 추출: {len(successful_extractions)}개")
    print(f"❌ 실패: {len(failed_extractions)}개")
    print(f"🎯 성공률: {len(successful_extractions)/total_files*100:.1f}%")
    
    # 패턴별 통계
    pattern_stats = defaultdict(int)
    for ext in successful_extractions:
        pattern_stats[ext['pattern']] += 1
    
    print(f"\n📊 패턴별 분포:")
    for pattern, count in sorted(pattern_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern:20s}: {count:3d}개")
    
    # 시도별 통계
    sido_stats = defaultdict(int)
    for ext in successful_extractions:
        sido_stats[ext['sido']] += 1
    
    print(f"\n🏛️ 시도별 분포:")
    for sido, count in sorted(sido_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sido:20s}: {count:3d}개")
    
    # 시군구별 통계
    sigungu_stats = defaultdict(int)
    for ext in successful_extractions:
        sigungu_stats[ext['sigungu']] += 1
    
    print(f"\n🏘️ 시군구별 분포 (상위 20개):")
    sorted_sigungu = sorted(sigungu_stats.items(), key=lambda x: x[1], reverse=True)
    for sigungu, count in sorted_sigungu[:20]:
        print(f"  {sigungu:15s}: {count:3d}개")
    
    # 실패한 파일들 분석
    if failed_extractions:
        print(f"\n❌ 패턴 매칭 실패 파일들 ({len(failed_extractions)}개):")
        for failed_file in failed_extractions[:10]:
            print(f"    {failed_file}")
        if len(failed_extractions) > 10:
            print(f"    ... 및 {len(failed_extractions)-10}개 더")
    
    # 시도-시군구 매핑 생성
    sido_sigungu_mapping = defaultdict(set)
    for ext in successful_extractions:
        sido_sigungu_mapping[ext['sido']].add(ext['sigungu'])
    
    print(f"\n🗺️ 시도-시군구 완전 매핑:")
    for sido, sigungu_set in sorted(sido_sigungu_mapping.items()):
        print(f"  {sido}:")
        for sigungu in sorted(sigungu_set):
            print(f"    {sigungu}")
    
    # 추천 정규식 패턴 생성
    print(f"\n💡 완벽한 추출을 위한 추천 정규식:")
    print("="*80)
    
    # 가장 효과적인 패턴 순서대로 추천
    recommended_patterns = []
    for pattern, count in sorted(pattern_stats.items(), key=lambda x: x[1], reverse=True):
        for pattern_regex, pattern_name in extraction_patterns:
            if pattern_name == pattern:
                recommended_patterns.append((pattern_regex, pattern_name, count))
                break
    
    print("# 추천 패턴 (효과 순서대로):")
    for i, (regex, name, count) in enumerate(recommended_patterns):
        print(f"패턴{i+1}: {name} ({count}개)")
        print(f"  정규식: {regex}")
        print()
    
    # 완전 분석 결과 JSON 저장
    complete_analysis = {
        'total_files': total_files,
        'successful_extractions': len(successful_extractions),
        'failed_extractions': len(failed_extractions),
        'success_rate': len(successful_extractions)/total_files*100,
        'pattern_statistics': dict(pattern_stats),
        'sido_statistics': dict(sido_stats), 
        'sigungu_statistics': dict(sigungu_stats),
        'sido_sigungu_mapping': {k: list(v) for k, v in sido_sigungu_mapping.items()},
        'successful_extractions_detail': successful_extractions,
        'failed_files': failed_extractions,
        'recommended_patterns': [(regex, name, count) for regex, name, count in recommended_patterns]
    }
    
    output_file = "complete_district_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(complete_analysis, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 완전 분석 결과 저장: {output_file}")
    
    return complete_analysis

if __name__ == "__main__":
    # 전체 완전 분석 실행
    analysis = complete_analysis_all_files()
    
    print(f"\n✅ 전체 340개 파일 완전 분석 완료!")
    print(f"📊 이 결과를 바탕으로 100% 매칭되는 완벽한 클리핑 코드를 만들어드리겠습니다!")