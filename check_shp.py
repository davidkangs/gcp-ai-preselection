"""
geometry 디버깅 도구
문제가 되는 shapefile을 분석하여 오류 원인을 찾습니다.
"""

import geopandas as gpd
import sys
from pathlib import Path

def debug_shapefile(file_path):
    """shapefile의 geometry 문제를 진단"""
    print(f"🔍 Shapefile 분석: {file_path}")
    print("=" * 50)
    
    try:
        # 파일 로드
        gdf = gpd.read_file(file_path)
        print(f"✅ 파일 로드 성공")
        print(f"   - 총 피처 수: {len(gdf)}")
        print(f"   - CRS: {gdf.crs}")
        print(f"   - Bounds: {gdf.total_bounds}")
        
        # geometry 타입 분석
        geom_types = gdf.geometry.geom_type.value_counts()
        print(f"\n📊 Geometry 타입:")
        for geom_type, count in geom_types.items():
            print(f"   - {geom_type}: {count}개")
        
        # 각 geometry 검사
        print(f"\n🔍 개별 geometry 검사:")
        problem_indices = []
        
        for idx, geom in enumerate(gdf.geometry):
            try:
                # 기본 정보
                is_valid = geom.is_valid
                is_empty = geom.is_empty
                
                # 좌표 개수 확인 (폴리곤인 경우)
                coord_count = 0
                if hasattr(geom, 'exterior') and geom.exterior is not None:
                    coord_count = len(list(geom.exterior.coords))
                elif hasattr(geom, 'geoms'):  # MultiPolygon
                    coord_counts = []
                    for poly in geom.geoms:
                        if hasattr(poly, 'exterior') and poly.exterior is not None:
                            coord_counts.append(len(list(poly.exterior.coords)))
                    coord_count = min(coord_counts) if coord_counts else 0
                
                # 문제 있는 geometry 식별
                if not is_valid or is_empty or coord_count < 4:
                    problem_indices.append(idx)
                    print(f"   ❌ 인덱스 {idx}: valid={is_valid}, empty={is_empty}, coords={coord_count}")
                    
                    # 상세 오류 정보
                    if not is_valid:
                        from shapely.validation import explain_validity
                        print(f"      오류: {explain_validity(geom)}")
                        
            except Exception as e:
                problem_indices.append(idx)
                print(f"   💥 인덱스 {idx}: 검사 중 오류 - {e}")
        
        if not problem_indices:
            print("   ✅ 모든 geometry가 유효합니다!")
        else:
            print(f"\n⚠️  문제 있는 geometry: {len(problem_indices)}개")
            print(f"   인덱스: {problem_indices}")
            
            # 수정 시도
            print(f"\n🔧 수정 시도 중...")
            from shapely.validation import make_valid
            
            fixed_count = 0
            for idx in problem_indices:
                try:
                    original = gdf.geometry.iloc[idx]
                    fixed = make_valid(original)
                    
                    if fixed.is_valid and not fixed.is_empty:
                        gdf.geometry.iloc[idx] = fixed
                        fixed_count += 1
                        print(f"   ✅ 인덱스 {idx} 수정 성공")
                    else:
                        print(f"   ❌ 인덱스 {idx} 수정 실패")
                        
                except Exception as e:
                    print(f"   💥 인덱스 {idx} 수정 중 오류: {e}")
            
            print(f"\n📊 수정 결과: {fixed_count}/{len(problem_indices)}개 수정됨")
            
            # 수정된 파일 저장 옵션
            if fixed_count > 0:
                output_path = Path(file_path).parent / f"{Path(file_path).stem}_fixed.shp"
                try:
                    gdf.to_file(output_path)
                    print(f"💾 수정된 파일 저장: {output_path}")
                except Exception as e:
                    print(f"❌ 파일 저장 실패: {e}")
        
    except Exception as e:
        print(f"❌ 파일 분석 실패: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python debug_geometry.py <shapefile_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        sys.exit(1)
    
    debug_shapefile(file_path)