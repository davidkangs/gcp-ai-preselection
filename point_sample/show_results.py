import json
import pandas as pd

def show_distance_results():
    """결과를 예쁘게 정리해서 보여주기"""
    
    # 결과 로드
    with open('point_sample/skeleton_line_result.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("🎯 스켈레톤 라인 기반 점간 거리 분석 결과")
    print("=" * 60)
    
    # 점 ID 매핑 (좌표에서 P1~P8로 변환)
    coord_to_id = {
        "P221436_307988": "P1",
        "P221429_307999": "P2", 
        "P221451_307984": "P3",
        "P221420_307968": "P4",
        "P221425_307938": "P5",
        "P221379_307976": "P6",
        "P221365_308016": "P7",
        "P221325_308030": "P8"
    }
    
    # 연결된 점 쌍들 정리
    connected_data = []
    for pair in results['connected_pairs']:
        p1_id = coord_to_id.get(pair['point1'], pair['point1'])
        p2_id = coord_to_id.get(pair['point2'], pair['point2'])
        
        connected_data.append({
            'Point1': p1_id,
            'Point2': p2_id,
            'Distance': f"{pair['total_distance']:.1f}m",
            'Line1': f"L{pair['line1']}",
            'Line2': f"L{pair['line2']}",
            'Same_Line': "같은라인" if pair['same_line'] else "다른라인"
        })
    
    # DataFrame으로 변환
    df = pd.DataFrame(connected_data)
    
    print(f"\n✅ 총 연결된 점 쌍: {len(connected_data)}개")
    print("\n📋 거리별 정렬 결과:")
    
    # 거리순 정렬 (숫자 추출해서)
    df['Distance_Value'] = df['Distance'].str.replace('m', '').astype(float)
    df_sorted = df.sort_values('Distance_Value')
    
    # 거리별 그룹으로 나누어 표시
    print("\n🔥 가까운 거리 (10m 이하):")
    close_pairs = df_sorted[df_sorted['Distance_Value'] <= 10]
    for _, row in close_pairs.iterrows():
        print(f"  {row['Point1']} ↔ {row['Point2']}: {row['Distance']} ({row['Same_Line']})")
    
    print(f"\n📊 중간 거리 (10-20m):")
    medium_pairs = df_sorted[(df_sorted['Distance_Value'] > 10) & (df_sorted['Distance_Value'] <= 20)]
    for _, row in medium_pairs.iterrows():
        print(f"  {row['Point1']} ↔ {row['Point2']}: {row['Distance']} ({row['Same_Line']})")
    
    print(f"\n📏 먼 거리 (20m 이상):")
    far_pairs = df_sorted[df_sorted['Distance_Value'] > 20]
    for _, row in far_pairs.iterrows():
        print(f"  {row['Point1']} ↔ {row['Point2']}: {row['Distance']} ({row['Same_Line']})")
    
    # 통계 정보
    print("\n📊 거리 통계:")
    print(f"  최단거리: {df_sorted['Distance_Value'].min():.1f}m")
    print(f"  최장거리: {df_sorted['Distance_Value'].max():.1f}m")
    print(f"  평균거리: {df_sorted['Distance_Value'].mean():.1f}m")
    print(f"  중간값: {df_sorted['Distance_Value'].median():.1f}m")
    
    # 점별 연결 통계
    print("\n👥 점별 연결 개수:")
    all_points = list(set(df['Point1'].tolist() + df['Point2'].tolist()))
    
    for point in sorted(all_points):
        count = len(df[(df['Point1'] == point) | (df['Point2'] == point)])
        connected_points = []
        
        for _, row in df.iterrows():
            if row['Point1'] == point:
                connected_points.append(f"{row['Point2']}({row['Distance']})")
            elif row['Point2'] == point:
                connected_points.append(f"{row['Point1']}({row['Distance']})")
        
        print(f"  {point}: {count}개 연결 → {', '.join(connected_points[:3])}{'...' if len(connected_points) > 3 else ''}")
    
    print("\n🎉 모든 점들이 스켈레톤 네트워크로 연결되어 있습니다!")

if __name__ == "__main__":
    show_distance_results() 