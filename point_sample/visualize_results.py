import geopandas as gpd
import matplotlib.pyplot as plt
import json
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union
from skimage.morphology import skeletonize
from skimage.measure import find_contours
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

def extract_skeleton_lines(road_gdf, resolution=2.0):
    """스켈레톤 라인 추출"""
    union_geom = unary_union(road_gdf.geometry)
    bounds = union_geom.bounds
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    
    if hasattr(union_geom, 'geoms'):
        geoms = list(union_geom.geoms)
    else:
        geoms = [union_geom]
    
    raster = rasterize(geoms, out_shape=(height, width), transform=transform, fill=0, default_value=1)
    skeleton = skeletonize(raster.astype(bool))
    
    lines = []
    contours = find_contours(skeleton.astype(float), 0.5)
    
    for contour in contours:
        if len(contour) >= 2:
            coords = []
            for i, j in contour:
                x, y = rasterio.transform.xy(transform, i, j)
                coords.append((x, y))
            if len(coords) >= 2:
                lines.append(LineString(coords))
    
    return lines

def create_beautiful_visualization():
    """결과를 예쁘게 시각화"""
    
    # 데이터 로드
    points_gdf = gpd.read_file('point_sample/p.geojson')
    road_gdf = gpd.read_file('point_sample/road.geojson')
    
    with open('point_sample/skeleton_line_result.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 스켈레톤 라인 추출
    skeleton_lines = extract_skeleton_lines(road_gdf)
    
    # 점 ID 매핑
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
    
    # 좌표 딕셔너리 생성
    point_coords = {}
    for idx, point in points_gdf.iterrows():
        point_id = f"P{point['id']}"
        point_coords[point_id] = (point.geometry.x, point.geometry.y)
    
    # Figure 생성 (큰 사이즈로)
    plt.figure(figsize=(18, 12))
    
    # 스켈레톤 라인들 그리기 (배경)
    for line in skeleton_lines:
        x, y = line.xy
        plt.plot(x, y, color='lightgray', linewidth=1, alpha=0.6)
    
    # 도로 폴리곤 그리기 (연한 배경)
    for idx, road in road_gdf.iterrows():
        if road.geometry.geom_type == 'Polygon':
            x, y = road.geometry.exterior.xy
            plt.fill(x, y, color='lightblue', alpha=0.2, edgecolor='lightblue', linewidth=0.5)
        elif road.geometry.geom_type == 'MultiPolygon':
            for geom in road.geometry.geoms:
                x, y = geom.exterior.xy
                plt.fill(x, y, color='lightblue', alpha=0.2, edgecolor='lightblue', linewidth=0.5)
    
    # 연결선들 그리기 (거리별 색상)
    for pair in results['connected_pairs']:
        p1_id = coord_to_id.get(pair['point1'], pair['point1'])
        p2_id = coord_to_id.get(pair['point2'], pair['point2'])
        
        if p1_id in point_coords and p2_id in point_coords:
            x1, y1 = point_coords[p1_id]
            x2, y2 = point_coords[p2_id]
            distance = pair['total_distance']
            
            # 거리별 색상 및 두께
            if distance <= 5:
                color = 'red'
                linewidth = 3
                alpha = 0.9
            elif distance <= 10:
                color = 'orange'
                linewidth = 2.5
                alpha = 0.8
            elif distance <= 15:
                color = 'green'
                linewidth = 2
                alpha = 0.7
            elif distance <= 25:
                color = 'blue'
                linewidth = 1.5
                alpha = 0.6
            else:
                color = 'purple'
                linewidth = 1
                alpha = 0.5
            
            # 연결선 그리기
            plt.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha)
            
            # 거리 텍스트 (가까운 것들만)
            if distance <= 10:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                plt.text(mid_x, mid_y, f'{distance:.1f}m', 
                        fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                        weight='bold')
    
    # 점들 그리기 (크고 예쁘게)
    for point_id, (x, y) in point_coords.items():
        plt.scatter(x, y, s=200, c='black', edgecolors='white', linewidth=2, zorder=10)
        plt.text(x+3, y+3, point_id, fontsize=14, fontweight='bold', 
                color='black', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    # 범례 생성
    legend_elements = [
        plt.Line2D([0], [0], color='red', linewidth=3, label='매우 가까움 (≤5m)'),
        plt.Line2D([0], [0], color='orange', linewidth=2.5, label='가까움 (5-10m)'),
        plt.Line2D([0], [0], color='green', linewidth=2, label='보통 (10-15m)'),
        plt.Line2D([0], [0], color='blue', linewidth=1.5, label='멀음 (15-25m)'),
        plt.Line2D([0], [0], color='purple', linewidth=1, label='매우 멀음 (>25m)')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # 제목과 레이블
    plt.title('🎯 스켈레톤 기반 점간 거리 네트워크\n(모든 점이 도로망을 통해 연결됨)', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('X 좌표 (m)', fontsize=12)
    plt.ylabel('Y 좌표 (m)', fontsize=12)
    
    # 격자 및 스타일
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    
    # 여백 조정
    plt.tight_layout()
    
    # 저장 및 표시
    plt.savefig('point_sample/distance_network_visualization.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 통계 출력
    print("\n📊 시각화 범례:")
    print("🔴 빨간선: 매우 가까움 (≤5m)")
    print("🟠 주황선: 가까움 (5-10m)")  
    print("🟢 초록선: 보통 (10-15m)")
    print("🔵 파란선: 멀음 (15-25m)")
    print("🟣 보라선: 매우 멀음 (>25m)")
    print("\n💡 거리가 10m 이하인 연결에만 거리 텍스트 표시")

if __name__ == "__main__":
    create_beautiful_visualization() 