import json
import numpy as np
from shapely.geometry import shape, Point, LineString
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.filters.hybrid_filter import create_hybrid_filter

# 파일 경로
POINT_PATH = 'point_sample/p.geojson'
ROAD_PATH = 'point_sample/road.geojson'

# 1. 점 데이터 로드
with open(POINT_PATH, encoding='utf-8') as f:
    point_data = json.load(f)
points = [(feat['properties']['id'], tuple(feat['geometry']['coordinates'])) for feat in point_data['features']]
point_coords = [pt for _, pt in points]
point_roles = {pt: 'unknown' for _, pt in points}

# 2. 도로 멀티폴리곤 → 중심선(LineString) 추출 (스켈레톤)
with open(ROAD_PATH, encoding='utf-8') as f:
    road_data = json.load(f)
skeleton_lines = []
for feat in road_data['features']:
    geom = shape(feat['geometry'])
    if geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            coords = list(poly.exterior.coords)
            if len(coords) >= 2:
                skeleton_lines.append(LineString(coords))
    elif geom.geom_type == 'Polygon':
        coords = list(geom.exterior.coords)
        if len(coords) >= 2:
            skeleton_lines.append(LineString(coords))

# 3. 하이브리드 필터 적용
hybrid_filter = create_hybrid_filter()
# skeleton_lines를 하나의 LineString으로 합쳐서 넘김 (테스트 목적)
all_skel_coords = []
for skel in skeleton_lines:
    all_skel_coords.extend(list(skel.coords))
filtered_points = hybrid_filter.hybrid_filter(
    points=point_coords,
    skeleton=all_skel_coords,
    road_polygon=None,
    point_roles=point_roles
)

# 4. 시각화: 스켈레톤, 필터링 전후 점, 연결 구조
fig, ax = plt.subplots(figsize=(8, 6))
# 스켈레톤(중심선) 그리기
for skel in skeleton_lines:
    x, y = skel.xy
    ax.plot(x, y, color='black', linewidth=2, alpha=0.3, label='스켈레톤')
# 필터링 전 점(회색)
for pid, coord in points:
    ax.scatter(coord[0], coord[1], color='gray', s=60, zorder=2, alpha=0.5)
    ax.text(coord[0], coord[1], str(pid), color='gray', fontsize=9, ha='left', va='top')
# 필터링 후 점(파랑)
for coord in filtered_points:
    ax.scatter(coord[0], coord[1], color='blue', s=80, zorder=3)
    ax.text(coord[0], coord[1], f"{coord}", color='blue', fontsize=10, ha='right', va='bottom')
ax.set_title('스켈레톤 기반 하이브리드 필터링 결과')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"필터링 전 점 개수: {len(points)}")
print(f"필터링 후 점 개수: {len(filtered_points)}") 