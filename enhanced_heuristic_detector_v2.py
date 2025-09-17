import networkx as nx
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.validation import make_valid
from sklearn.cluster import DBSCAN
from rasterio.features import rasterize
from rasterio.transform import from_origin
from skimage.morphology import skeletonize
import warnings
warnings.filterwarnings('ignore')

class EnhancedHeuristicDetectorV2:
    def __init__(self):
        self.derivative_threshold = 0.3
        self.area_percentile = 5
        self.intersection_radius = 30
        self.curve_radius = 10
        self.curve_intersection_distance = 30
        self.skeleton_resolution = 1
        
        # 🆕 교차점 중복 제거 파라미터
        self.intersection_merge_distance = 30  # 30M 이내 교차점 통합
        self.linearity_ratio_threshold = 0.85  # 직선성 비율 임계값
        self.angle_change_threshold = 30       # 각도 변화 누적 임계값 (도)
        
        # 🆕 커브 중복 제거 파라미터
        self.curve_merge_distance = 30         # 30M 이내 커브 검사
        self.curve_linearity_threshold = 0.90  # 커브 직선성 임계값 (더 엄격)
        self.curve_angle_threshold = 20        # 커브 각도 변화 임계값 (더 엄격)
        
        # 현재 분석중인 폴리곤들 저장
        self.current_polygons = []
        
    def detect_all(self, gdf, skeleton=None):
        print("🔍 향상된 휴리스틱 검출 시작...")
        
        polygons = self._extract_and_filter_polygons(gdf)
        self.current_polygons = polygons  # 폴리곤 저장
        raw_curve_points = self._detect_curves_from_boundaries(polygons)
        
        if skeleton is None:
            skeleton_data = self._extract_skeleton(polygons)
            skeleton_array = skeleton_data['skeleton_points']
        else:
            skeleton_array = skeleton if isinstance(skeleton, list) else skeleton.tolist()
            skeleton_data = self._extract_skeleton(polygons)  # 연결성 분석을 위해 필요
        
        intersection_points = self._detect_intersections_from_skeleton(polygons, skeleton_data)
        intersection_centers = self._cluster_points(intersection_points, self.intersection_radius)
        
        # 🆕 교차점 중복 제거
        merged_intersections = self._remove_redundant_intersections(intersection_centers, skeleton_data)
        
        # 🆕 향상된 커브 필터링 적용
        filtered_curves = self._enhance_curve_filtering(raw_curve_points, merged_intersections, skeleton_data)
        curve_centers = self._cluster_points(filtered_curves, self.curve_radius)
        
        # 🆕 커브 중복 제거 (폭 변화 노이즈 제거)
        final_curves = self._remove_redundant_curves(curve_centers, skeleton_data)
        
        endpoint_points = self._detect_endpoints(skeleton_array)
        
        print(f"✅ 검출 완료: 교차점 {len(merged_intersections)}개, 커브 {len(final_curves)}개, 끝점 {len(endpoint_points)}개")
        
        return {
            'intersection': [(p.x, p.y) for p in merged_intersections],
            'curve': [(p.x, p.y) for p in final_curves],
            'endpoint': [(p[0], p[1]) for p in endpoint_points]
        }
    
    def _remove_redundant_intersections(self, intersections, skeleton_data):
        """30M 이내 교차점들 중 직선으로 연결된 것들을 통합"""
        if len(intersections) <= 1:
            return intersections
        
        print("🔗 교차점 중복 제거 분석...")
        
        # 스켈레톤 그래프 생성
        skeleton = skeleton_data['skeleton']
        transform = skeleton_data['transform']
        G = self._skeleton_to_graph(skeleton)
        
        if G.number_of_nodes() == 0:
            return intersections
        
        # 교차점들을 그래프 노드로 매핑
        intersection_nodes = []
        for intersection in intersections:
            col = int((intersection.x - transform.c) / transform.a)
            row = int((transform.f - intersection.y) / abs(transform.e))
            
            closest_node = None
            min_dist = float('inf')
            for node in G.nodes():
                node_dist = ((node[0] - col)**2 + (node[1] - row)**2)**0.5
                if node_dist < min_dist:
                    min_dist = node_dist
                    closest_node = node
            
            if closest_node and min_dist < 5:
                intersection_nodes.append((intersection, closest_node))
            else:
                intersection_nodes.append((intersection, None))
        
        # 가까운 교차점 쌍들 찾기
        merge_pairs = []
        for i in range(len(intersections)):
            for j in range(i + 1, len(intersections)):
                dist = intersections[i].distance(intersections[j])
                if dist <= self.intersection_merge_distance:
                    # 두 교차점이 모두 그래프에 매핑되었는지 확인
                    if intersection_nodes[i][1] is not None and intersection_nodes[j][1] is not None:
                        merge_pairs.append((i, j, dist))
        
        print(f"  📏 30M 이내 교차점 쌍: {len(merge_pairs)}개")
        
        # 직선성 평가 및 통합 결정
        to_merge = []
        for i, j, dist in merge_pairs:
            node_i = intersection_nodes[i][1]
            node_j = intersection_nodes[j][1]
            
            # 경로의 직선성 평가
            is_linear = self._evaluate_path_linearity(G, node_i, node_j, transform)
            
            if is_linear:
                to_merge.append((i, j))
        
        print(f"  ✅ 직선 연결된 쌍: {len(to_merge)}개")
        
        # 통합 실행
        merged_intersections = self._merge_close_intersections(intersections, to_merge, skeleton_data)
        
        print(f"  🔄 교차점 통합: {len(intersections)} → {len(merged_intersections)}")
        return merged_intersections
    
    def _remove_redundant_curves(self, curves, skeleton_data):
        """체인 분석으로 무의미한 커브들 제거"""
        if len(curves) == 0:
            return curves
        
        print("🌀 커브 체인 분석 (무의미한 중간 커브 제거)...")
        
        # 모든 포인트들 (교차점 + 커브) 통합 분석
        all_intersections = self._get_current_intersections(skeleton_data)
        all_points = all_intersections + curves
        
        if len(all_points) < 2:
            return curves
        
        # 연결 체인들 찾기
        chains = self._find_connection_chains(all_points, skeleton_data)
        print(f"  🔗 발견된 연결 체인: {len(chains)}개")
        
        # 각 체인의 직선성 분석 및 중간 요소 제거
        curves_to_remove = set()
        for chain in chains:
            if len(chain) >= 3:  # 시작점 + 중간요소들 + 끝점
                should_remove = self._analyze_chain_linearity(chain, skeleton_data)
                if should_remove:
                    # 체인의 중간 커브들만 제거 대상에 추가
                    for i in range(1, len(chain) - 1):  # 시작점과 끝점 제외
                        point = chain[i]
                        if point in curves:
                            curves_to_remove.add(id(point))
        
        # 제거 대상이 아닌 커브들만 유지
        final_curves = []
        removed_count = 0
        for curve in curves:
            if id(curve) not in curves_to_remove:
                final_curves.append(curve)
            else:
                removed_count += 1
        
        print(f"  🚫 체인 분석으로 제거된 커브: {removed_count}개")
        print(f"  ✅ 최종 커브: {len(curves)} → {len(final_curves)}개")
        return final_curves
    
    def _get_current_intersections(self, skeleton_data):
        """현재 단계의 교차점들을 다시 추출 (중복 제거 후)"""
        skeleton = skeleton_data['skeleton']
        transform = skeleton_data['transform']
        G = self._skeleton_to_graph(skeleton)
        
        intersections = []
        for node, degree in G.degree():
            if degree >= 3:
                col, row = node
                x = transform.c + col * transform.a
                y = transform.f + row * transform.e
                intersections.append(Point(x, y))
        
        return intersections
    
    def _find_connection_chains(self, all_points, skeleton_data):
        """스켈레톤 그래프에서 연결된 포인트 체인들을 찾기"""
        skeleton = skeleton_data['skeleton']
        transform = skeleton_data['transform']
        G = self._skeleton_to_graph(skeleton)
        
        if G.number_of_nodes() == 0:
            return []
        
        # 포인트들을 그래프 노드로 매핑
        point_to_node = {}
        for point in all_points:
            col = int((point.x - transform.c) / transform.a)
            row = int((transform.f - point.y) / abs(transform.e))
            
            closest_node = None
            min_dist = float('inf')
            for node in G.nodes():
                node_dist = ((node[0] - col)**2 + (node[1] - row)**2)**0.5
                if node_dist < min_dist:
                    min_dist = node_dist
                    closest_node = node
            
            if closest_node and min_dist < 5:
                point_to_node[point] = closest_node
        
        # 연결된 체인들 찾기
        chains = []
        visited_points = set()
        
        for start_point in all_points:
            if start_point in visited_points or start_point not in point_to_node:
                continue
            
            # 이 포인트에서 시작하는 체인 탐색
            chain = self._trace_chain_from_point(start_point, all_points, point_to_node, G, transform)
            
            if len(chain) >= 3:  # 최소 3개 포인트 (시작-중간-끝)
                chains.append(chain)
                visited_points.update(chain)
        
        return chains
    
    def _trace_chain_from_point(self, start_point, all_points, point_to_node, graph, transform):
        """특정 포인트에서 시작하여 연결된 체인을 추적"""
        if start_point not in point_to_node:
            return [start_point]
        
        chain = [start_point]
        current_node = point_to_node[start_point]
        visited_nodes = {current_node}
        
        # 양방향으로 체인 확장
        for direction in [1, -1]:  # 정방향, 역방향
            temp_chain = []
            temp_node = current_node
            temp_visited = set(visited_nodes)
            
            while True:
                # 현재 노드와 연결된 다음 포인트 찾기
                next_point = None
                next_node = None
                
                # 인접한 노드들 중에서 다른 포인트와 매핑된 것 찾기
                neighbors = list(graph.neighbors(temp_node))
                for neighbor in neighbors:
                    if neighbor in temp_visited:
                        continue
                    
                    # 이 노드 근처에 다른 포인트가 있는지 확인
                    for point in all_points:
                        if point == start_point or point in chain or point in temp_chain:
                            continue
                        if point in point_to_node and point_to_node[point] == neighbor:
                            # 거리 확인 (너무 멀면 체인이 아님)
                            world_x = transform.c + neighbor[0] * transform.a
                            world_y = transform.f + neighbor[1] * transform.e
                            dist = ((point.x - world_x)**2 + (point.y - world_y)**2)**0.5
                            if dist <= 30:  # 30m 이내
                                next_point = point
                                next_node = neighbor
                                break
                    
                    if next_point:
                        break
                
                if not next_point:
                    break
                
                temp_chain.append(next_point)
                temp_visited.add(next_node)
                temp_node = next_node
            
            # 역방향이면 뒤집어서 추가
            if direction == -1:
                chain = list(reversed(temp_chain)) + chain
            else:
                chain.extend(temp_chain)
        
        return chain
    
    def _analyze_chain_linearity(self, chain, skeleton_data):
        """체인의 시작점과 끝점 간 직선성 분석"""
        if len(chain) < 3:
            return False
        
        start_point = chain[0]
        end_point = chain[-1]
        
        skeleton = skeleton_data['skeleton']
        transform = skeleton_data['transform']
        G = self._skeleton_to_graph(skeleton)
        
        # 시작점과 끝점을 노드로 변환
        start_node = self._point_to_node(start_point, G, transform)
        end_node = self._point_to_node(end_point, G, transform)
        
        if not start_node or not end_node:
            return False
        
        # 경로 직선성 평가
        try:
            path = nx.shortest_path(G, start_node, end_node)
            is_linear = self._evaluate_path_linearity_strict(path, transform)
            
            if is_linear:
                chain_length = len(chain)
                print(f"    📏 직선 체인 발견: {chain_length}개 포인트 → 중간 {chain_length-2}개 제거 예정")
                return True
            
        except nx.NetworkXNoPath:
            pass
        
        return False
    
    def _point_to_node(self, point, graph, transform):
        """포인트를 그래프 노드로 변환"""
        col = int((point.x - transform.c) / transform.a)
        row = int((transform.f - point.y) / abs(transform.e))
        
        closest_node = None
        min_dist = float('inf')
        for node in graph.nodes():
            node_dist = ((node[0] - col)**2 + (node[1] - row)**2)**0.5
            if node_dist < min_dist:
                min_dist = node_dist
                closest_node = node
        
        return closest_node if min_dist < 5 else None
    
    def _evaluate_path_linearity_strict(self, path, transform):
        """경로의 직선성을 엄격하게 평가 (체인용)"""
        if len(path) < 2:
            return True
        
        # 월드 좌표로 변환
        world_path = []
        for node in path:
            col, row = node
            world_x = transform.c + col * transform.a
            world_y = transform.f + row * transform.e
            world_path.append((world_x, world_y))
        
        # 직선거리 vs 실제경로거리 비율
        start_point = np.array(world_path[0])
        end_point = np.array(world_path[-1])
        straight_distance = np.linalg.norm(end_point - start_point)
        
        path_distance = 0
        for k in range(len(world_path) - 1):
            seg_dist = np.linalg.norm(np.array(world_path[k+1]) - np.array(world_path[k]))
            path_distance += seg_dist
        
        if path_distance == 0:
            return True
        
        linearity_ratio = straight_distance / path_distance
        
        # 체인 분석용 엄격한 기준
        return linearity_ratio >= 0.92  # 92% 이상 직선성
    
    def _evaluate_path_linearity(self, graph, node_start, node_end, transform):
        """두 노드 간 경로의 직선성을 평가"""
        try:
            path = nx.shortest_path(graph, node_start, node_end)
        except nx.NetworkXNoPath:
            return False
        
        if len(path) < 2:
            return True
        
        # 월드 좌표로 변환
        world_path = []
        for node in path:
            col, row = node
            world_x = transform.c + col * transform.a
            world_y = transform.f + row * transform.e
            world_path.append((world_x, world_y))
        
        # 1. 직선거리 vs 실제경로거리 비율
        start_point = np.array(world_path[0])
        end_point = np.array(world_path[-1])
        straight_distance = np.linalg.norm(end_point - start_point)
        
        path_distance = 0
        for k in range(len(world_path) - 1):
            seg_dist = np.linalg.norm(np.array(world_path[k+1]) - np.array(world_path[k]))
            path_distance += seg_dist
        
        if path_distance == 0:
            return True
            
        linearity_ratio = straight_distance / path_distance
        
        # 2. 각도 변화 누적
        angle_changes = 0
        if len(world_path) >= 3:
            for k in range(1, len(world_path) - 1):
                p1 = np.array(world_path[k-1])
                p2 = np.array(world_path[k])
                p3 = np.array(world_path[k+1])
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_change = np.degrees(np.arccos(cos_angle))
                    angle_changes += angle_change
        
        # 직선성 판단
        is_linear = (linearity_ratio >= self.linearity_ratio_threshold and 
                    angle_changes <= self.angle_change_threshold)
        
        return is_linear
    
    def _merge_close_intersections(self, intersections, merge_pairs, skeleton_data):
        """교차점들을 중요도 기반으로 선택적 제거"""
        if not merge_pairs:
            return intersections
        
        print("🏆 교차점 중요도 분석...")
        
        # 각 교차점의 중요도 계산
        importance_scores = self._calculate_intersection_importance(intersections, skeleton_data)
        
        # 통합할 그룹들 찾기 (Union-Find 알고리즘)
        parent = list(range(len(intersections)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for i, j in merge_pairs:
            union(i, j)
        
        # 그룹별로 가장 중요한 교차점 선택
        groups = {}
        for i in range(len(intersections)):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        selected_intersections = []
        removed_count = 0
        
        for group_indices in groups.values():
            if len(group_indices) == 1:
                # 단독 교차점은 그대로 유지
                selected_intersections.append(intersections[group_indices[0]])
            else:
                # 그룹에서 가장 중요한 교차점 선택
                best_idx = max(group_indices, key=lambda idx: importance_scores[idx])
                selected_intersections.append(intersections[best_idx])
                removed_count += len(group_indices) - 1
                
                # 중요도 로그 출력
                print(f"    🎯 그룹 통합: {len(group_indices)}개 → 1개 선택")
                for idx in group_indices:
                    status = "✅선택" if idx == best_idx else "❌제거"
                    print(f"      {status} 교차점{idx}: 도로{importance_scores[idx]['roads']}갈래, "
                          f"평균길이{importance_scores[idx]['avg_length']:.0f}m, "
                          f"점수{importance_scores[idx]['score']:.0f}")
        
        print(f"  🔄 중요도 기반 선택: {len(intersections)} → {len(selected_intersections)} (제거: {removed_count}개)")
        return selected_intersections
    
    def _calculate_intersection_importance(self, intersections, skeleton_data):
        """각 교차점의 중요도 점수 계산 (도로수 × 도로길이)"""
        skeleton = skeleton_data['skeleton']
        transform = skeleton_data['transform']
        G = self._skeleton_to_graph(skeleton)
        
        importance_scores = []
        
        for i, intersection in enumerate(intersections):
            # 교차점을 그래프 노드로 변환
            col = int((intersection.x - transform.c) / transform.a)
            row = int((transform.f - intersection.y) / abs(transform.e))
            
            closest_node = None
            min_dist = float('inf')
            for node in G.nodes():
                node_dist = ((node[0] - col)**2 + (node[1] - row)**2)**0.5
                if node_dist < min_dist:
                    min_dist = node_dist
                    closest_node = node
            
            if closest_node and min_dist < 5:
                # 연결된 도로들의 정보 계산
                road_info = self._analyze_connected_roads(G, closest_node, transform)
                road_count = road_info['count']
                avg_road_length = road_info['avg_length']
                total_road_length = road_info['total_length']
                
                # 중요도 점수 = 도로수 × 총길이 + 도로수 보너스
                importance_score = road_count * total_road_length + road_count * 1000
                
                importance_scores.append({
                    'score': importance_score,
                    'roads': road_count,
                    'avg_length': avg_road_length,
                    'total_length': total_road_length
                })
            else:
                # 매핑 실패시 기본값
                importance_scores.append({
                    'score': 0,
                    'roads': 0,
                    'avg_length': 0,
                    'total_length': 0
                })
        
        return importance_scores
    
    def _analyze_connected_roads(self, graph, center_node, transform):
        """교차점에서 뻗어나가는 도로들의 정보 분석"""
        neighbors = list(graph.neighbors(center_node))
        road_lengths = []
        
        for neighbor in neighbors:
            # 각 방향으로 도로 끝까지 추적
            road_length = self._trace_road_length(graph, center_node, neighbor, transform)
            road_lengths.append(road_length)
        
        road_count = len(road_lengths)
        total_length = sum(road_lengths) if road_lengths else 0
        avg_length = total_length / road_count if road_count > 0 else 0
        
        return {
            'count': road_count,
            'total_length': total_length,
            'avg_length': avg_length,
            'lengths': road_lengths
        }
    
    def _trace_road_length(self, graph, start_node, direction_node, transform, max_distance=1000):
        """특정 방향으로 도로를 추적하여 길이 계산"""
        visited = {start_node}
        current_node = direction_node
        total_length = 0
        
        while current_node and total_length < max_distance:
            if current_node in visited:
                break
            
            visited.add(current_node)
            
            # 현재 노드의 월드 좌표
            col, row = current_node
            world_x = transform.c + col * transform.a
            world_y = transform.f + row * transform.e
            
            # 이전 노드와의 거리 계산
            if len(visited) > 1:
                prev_nodes = [n for n in visited if n != current_node]
                if prev_nodes:
                    prev_node = prev_nodes[-1]
                    prev_col, prev_row = prev_node
                    prev_world_x = transform.c + prev_col * transform.a
                    prev_world_y = transform.f + prev_row * transform.e
                    
                    segment_length = ((world_x - prev_world_x)**2 + (world_y - prev_world_y)**2)**0.5
                    total_length += segment_length
            
            # 다음 노드 찾기 (degree가 3 이상이면 다른 교차점이므로 중단)
            neighbors = list(graph.neighbors(current_node))
            if len(neighbors) >= 3:  # 교차점 도달
                break
            
            # 방문하지 않은 다음 노드로 이동
            next_node = None
            for neighbor in neighbors:
                if neighbor not in visited:
                    next_node = neighbor
                    break
            
            current_node = next_node
        
        return total_length
    
    def _extract_and_filter_polygons(self, gdf):
        print("📊 폴리곤 추출 및 면적 분석...")
        polygons = []
        
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
                
            if not geom.is_valid:
                geom = make_valid(geom)
            
            if geom.geom_type == 'Polygon':
                polygons.append(geom)
            elif geom.geom_type == 'MultiPolygon':
                polygons.extend(list(geom.geoms))
            elif geom.geom_type in ['LineString', 'MultiLineString']:
                buffered = geom.buffer(1.0)
                if buffered.geom_type == 'Polygon':
                    polygons.append(buffered)
                elif buffered.geom_type == 'MultiPolygon':
                    polygons.extend(list(buffered.geoms))
        
        if len(polygons) == 0:
            print("  ⚠️ 폴리곤이 없습니다.")
            return []
        
        areas = np.array([poly.area for poly in polygons])
        std_dev = np.std(areas)
        mean_area = np.mean(areas)
        
        print(f"  📈 면적 통계: 평균={mean_area:.2f}, 표준편차={std_dev:.2f}")
        
        if std_dev >= 0.5 * mean_area:
            threshold_area = np.percentile(areas, self.area_percentile)
            filtered_polygons = [poly for poly in polygons if poly.area >= threshold_area]
            print(f"  ✂️ 면적 필터링: {len(polygons)} → {len(filtered_polygons)}")
            return filtered_polygons
        else:
            print(f"  ✅ 편차가 작아 필터링 생략 ({len(polygons)}개 유지)")
            return polygons
    
    def _detect_curves_from_boundaries(self, polygons):
        print("📐 경계선 커브 검출...")
        curve_points = []
        
        for poly in polygons:
            if poly.geom_type == 'Polygon':
                curves = self._detect_sensitive_curve_from_boundary(poly.exterior)
                curve_points.extend(curves)
        
        print(f"  📍 검출된 커브 후보: {len(curve_points)}개")
        return curve_points
    
    def _detect_sensitive_curve_from_boundary(self, line):
        coords = np.array(line.coords)
        if len(coords) < 3:
            return []
        
        deltas = np.diff(coords, axis=0)
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        angle_derivative = np.gradient(angles)
        curve_indices = np.where(np.abs(angle_derivative) > self.derivative_threshold)[0] + 1
        
        return [Point(coords[i]) for i in curve_indices if 0 < i < len(coords) - 1]
    
    def _extract_skeleton(self, polygons):
        print("🦴 스켈레톤 추출...")
        
        if len(polygons) == 0:
            return {'skeleton': np.array([]), 'transform': None, 'skeleton_points': []}
        
        all_geoms = gpd.GeoSeries(polygons)
        bounds = all_geoms.total_bounds
        x_min, y_min, x_max, y_max = bounds
        
        width = int((x_max - x_min) / self.skeleton_resolution)
        height = int((y_max - y_min) / self.skeleton_resolution)
        width = max(width, 10)
        height = max(height, 10)
        
        transform = from_origin(x_min, y_max, self.skeleton_resolution, self.skeleton_resolution)
        
        shapes = [(geom, 1) for geom in polygons if geom.is_valid]
        raster = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype='uint8')
        skeleton = skeletonize(raster > 0)
        
        rows, cols = np.where(skeleton)
        skeleton_points = []
        
        for row, col in zip(rows, cols):
            x = x_min + (col + 0.5) * self.skeleton_resolution
            y = y_max - (row + 0.5) * self.skeleton_resolution
            skeleton_points.append([x, y])
        
        print(f"  ✅ 스켈레톤 포인트: {len(skeleton_points)}개")
        
        return {
            'skeleton': skeleton,
            'transform': transform,
            'skeleton_points': skeleton_points,
            'bounds': (x_min, y_min, x_max, y_max)
        }
    
    def _detect_intersections_from_skeleton(self, polygons, skeleton_data=None):
        print("🔀 교차점 검출 (NetworkX)...")
        
        if skeleton_data is None:
            skeleton_data = self._extract_skeleton(polygons)
        
        skeleton = skeleton_data['skeleton']
        transform = skeleton_data['transform']
        
        # skeleton이 비어있는지 확인
        if skeleton.size == 0:
            return []
        
        G = self._skeleton_to_graph(skeleton)
        intersections = []
        
        for node, degree in G.degree():
            if degree >= 3:
                col, row = node
                x = transform.c + col * transform.a
                y = transform.f + row * transform.e
                intersections.append(Point(x, y))
        
        print(f"  🔍 검출된 교차점: {len(intersections)}개")
        return intersections
    
    def _skeleton_to_graph(self, skeleton):
        G = nx.Graph()
        rows, cols = np.where(skeleton)
        height, width = skeleton.shape
        
        for y, x in zip(rows, cols):
            G.add_node((x, y))
        
        for y, x in zip(rows, cols):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx_new = y + dy, x + dx
                    if 0 <= nx_new < width and 0 <= ny < height and skeleton[ny, nx_new]:
                        G.add_edge((x, y), (nx_new, ny))
        
        return G
    
    def _enhance_curve_filtering(self, curves, intersections, skeleton_data):
        """기존 거리 기반 + 연결성 분석 조합"""
        # 1단계: 기존 거리 기반 필터링 (너무 가까운 것들)
        step1_filtered = []
        for curve in curves:
            too_close = False
            for intersection in intersections:
                if curve.distance(intersection) <= self.intersection_radius:  # 교차점 바로 근처
                    too_close = True
                    break
            if not too_close:
                step1_filtered.append(curve)
        
        print(f"  🔄 1단계 거리 필터링: {len(curves)} → {len(step1_filtered)}")
        
        # 2단계: 연결성 분석 기반 필터링
        step2_filtered = self._filter_curves_with_connectivity_analysis(
            step1_filtered, intersections, skeleton_data
        )
        
        return step2_filtered
    
    def _filter_curves_with_connectivity_analysis(self, curves, intersections, skeleton_data):
        """교차점 간 연결성을 분석하여 경로상의 커브들을 제거"""
        if len(curves) == 0 or len(intersections) == 0:
            return curves
        
        print("🔗 교차점 간 연결성 분석 시작...")
        
        # 스켈레톤 그래프 생성
        skeleton = skeleton_data['skeleton']
        transform = skeleton_data['transform']
        G = self._skeleton_to_graph(skeleton)
        
        if G.number_of_nodes() == 0:
            return curves
        
        # 교차점들을 그래프 노드로 변환
        intersection_nodes = []
        for intersection in intersections:
            # 월드 좌표를 그래프 노드 좌표로 변환
            col = int((intersection.x - transform.c) / transform.a)
            row = int((transform.f - intersection.y) / abs(transform.e))
            
            # 가장 가까운 실제 노드 찾기
            closest_node = None
            min_dist = float('inf')
            for node in G.nodes():
                node_dist = ((node[0] - col)**2 + (node[1] - row)**2)**0.5
                if node_dist < min_dist:
                    min_dist = node_dist
                    closest_node = node
            
            if closest_node and min_dist < 5:  # 5픽셀 이내
                intersection_nodes.append(closest_node)
        
        print(f"  📍 매핑된 교차점 노드: {len(intersection_nodes)}개")
        
        # 교차점 간 모든 경로 찾기
        connection_paths = []
        for i in range(len(intersection_nodes)):
            for j in range(i + 1, len(intersection_nodes)):
                try:
                    path = nx.shortest_path(G, intersection_nodes[i], intersection_nodes[j])
                    connection_paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        print(f"  🛤️ 찾은 연결 경로: {len(connection_paths)}개")
        
        # 경로상의 월드 좌표들 수집
        path_world_coords = set()
        for path in connection_paths:
            for node in path:
                col, row = node
                world_x = transform.c + col * transform.a
                world_y = transform.f + row * transform.e
                path_world_coords.add((world_x, world_y))
        
        # 커브들이 경로상에 있는지 확인
        filtered_curves = []
        removed_count = 0
        
        for curve in curves:
            is_on_path = False
            curve_x, curve_y = curve.x, curve.y
            
            # 커브가 경로상의 어떤 점과 가까운지 확인
            for path_x, path_y in path_world_coords:
                distance = ((curve_x - path_x)**2 + (curve_y - path_y)**2)**0.5
                if distance <= self.curve_intersection_distance:
                    is_on_path = True
                    break
            
            if not is_on_path:
                filtered_curves.append(curve)
            else:
                removed_count += 1
        
        print(f"  🚫 연결 경로상 커브 제거: {len(curves)} → {len(filtered_curves)} (제거: {removed_count}개)")
        return filtered_curves
    
    def _cluster_points(self, points, radius):
        if len(points) == 0:
            return []
        
        coords = np.array([(p.x, p.y) for p in points])
        clustering = DBSCAN(eps=radius, min_samples=1).fit(coords)
        labels = clustering.labels_
        
        centers = []
        for label in set(labels):
            cluster_points = coords[labels == label]
            center = cluster_points.mean(axis=0)
            centers.append(Point(center))
        
        return centers
    
    def _detect_endpoints(self, skeleton_points):
        if len(skeleton_points) < 2:
            return []
        
        endpoints = [skeleton_points[0], skeleton_points[-1]]
        
        if len(endpoints) == 2:
            dist = np.linalg.norm(np.array(endpoints[0]) - np.array(endpoints[1]))
            if dist < 30:
                endpoints = [endpoints[0]]
        
        return endpoints