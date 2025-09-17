import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union, linemerge
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)

class RoadProcessor:
    def __init__(self):
        self.min_segment_length = 10
        self.merge_threshold = 20
        self.cluster_eps = 30
        
    def post_process_detections(self, skeleton, labels):
        return {
            'intersection': self.refine_intersections(labels.get('intersection', [])),
            'curve': self.refine_curves(skeleton, labels.get('curve', [])),
            'endpoint': self.refine_endpoints(skeleton, labels.get('endpoint', []))
        }
    
    def refine_intersections(self, intersections):
        if len(intersections) < 2:
            return intersections
        
        points = np.array(intersections)
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=1).fit(points)
        
        refined = []
        for cluster_id in set(clustering.labels_):
            cluster_points = points[clustering.labels_ == cluster_id]
            refined.append(tuple(cluster_points.mean(axis=0)))
        
        return refined
    
    def calculate_curvature(self, point, skeleton, window=10):
        sk = np.array(skeleton, dtype=float) if not isinstance(skeleton, np.ndarray) else skeleton
        distances = np.linalg.norm(sk - point, axis=1)
        nearby_indices = np.where(distances < window * 3)[0]
        
        if len(nearby_indices) < 3:
            return 0.0
        
        nearby_points = sk[nearby_indices]
        p1, p2, p3 = nearby_points[0], nearby_points[len(nearby_points) // 2], nearby_points[-1]
        
        a, b, c = np.linalg.norm(p2 - p1), np.linalg.norm(p3 - p2), np.linalg.norm(p3 - p1)
        s = (a + b + c) / 2
        area = max(s * (s - a) * (s - b) * (s - c), 1e-6) ** 0.5
        R = (a * b * c) / (4 * area)
        
        return 1.0 / R
    
    def refine_curves(self, skeleton, curves):
        if not curves:
            return curves
        
        sk = np.array(skeleton, dtype=float) if not isinstance(skeleton, np.ndarray) else skeleton
        curve_points = np.array(curves, dtype=float)
        
        refined = [tuple(point) for point in curve_points if self.calculate_curvature(point, sk) > 0.01]
        return self.remove_close_points(refined, self.merge_threshold)
    
    def refine_endpoints(self, skeleton, endpoints):
        if not endpoints:
            return endpoints
        
        return [tuple(point) for point in endpoints if self.is_true_endpoint(point, skeleton)]
    
    def is_true_endpoint(self, point, skeleton, threshold=30):
        distances = np.linalg.norm(skeleton - point, axis=1)
        return np.sum(distances < threshold) <= 3
    
    def remove_close_points(self, points, threshold):
        if len(points) < 2:
            return points
        
        points_array = np.array(points)
        keep = np.ones(len(points), dtype=bool)
        
        for i in range(len(points)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(points)):
                if keep[j] and np.linalg.norm(points_array[i] - points_array[j]) < threshold:
                    keep[j] = False
        
        return [points[i] for i in range(len(points)) if keep[i]]
    
    def segment_road_network(self, gdf, intersections):
        segments = []
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            if isinstance(geom, LineString):
                segments.extend(self.split_line_at_points(geom, intersections))
            elif isinstance(geom, MultiLineString):
                for line in geom.geoms:
                    segments.extend(self.split_line_at_points(line, intersections))
        
        return segments
    
    def split_line_at_points(self, line, points, tolerance=20):
        if not points:
            return [line]
        
        segments = [line]
        
        for point in points:
            new_segments = []
            p = Point(point)
            
            for segment in segments:
                if segment.distance(p) < tolerance:
                    split_point = segment.interpolate(segment.project(p))
                    coords = list(segment.coords)
                    
                    split_idx = min(range(len(coords) - 1), 
                                  key=lambda i: LineString([coords[i], coords[i+1]]).distance(split_point)) + 1
                    
                    if 0 < split_idx < len(coords):
                        seg1 = LineString(coords[:split_idx] + [split_point.coords[0]])
                        seg2 = LineString([split_point.coords[0]] + coords[split_idx:])
                        
                        new_segments.extend([seg for seg in [seg1, seg2] if seg.length > self.min_segment_length])
                    else:
                        new_segments.append(segment)
                else:
                    new_segments.append(segment)
            
            segments = new_segments
        
        return segments
    
    def calculate_road_statistics(self, gdf, labels):
        total_length = sum(row.geometry.length for idx, row in gdf.iterrows() if row.geometry)
        segments = self.segment_road_network(gdf, labels.get('intersection', []))
        
        stats = {
            'total_length': total_length,
            'num_segments': len(segments),
            'num_intersections': len(labels.get('intersection', [])),
            'num_curves': len(labels.get('curve', [])),
            'num_endpoints': len(labels.get('endpoint', [])),
            'avg_segment_length': np.mean([seg.length for seg in segments]) if segments else 0,
            'network_density': len(labels.get('intersection', [])) / (total_length / 1000) if total_length > 0 else 0
        }
        
        return stats
    
    def export_to_shapefile(self, segments, labels, output_path):
        try:
            segment_gdf = gpd.GeoDataFrame(
                {'geometry': segments, 'type': 'road_segment'},
                crs='EPSG:4326'
            )
            
            features = []
            for label_type, points in labels.items():
                features.extend([{'geometry': Point(p), 'type': label_type} for p in points])
            
            if features:
                feature_gdf = gpd.GeoDataFrame(features, crs='EPSG:4326')
                base_path = output_path.rsplit('.', 1)[0]
                segment_gdf.to_file(f"{base_path}_segments.gpkg", driver='GPKG')
                feature_gdf.to_file(f"{base_path}_features.gpkg", driver='GPKG')
                logger.info(f"결과 저장 완료: {base_path}")
                return True
                
        except Exception as e:
            logger.error(f"Shapefile 내보내기 실패: {e}")
            return False