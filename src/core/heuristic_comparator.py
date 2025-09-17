# src/core/heuristic_comparator.py
import numpy as np
from typing import Dict, List, Tuple

class HeuristicComparator:
    def __init__(self):
        self.confidence_weights = {
            'intersection': 0.9,
            'curve': 0.7,
            'endpoint': 0.8
        }
    
    def compare_results(self, heuristic_results: Dict, user_results: Dict) -> Dict:
        comparison = {
            'added': {cat: [] for cat in ['intersection', 'curve', 'endpoint']},
            'removed': {cat: [] for cat in ['intersection', 'curve', 'endpoint']},
            'kept': {cat: [] for cat in ['intersection', 'curve', 'endpoint']},
            'improvement_score': 0.0
        }
        
        for category in ['intersection', 'curve', 'endpoint']:
            h_points = set(map(tuple, heuristic_results.get(category, [])))
            u_points = set(map(tuple, user_results.get(category, [])))
            
            comparison['added'][category] = list(u_points - h_points)
            comparison['removed'][category] = list(h_points - u_points)
            comparison['kept'][category] = list(h_points & u_points)
        
        comparison['improvement_score'] = self._calculate_improvement_score(comparison)
        return comparison
    
    def _calculate_improvement_score(self, comparison: Dict) -> float:
        score = 0.0
        for cat in ['intersection', 'curve', 'endpoint']:
            score += len(comparison['added'][cat]) * 1.0
            score += len(comparison['removed'][cat]) * 0.8
            score -= len(comparison['kept'][cat]) * 0.1
        return score
    
    def get_heuristic_features(self, x: float, y: float, heuristic_results: Dict, 
                              skeleton: List, radius: float = 50.0) -> List[float]:
        heuristic_class = self._get_heuristic_class(x, y, heuristic_results)
        heuristic_onehot = [0, 0, 0, 0]
        heuristic_onehot[heuristic_class] = 1
        
        confidence = self._get_heuristic_confidence(x, y, heuristic_results, heuristic_class)
        
        nearby_counts = self._count_nearby_heuristics(x, y, heuristic_results, radius)
        
        nearest_dist = self._nearest_heuristic_distance(x, y, heuristic_results)
        
        density = self._calculate_heuristic_density(x, y, heuristic_results, radius)
        
        return heuristic_onehot + [confidence] + nearby_counts + [nearest_dist, density]
    
    def _get_heuristic_class(self, x: float, y: float, heuristic_results: Dict) -> int:
        threshold = 5.0
        for i, cat in enumerate(['intersection', 'curve', 'endpoint']):
            for px, py in heuristic_results.get(cat, []):
                if np.sqrt((x - px)**2 + (y - py)**2) < threshold:
                    return i + 1
        return 0
    
    def _get_heuristic_confidence(self, x: float, y: float, heuristic_results: Dict, 
                                  heuristic_class: int) -> float:
        if heuristic_class == 0:
            return 0.0
        categories = ['intersection', 'curve', 'endpoint']
        return self.confidence_weights.get(categories[heuristic_class - 1], 0.5)
    
    def _count_nearby_heuristics(self, x: float, y: float, heuristic_results: Dict, 
                                 radius: float) -> List[float]:
        counts = []
        for cat in ['intersection', 'curve', 'endpoint']:
            count = 0
            for px, py in heuristic_results.get(cat, []):
                if np.sqrt((x - px)**2 + (y - py)**2) <= radius:
                    count += 1
            counts.append(float(count))
        return counts
    
    def _nearest_heuristic_distance(self, x: float, y: float, heuristic_results: Dict) -> float:
        min_dist = 1000.0
        for cat in ['intersection', 'curve', 'endpoint']:
            for px, py in heuristic_results.get(cat, []):
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                min_dist = min(min_dist, dist)
        return min_dist
    
    def _calculate_heuristic_density(self, x: float, y: float, heuristic_results: Dict, 
                                    radius: float) -> float:
        total_count = 0
        for cat in ['intersection', 'curve', 'endpoint']:
            for px, py in heuristic_results.get(cat, []):
                if np.sqrt((x - px)**2 + (y - py)**2) <= radius:
                    total_count += 1
        return total_count / (np.pi * radius * radius) * 1000