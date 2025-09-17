# 🎯 Point-Sample 방식 거리 분석 시스템

**Process3에서 AI 분석 완료된 점들의 점간 거리 분석 및 시각화**

## 🔍 개요

기존의 모든 점을 연결하는 방식에서 **Point-Sample 방식**으로 개선하여, **의미있는 연결만** 표시하는 고도화된 거리 분석 시스템입니다.

## 📊 개선 전후 비교

| 구분 | 기존 방식 | Point-Sample 방식 |
|------|----------|------------------|
| **거리 범위** | 15m~300m (너무 넓음) | 15m~50m (의미있는 범위) |
| **연결성 체크** | 관대한 체크 | 엄격한 시통성 체크 |
| **결과 예시** | 15개 점 → 77개 연결 | 15개 점 → 8개 연결 |
| **평균 거리** | 158.9m | 33.4m |
| **의미성** | 모든 점이 연결됨 | 시각적으로 통하는 점만 |

## 🏗️ 시스템 구조

### 1. **모듈화된 구조**
```
src/distance_analysis/
├── __init__.py                    # 모듈 초기화
├── distance_calculator.py         # 메인 통합 계산기
├── network_connectivity.py        # NetworkX 기반 네트워크 분석
├── visual_connectivity.py         # 시각적 연결성 분석 (시통성)
└── importance_scorer.py           # 중요도 점수 계산
```

### 2. **4단계 분석 파이프라인**

#### **1단계: 거리 필터링**
- **범위**: 15m~50m (Point-sample 방식)
- **로직**: 너무 가깝거나 먼 점들 제외

#### **2단계: 네트워크 연결성 분석**
- **기준**: 도로 경계선 20m 이내 + 70% 도로 커버리지
- **방법**: NetworkX 그래프 + 도로 네트워크 따라가기

#### **3단계: 시각적 연결성 분석**
- **시통성 체크**: 두 점 사이가 시각적으로 통하는지 확인
- **방해점 체크**: 8m 버퍼 내 3개 이하 스켈레톤 점
- **스켈레톤 따라가기**: 70% 이상 10m 내 스켈레톤 커버

#### **4단계: 중요도 기반 우선순위**
- **점 중요도**: 카테고리별 가중치 + 스켈레톤 중요도
- **거리 우선순위**: 25m~40m 최적, 15m~50m 허용
- **최종 선택**: 상위 20개 + 60% 이상 우선순위

## 🎯 Point-Sample 핵심 로직

### **시각적 연결성 체크**
```python
def check_visual_connectivity(self, point1, point2):
    # 1. 거리 체크 (15m~50m)
    if not (15 <= distance <= 50):
        return False
    
    # 2. 도로 내부 연결성 체크
    if not self._check_road_containment(point1, point2):
        return False
    
    # 3. 방해점 체크 (8m 버퍼 내 3개 이하)
    if self._has_blocking_points(point1, point2):
        return False
    
    # 4. 스켈레톤 따라가기 체크 (70% 커버리지)
    if not self._follows_skeleton_path(point1, point2):
        return False
    
    return True
```

### **우선순위 계산**
```python
def calculate_connection_priority(self, point1, point2, category1, category2):
    # 점 중요도 (교차점 > 커브점 > 끝점)
    avg_importance = (importance1 + importance2) / 2.0
    
    # 거리 우선순위 (25-40m 최적)
    if 25 <= distance <= 40:
        distance_priority = 1.0
    elif 15 <= distance < 25 or 40 < distance <= 50:
        distance_priority = 0.8
    
    # 카테고리 조합 우선순위
    category_priority = get_category_priority(category1, category2)
    
    # 가중 평균 (거리 50%, 중요도 30%, 카테고리 20%)
    final_priority = (avg_importance * 0.3 + 
                     distance_priority * 0.5 + 
                     category_priority * 0.2)
```

## 🚀 사용 방법

### **1. Process3에서 사용**
1. **파일 선택**: 도로망 또는 지구계 파일 로드
2. **AI 분석**: 점 추출 완료 후
3. **거리 분석**: "🔍 점간 거리 분석 시작" 클릭
4. **인터렉티브 편집**: 점 추가/삭제 시 실시간 업데이트

### **2. 독립 모듈로 사용**
```python
from src.distance_analysis import AdvancedDistanceCalculator

# 초기화
calculator = AdvancedDistanceCalculator(
    skeleton_points=skeleton_points,
    road_polygons=road_polygons,
    max_distance=50.0,  # Point-sample 방식
    min_distance=15.0
)

# 점 데이터 준비
points_data = {
    'intersection': [(x1, y1), (x2, y2), ...],
    'curve': [(x3, y3), (x4, y4), ...],
    'endpoint': [(x5, y5), (x6, y6), ...]
}

# 분석 실행
result = calculator.calculate_optimal_distances(points_data)

# Canvas 표시용 데이터
display_data = calculator.get_canvas_display_data()

# 통계 정보
stats_text = calculator.get_statistics_text()
```

## 📋 분석 결과

### **연결 정보**
- **총 연결 수**: 의미있는 연결만 표시
- **점선 표시**: Canvas에서 시각적 표시
- **거리 텍스트**: 중간 지점에 거리 표시

### **통계 정보**
```
거리 분석 결과:
총 연결: 8개
평균 거리: 33.4m
거리 범위: 22.4m ~ 50.0m
평균 우선순위: 0.73

거리 분포:
• 단거리 (≤30m): 4개
• 중거리 (30~45m): 3개
• 장거리 (>45m): 1개
```

### **우선순위 정보**
- **0.8~1.0**: 최우선 연결 (교차점 간 최적 거리)
- **0.6~0.8**: 중요 연결 (교차점-커브점)
- **0.4~0.6**: 일반 연결 (커브점-끝점)

## 🎯 Point-Sample 방식의 장점

### **1. 의미있는 연결만 표시**
- **기존**: 모든 점이 연결되어 복잡함
- **개선**: 시각적으로 통하는 점들만 연결

### **2. 실제 도로망 구조 반영**
- **스켈레톤 기반**: 도로 중심선을 따라 연결
- **도로 내부**: 연결선이 도로 영역 내부를 지나감

### **3. 인터렉티브 분석**
- **실시간 업데이트**: 점 수정 시 즉시 재계산
- **우선순위 기반**: 중요한 연결 위주로 표시

### **4. 성능 최적화**
- **공간 인덱스**: 100m 그리드 기반 효율적 검색
- **단계별 필터링**: 불필요한 계산 최소화

## 🛠️ 설정 가능한 파라미터

```python
# 거리 범위 조정
calculator.update_parameters(
    max_distance=60.0,  # 최대 거리 (기본 50m)
    min_distance=10.0   # 최소 거리 (기본 15m)
)

# 우선순위 임계값 조정
priority_connections = scorer.get_top_connections(
    connections,
    top_n=30,           # 상위 N개 (기본 20개)
    min_priority=0.5    # 최소 우선순위 (기본 0.6)
)
```

## 🔧 문제 해결

### **연결이 너무 적을 때**
1. **거리 범위 확대**: `max_distance=70.0`
2. **우선순위 완화**: `min_priority=0.4`
3. **연결성 체크 완화**: `coverage_ratio >= 0.6`

### **연결이 너무 많을 때**
1. **거리 범위 축소**: `max_distance=40.0`
2. **우선순위 강화**: `min_priority=0.7`
3. **상위 개수 제한**: `top_n=15`

---

**Point-Sample 방식으로 더욱 정확하고 의미있는 거리 분석이 가능합니다!** 🎯 