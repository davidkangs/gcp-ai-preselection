# src/ui/canvas_widget.py - 최적화된 완전한 버전

import numpy as np
from pathlib import Path
import logging
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

# PyQt5 imports
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsEllipseItem, QGraphicsRectItem,
    QGraphicsPolygonItem, QGraphicsPathItem, QMessageBox,
    QCheckBox, QPushButton, QSlider, QGroupBox
)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QPen, QBrush, QColor, QPainterPath, QPolygonF, QWheelEvent,
    QFont, QTransform, QPainter
)

logger = logging.getLogger(__name__)


class InteractiveCanvas(QGraphicsView):
    """인터랙티브 캔버스 - 도로망 시각화 및 라벨링"""
    
    # 시그널 정의
    point_added = pyqtSignal(str, float, float)
    point_removed = pyqtSignal(str, float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Scene 설정
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # 뷰 설정 - 성능 최적화
        self.setRenderHint(QPainter.Antialiasing, False)  # 안티앨리어싱 끄기
        self.setRenderHint(QPainter.TextAntialiasing, False)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)  # 스마트 업데이트
        
        # 드래그 모드 설정
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # 캐싱 활성화
        self.setCacheMode(QGraphicsView.CacheBackground)
        
        # 레이어 가시성
        self.show_road_layer = True
        self.show_skeleton_layer = True
        self.show_points_layer = True
        
        # 레이어 투명도
        self.road_opacity = 0.3
        self.skeleton_opacity = 1.0
        
        # 데이터 저장
        self.road_geometry = None
        self.skeleton_points = None
        self.points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        
        # 삭제된 포인트 추적
        self.deleted_points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        
        # AI 관련 속성
        self.ai_points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        self.deleted_points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        self.deleted_points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        self.ai_assist_mode = False
        self.show_ai_predictions = True
        self.ai_confidence_threshold = 0.7
        
        # 그래픽 아이템 그룹
        self.road_items = []
        self.skeleton_items = []  # 스켈레톤을 여러 아이템으로
        self.point_items = []
        self.ai_point_items = []
        
        # 줌 레벨
        self.zoom_factor = 1.15
        
        # 색상 정의
        self.colors = {
            'road': QColor(200, 200, 200),
            'road_stroke': QColor(150, 150, 150),
            'skeleton': QColor(50, 50, 200),
            'intersection': QColor(255, 0, 0),
            'curve': QColor(0, 0, 255),
            'endpoint': QColor(0, 255, 0),
            'ai_intersection': QColor(128, 0, 128),
            'ai_curve': QColor(255, 165, 0),
            'ai_endpoint': QColor(139, 69, 19)
        }
        
        # 성능 설정
        self.max_skeleton_points = 5000  # 최대 표시 포인트 수
        self.skeleton_sampling_rate = 1  # 샘플링 비율
    
    def clear_canvas(self):
        """캔버스 완전 초기화"""
        # 씬 클리어
        self.scene.clear()
        
        # 모든 데이터 초기화
        self.road_geometry = None
        self.skeleton_points = None
        self.points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        self.ai_points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        self.deleted_points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        
        # 아이템 리스트 초기화
        self.road_items = []
        self.skeleton_items = []
        self.point_items = []
        self.ai_point_items = []
        
        # 줌 리셋
        self.resetTransform()
    
    def set_road_geometry(self, geometry):
        """원본 도로 geometry 설정"""
        self.road_geometry = geometry
        self.draw_road_layer()
    
    def draw_road_layer(self):
        """도로망 레이어 그리기"""
        # 씬 업데이트 일시 중지
        self.setUpdatesEnabled(False)
        
        try:
            # 기존 도로 아이템 제거
            for item in self.road_items:
                self.scene.removeItem(item)
            self.road_items.clear()
            
            if not self.road_geometry or not self.show_road_layer:
                return
            
            # Geometry 타입에 따라 그리기
            if hasattr(self.road_geometry, '__iter__'):
                for geom in self.road_geometry:
                    self._draw_single_geometry(geom)
            else:
                self._draw_single_geometry(self.road_geometry)
        finally:
            self.setUpdatesEnabled(True)
    
    def _draw_single_geometry(self, geom):
        """단일 geometry 그리기"""
        if geom.geom_type == 'LineString':
            path = QPainterPath()
            coords = list(geom.coords)
            path.moveTo(coords[0][0], coords[0][1])
            for x, y in coords[1:]:
                path.lineTo(x, y)
            
            item = QGraphicsPathItem(path)
            pen = QPen(self.colors['road_stroke'], 5)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            item.setPen(pen)
            item.setOpacity(self.road_opacity)
            item.setZValue(-2)
            self.scene.addItem(item)
            self.road_items.append(item)
            
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                self._draw_single_geometry(line)
                
        elif geom.geom_type == 'Polygon':
            path = QPainterPath()
            exterior = list(geom.exterior.coords)
            path.moveTo(exterior[0][0], exterior[0][1])
            for x, y in exterior[1:]:
                path.lineTo(x, y)
            path.closeSubpath()
            
            item = QGraphicsPathItem(path)
            item.setPen(QPen(self.colors['road_stroke'], 2))
            item.setBrush(QBrush(self.colors['road']))
            item.setOpacity(self.road_opacity)
            item.setZValue(-2)
            self.scene.addItem(item)
            self.road_items.append(item)
            
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                self._draw_single_geometry(poly)
    
    def set_skeleton(self, skeleton_points):
        """스켈레톤 설정 및 표시"""
        self.skeleton_points = skeleton_points
        
        # numpy array나 list 모두 처리
        if skeleton_points is not None and len(skeleton_points) > 0:
            # 포인트 수가 많으면 샘플링
            if len(skeleton_points) > self.max_skeleton_points:
                self.skeleton_sampling_rate = len(skeleton_points) // self.max_skeleton_points
            else:
                self.skeleton_sampling_rate = 1
        
        self.draw_skeleton_layer()
        
        # Scene 범위 조정
        if self.skeleton_items:
            # 모든 아이템의 경계 상자 계산
            rect = QRectF()
            for item in self.skeleton_items:
                rect = rect.united(item.boundingRect())
            self.scene.setSceneRect(rect.adjusted(-50, -50, 50, 50))
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def draw_skeleton_layer(self):
        """스켈레톤 레이어 그리기 - 최적화 버전"""
        # 씬 업데이트 일시 중지
        self.setUpdatesEnabled(False)
        
        try:
            # 기존 아이템 제거
            for item in self.skeleton_items:
                self.scene.removeItem(item)
            self.skeleton_items.clear()
            
            # numpy array 체크 수정 - 올바른 방법으로 체크
            if self.skeleton_points is None or len(self.skeleton_points) == 0 or not self.show_skeleton_layer:
                return
            
            # 샘플링된 포인트 사용
            sampled_points = self.skeleton_points[::self.skeleton_sampling_rate]
            
            # 포인트 수에 따라 그리기 방식 결정
            if len(sampled_points) < 1000:
                # 포인트가 적으면 선으로 연결
                self._draw_skeleton_as_lines(sampled_points)
            else:
                # 포인트가 많으면 점으로만 표시
                self._draw_skeleton_as_dots(sampled_points)
                
        finally:
            self.setUpdatesEnabled(True)
    
    def _draw_skeleton_as_lines(self, points):
        """스켈레톤을 선으로 그리기"""
        path = QPainterPath()
        
        # 연속된 점들을 선으로 연결
        drawn_points = set()
        remaining_indices = list(range(len(points)))
        
        while remaining_indices:
            # 새로운 경로 시작
            start_idx = remaining_indices.pop(0)
            current_idx = start_idx
            path.moveTo(points[current_idx][0], points[current_idx][1])
            drawn_points.add(current_idx)
            
            # 가까운 점들을 연결
            while True:
                current_point = points[current_idx]
                next_idx = None
                min_dist = 50  # 최대 연결 거리
                
                for idx in remaining_indices:
                    dist = np.linalg.norm(np.array(current_point) - np.array(points[idx]))
                    if dist < min_dist:
                        min_dist = dist
                        next_idx = idx
                
                if next_idx is not None:
                    path.lineTo(points[next_idx][0], points[next_idx][1])
                    current_idx = next_idx
                    remaining_indices.remove(next_idx)
                    drawn_points.add(next_idx)
                else:
                    break
        
        # 경로 아이템 생성
        item = QGraphicsPathItem(path)
        pen = QPen(self.colors['skeleton'], 2)
        item.setPen(pen)
        item.setOpacity(self.skeleton_opacity)
        item.setZValue(0)
        self.scene.addItem(item)
        self.skeleton_items.append(item)
    
    def _draw_skeleton_as_dots(self, points):
        """스켈레톤을 점으로 그리기"""
        # 점들을 하나의 경로로 그리기 (더 효율적)
        for i in range(0, len(points), 100):  # 100개씩 묶어서 처리
            path = QPainterPath()
            for j in range(i, min(i + 100, len(points))):
                x, y = points[j]
                path.addEllipse(x - 1, y - 1, 2, 2)
            
            item = QGraphicsPathItem(path)
            item.setPen(QPen(Qt.NoPen))
            item.setBrush(QBrush(self.colors['skeleton']))
            item.setOpacity(self.skeleton_opacity)
            item.setZValue(0)
            self.scene.addItem(item)
            self.skeleton_items.append(item)
    
    def update_display(self):
        """전체 디스플레이 업데이트 - 최적화"""
        # 씬 업데이트 일시 중지
        self.setUpdatesEnabled(False)
        
        try:
            self.draw_road_layer()
            self.draw_skeleton_layer()
            self.draw_points_layer()
            
            if self.show_ai_predictions and self.ai_points:
                self.draw_ai_predictions()
        finally:
            # 업데이트 재개
            self.setUpdatesEnabled(True)
            self.scene.update()
    
    def draw_points_layer(self):
        """포인트 레이어 그리기"""
        # 기존 포인트 아이템 제거
        for item in self.point_items:
            self.scene.removeItem(item)
        self.point_items.clear()
        
        if not self.show_points_layer:
            return
        
        # 교차점 (빨간 원)
        for x, y in self.points.get('intersection', []):
            item = self.scene.addEllipse(
                x - 6, y - 6, 12, 12,
                QPen(self.colors['intersection'], 2),
                QBrush(self.colors['intersection'])
            )
            item.setZValue(10)
            self.point_items.append(item)
        
        # 커브 (파란 사각형)
        for x, y in self.points.get('curve', []):
            item = self.scene.addRect(
                x - 6, y - 6, 12, 12,
                QPen(self.colors['curve'], 2),
                QBrush(self.colors['curve'])
            )
            item.setZValue(10)
            self.point_items.append(item)
        
        # 끝점 (녹색 별)
        for x, y in self.points.get('endpoint', []):
            star = QPolygonF()
            for i in range(5):
                angle = i * 144 * 3.14159 / 180
                if i % 2 == 0:
                    star.append(QPointF(x + 8 * np.cos(angle), y + 8 * np.sin(angle)))
                else:
                    star.append(QPointF(x + 4 * np.cos(angle), y + 4 * np.sin(angle)))
            
            item = self.scene.addPolygon(
                star,
                QPen(self.colors['endpoint'], 2),
                QBrush(self.colors['endpoint'])
            )
            item.setZValue(10)
            self.point_items.append(item)
    
    def toggle_road_layer(self, visible):
        """도로망 레이어 표시/숨김"""
        self.show_road_layer = visible
        self.update_display()
    
    def toggle_skeleton_layer(self, visible):
        """스켈레톤 레이어 표시/숨김"""
        self.show_skeleton_layer = visible
        self.update_display()
    
    def toggle_points_layer(self, visible):
        """포인트 레이어 표시/숨김"""
        self.show_points_layer = visible
        self.update_display()
    
    def set_road_opacity(self, opacity):
        """도로망 레이어 투명도 설정"""
        self.road_opacity = opacity / 100.0
        for item in self.road_items:
            item.setOpacity(self.road_opacity)
    
    def set_skeleton_opacity(self, opacity):
        """스켈레톤 레이어 투명도 설정"""
        self.skeleton_opacity = opacity / 100.0
        for item in self.skeleton_items:
            item.setOpacity(self.skeleton_opacity)
    
    def add_point(self, category, x, y):
        """포인트 추가"""
        print(f"add_point 호출: {category} at ({x:.2f}, {y:.2f})")
        if category not in self.points:
            return
        
        # 중복 체크
        if not self.is_point_exists(category, x, y):
            self.points[category].append((x, y))
            self.point_added.emit(category, x, y)
            # 포인트 레이어만 업데이트
            self.draw_points_layer()
    
    def remove_nearest_point(self, x, y):
        """가장 가까운 포인트 제거"""
        min_dist = float('inf')
        remove_category = None
        remove_idx = None
        
        # 모든 카테고리에서 가장 가까운 포인트 찾기
        for category, points in self.points.items():
            for i, (px, py) in enumerate(points):
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                if dist < min_dist:
                    min_dist = dist
                    remove_category = category
                    remove_idx = i
        
        # 제거
        if remove_category and remove_idx is not None and min_dist < 30:
            removed = self.points[remove_category].pop(remove_idx)
            # 삭제된 포인트 추적
            self.deleted_points[remove_category].append(removed)
            self.point_removed.emit(remove_category, removed[0], removed[1])
            # 포인트 레이어만 업데이트
            self.draw_points_layer()
            return True
        
        return False
    
    def draw_ai_predictions(self):
        """AI 예측 결과 그리기"""
        # 기존 AI 포인트 아이템 제거
        for item in self.ai_point_items:
            self.scene.removeItem(item)
        self.ai_point_items.clear()
        
        # AI 교차점 (보라색 다이아몬드)
        for x, y in self.ai_points.get('intersection', []):
            diamond = QPolygonF([
                QPointF(x, y - 6),
                QPointF(x + 6, y),
                QPointF(x, y + 6),
                QPointF(x - 6, y)
            ])
            item = self.scene.addPolygon(
                diamond,
                QPen(self.colors['ai_intersection'], 2),
                QBrush(QColor(128, 0, 128, 100))
            )
            item.setZValue(8)
            self.ai_point_items.append(item)
        
        # AI 커브 (주황색 삼각형)
        for x, y in self.ai_points.get('curve', []):
            triangle = QPolygonF([
                QPointF(x, y - 5),
                QPointF(x + 5, y + 5),
                QPointF(x - 5, y + 5)
            ])
            item = self.scene.addPolygon(
                triangle,
                QPen(self.colors['ai_curve'], 2),
                QBrush(QColor(255, 165, 0, 100))
            )
            item.setZValue(8)
            self.ai_point_items.append(item)
        
        # AI 끝점 (갈색 육각형)
        for x, y in self.ai_points.get('endpoint', []):
            hexagon = QPolygonF()
            for i in range(6):
                angle = i * 60 * 3.14159 / 180
                hx = x + 5 * np.cos(angle)
                hy = y + 5 * np.sin(angle)
                hexagon.append(QPointF(hx, hy))
            item = self.scene.addPolygon(
                hexagon,
                QPen(self.colors['ai_endpoint'], 2),
                QBrush(QColor(139, 69, 19, 100))
            )
            item.setZValue(8)
            self.ai_point_items.append(item)
    
    def mousePressEvent(self, event):
        """마우스 클릭 이벤트"""
        if not self.skeleton_points:
            super().mousePressEvent(event)
            return
        
        scene_pos = self.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()
        
        # 디버그 출력
        print(f"클릭 위치: ({x:.2f}, {y:.2f})")
        
        # 스켈레톤에 스냅
        if self.skeleton_points is not None and len(self.skeleton_points) > 0:
            # numpy array나 list 모두 처리
            skeleton_array = np.array(self.skeleton_points) if not isinstance(self.skeleton_points, np.ndarray) else self.skeleton_points
            
            if len(skeleton_array) > 0:
                # 가장 가까운 스켈레톤 포인트 찾기
                distances = np.linalg.norm(skeleton_array - np.array([x, y]), axis=1)
                min_idx = np.argmin(distances)
                
                if distances[min_idx] < 30:  # 30픽셀 이내면 스냅
                    x, y = skeleton_array[min_idx]
                    print(f"스냅된 위치: ({x:.2f}, {y:.2f})")
        
        # Shift 키를 누르고 있으면 제거 모드
        if event.modifiers() & Qt.ShiftModifier:
            self.remove_nearest_point(x, y)
        else:
            # 좌클릭: 커브 추가
            if event.button() == Qt.LeftButton:
                print(f"커브 추가: ({x:.2f}, {y:.2f})")
                self.add_point('curve', x, y)
                
            # 우클릭: 끝점 추가
            elif event.button() == Qt.RightButton:
                print(f"끝점 추가: ({x:.2f}, {y:.2f})")
                self.add_point('endpoint', x, y)
        
        # 부모 클래스의 이벤트 처리도 호출
        super().mousePressEvent(event)
    def wheelEvent(self, event):
        """마우스 휠 이벤트 - 줌"""
        # 줌 인/아웃
        if event.angleDelta().y() > 0:
            self.scale(self.zoom_factor, self.zoom_factor)
        else:
            self.scale(1 / self.zoom_factor, 1 / self.zoom_factor)
    
    def keyPressEvent(self, event):
        """키보드 이벤트"""
        # D키: 가장 가까운 포인트 삭제
        if event.key() == Qt.Key_D:
            cursor_pos = self.mapFromGlobal(self.cursor().pos())
            if self.rect().contains(cursor_pos):  # 커서가 캔버스 내에 있는지 확인
                scene_pos = self.mapToScene(cursor_pos)
                self.remove_nearest_point(scene_pos.x(), scene_pos.y())
            
            # 포커스 복원
            self.setFocus()
            event.accept()
            return
            
        # A키: AI 예측 표시 토글
        elif event.key() == Qt.Key_A:
            self.show_ai_predictions = not self.show_ai_predictions
            self.update_display()
        
        # S키: AI 제안 수락
        elif event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            self.accept_ai_suggestions()
        
        # R키: AI 재예측 요청
        elif event.key() == Qt.Key_R and event.modifiers() == Qt.ControlModifier:
            if hasattr(self.parent().parent(), 'run_ai_detection'):
                self.parent().parent().run_ai_detection()
        
        # Space키: 화면 맞춤
        elif event.key() == Qt.Key_Space:
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
        # 1,2,3키: 레이어 토글
        elif event.key() == Qt.Key_1:
            self.toggle_road_layer(not self.show_road_layer)
        elif event.key() == Qt.Key_2:
            self.toggle_skeleton_layer(not self.show_skeleton_layer)
        elif event.key() == Qt.Key_3:
            self.toggle_points_layer(not self.show_points_layer)
        
        super().keyPressEvent(event)
    
    def find_nearest_ai_suggestion(self, x, y, threshold=30):
        """가장 가까운 AI 제안 포인트 찾기"""
        min_dist = float('inf')
        nearest_point = None
        
        for category in ['intersection', 'curve', 'endpoint']:
            for px, py in self.ai_points.get(category, []):
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                if dist < min_dist and dist < threshold:
                    min_dist = dist
                    nearest_point = (px, py)
        
        return nearest_point
    
    def accept_ai_suggestions(self):
        """AI 제안을 실제 라벨로 변환"""
        count = 0
        
        for x, y in self.ai_points.get('intersection', []):
            if not self.is_point_exists('intersection', x, y):
                self.add_point('intersection', x, y)
                count += 1
        
        for x, y in self.ai_points.get('curve', []):
            if not self.is_point_exists('curve', x, y):
                self.add_point('curve', x, y)
                count += 1
        
        for x, y in self.ai_points.get('endpoint', []):
            if not self.is_point_exists('endpoint', x, y):
                self.add_point('endpoint', x, y)
                count += 1
        
        self.ai_points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        self.deleted_points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        
        self.update_display()
        
        if hasattr(self.parent().parent(), 'statusBar'):
            self.parent().parent().statusBar().showMessage(
                f"{count}개의 AI 제안이 수락되었습니다.", 3000
            )
    
    def is_point_exists(self, category, x, y, threshold=5):
        """해당 위치에 이미 포인트가 있는지 확인"""
        for px, py in self.points.get(category, []):
            if np.sqrt((x - px)**2 + (y - py)**2) < threshold:
                return True
        return False
    
    def clear_all_points(self):
        """모든 포인트 제거"""
        self.points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        self.ai_points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        self.deleted_points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        self.update_display()

    def get_deleted_points(self):
        """삭제된 포인트 반환"""
        return self.deleted_points.copy()
    
    def clear_deleted_points(self):
        """삭제된 포인트 기록 초기화"""
        self.deleted_points = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }

class CanvasWidget(QWidget):
    """캔버스 위젯 - 도로망 분석 UI"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 속성 초기화
        self.current_file = None
        self.skeleton = None
        self.road_geometry = None
        self.processing_time = 0
        
        # UI 초기화
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 툴바
        toolbar_layout = QHBoxLayout()
        
        # 현재 파일 표시
        self.file_label = QLabel("파일: 없음")
        self.file_label.setStyleSheet("QLabel { padding: 5px; }")
        toolbar_layout.addWidget(self.file_label)
        
        toolbar_layout.addStretch()
        
        # 통계 정보
        self.stats_label = QLabel("통계: -")
        self.stats_label.setStyleSheet("QLabel { padding: 5px; }")
        toolbar_layout.addWidget(self.stats_label)
        
        layout.addLayout(toolbar_layout)
        
        # 캔버스
        self.canvas = InteractiveCanvas(self)
        self.canvas.point_added.connect(self.on_point_added)
        self.canvas.point_removed.connect(self.on_point_removed)
        layout.addWidget(self.canvas)
        
        # 레이어 컨트롤 패널
        self.layer_control = self.create_layer_control()
        self.layer_control.setParent(self.canvas)
        self.layer_control.move(10, 60)
        
        # 범례
        self.legend_label = QLabel()
        self.legend_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 200);
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)
        self.update_legend()
        
        # 범례를 캔버스 우측 상단에 배치
        self.legend_label.setParent(self.canvas)
        self.legend_label.move(10, 350)
        
        self.setLayout(layout)
    
    def create_layer_control(self):
        """레이어 컨트롤 패널 생성"""
        control = QGroupBox("레이어 컨트롤")
        control.setStyleSheet("""
            QGroupBox {
                background-color: rgba(255, 255, 255, 200);
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
            QGroupBox::title {
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # 도로망 레이어
        road_layout = QHBoxLayout()
        self.road_checkbox = QCheckBox("도로망")
        self.road_checkbox.setChecked(True)
        self.road_checkbox.toggled.connect(self.canvas.toggle_road_layer)
        road_layout.addWidget(self.road_checkbox)
        
        self.road_opacity_slider = QSlider(Qt.Horizontal)
        self.road_opacity_slider.setMinimum(0)
        self.road_opacity_slider.setMaximum(100)
        self.road_opacity_slider.setValue(30)
        self.road_opacity_slider.setMaximumWidth(100)
        self.road_opacity_slider.valueChanged.connect(self.canvas.set_road_opacity)
        road_layout.addWidget(self.road_opacity_slider)
        
        layout.addLayout(road_layout)
        
        # 스켈레톤 레이어
        skeleton_layout = QHBoxLayout()
        self.skeleton_checkbox = QCheckBox("중심선")
        self.skeleton_checkbox.setChecked(True)
        self.skeleton_checkbox.toggled.connect(self.canvas.toggle_skeleton_layer)
        skeleton_layout.addWidget(self.skeleton_checkbox)
        
        self.skeleton_opacity_slider = QSlider(Qt.Horizontal)
        self.skeleton_opacity_slider.setMinimum(0)
        self.skeleton_opacity_slider.setMaximum(100)
        self.skeleton_opacity_slider.setValue(100)
        self.skeleton_opacity_slider.setMaximumWidth(100)
        self.skeleton_opacity_slider.valueChanged.connect(self.canvas.set_skeleton_opacity)
        skeleton_layout.addWidget(self.skeleton_opacity_slider)
        
        layout.addLayout(skeleton_layout)
        
        # 포인트 레이어
        self.points_checkbox = QCheckBox("분석 포인트")
        self.points_checkbox.setChecked(True)
        self.points_checkbox.toggled.connect(self.canvas.toggle_points_layer)
        layout.addWidget(self.points_checkbox)
        
        control.setLayout(layout)
        control.setFixedSize(200, 120)
        
        return control
    
    def update_display(self):
        """디스플레이 업데이트"""
        # 파일명 업데이트
        if self.current_file:
            self.file_label.setText(f"파일: {Path(self.current_file).name}")
        
        # 도로망 geometry 설정
        if self.road_geometry is not None:
            self.canvas.set_road_geometry(self.road_geometry)
        
        # 스켈레톤 설정
        if self.skeleton is not None:
            self.canvas.set_skeleton(self.skeleton)
        
        # 통계 업데이트
        self.update_stats()
        
        # 캔버스 업데이트
        self.canvas.update_display()
    
    def set_road_data(self, gdf):
        """도로 데이터 설정"""
        if gdf is not None and not gdf.empty:
            self.road_geometry = gdf.geometry.tolist()
            # 도로 데이터 설정 후 즉시 표시
            if hasattr(self, 'canvas'):
                self.canvas.set_road_geometry(self.road_geometry)
                self.canvas.draw_road_layer()
                self.canvas.scene.update()  # 씬 업데이트 강제
    def update_stats(self):
        """통계 정보 업데이트"""
        if self.skeleton is None:
            self.stats_label.setText("통계: -")
            return
        
        total_intersections = len(self.canvas.points.get('intersection', []))
        total_curves = len(self.canvas.points.get('curve', []))
        total_endpoints = len(self.canvas.points.get('endpoint', []))
        
        stats_text = f"교차점: {total_intersections} | 커브: {total_curves} | 끝점: {total_endpoints}"
        
        if self.processing_time > 0:
            stats_text += f" | 처리시간: {self.processing_time:.2f}초"
        
        # 스켈레톤 포인트 수와 샘플링 정보 추가
        if hasattr(self.canvas, 'skeleton_sampling_rate') and self.canvas.skeleton_sampling_rate > 1:
            stats_text += f" | 샘플링: 1/{self.canvas.skeleton_sampling_rate}"
        
        self.stats_label.setText(stats_text)
    
    def update_legend(self):
        """범례 업데이트"""
        legend_text = "=== 범례 ===\n\n"
        legend_text += "사용자 라벨:\n"
        legend_text += "● 교차점 (빨간색)\n"
        legend_text += "■ 커브 (파란색)\n"
        legend_text += "★ 끝점 (녹색)\n"
        
        if self.canvas.show_ai_predictions:
            legend_text += "\nAI 예측:\n"
            legend_text += "◆ 교차점 (보라색)\n"
            legend_text += "▲ 커브 (주황색)\n"
            legend_text += "⬢ 끝점 (갈색)\n"
        
        legend_text += "\n=== 조작법 ===\n"
        legend_text += "좌클릭: 커브 추가\n"
        legend_text += "우클릭: 끝점 추가\n"
        legend_text += "Shift+클릭: 포인트 제거\n"
        legend_text += "D키: 가장 가까운 점 삭제\n"
        legend_text += "마우스휠: 줌\n"
        legend_text += "Space: 화면 맞춤\n"
        legend_text += "\n1,2,3키: 레이어 토글"
        
        if self.canvas.ai_assist_mode:
            legend_text += "\n\n=== AI 보조 ===\n"
            legend_text += "A: AI 예측 표시 토글\n"
            legend_text += "Ctrl+S: AI 제안 수락\n"
            legend_text += "Ctrl+R: AI 재예측"
        
        self.legend_label.setText(legend_text)
    
    def on_point_added(self, category, x, y):
        """포인트 추가 시 호출"""
        self.update_stats()
        logger.info(f"포인트 추가: {category} at ({x:.2f}, {y:.2f})")
    
    def on_point_removed(self, category, x, y):
        """포인트 제거 시 호출"""
        self.update_stats()
        logger.info(f"포인트 제거: {category} at ({x:.2f}, {y:.2f})")
    
    def clear_all(self):
        """모든 데이터 초기화"""
        self.current_file = None
        self.skeleton = None
        self.road_geometry = None
        self.processing_time = 0
        
        # 캔버스 완전 초기화
        self.canvas.clear_canvas()
        
        # 파일명과 통계 초기화
        self.file_label.setText("파일: 없음")
        self.update_stats()
    
    def set_ai_mode(self, enabled):
        """AI 보조 모드 설정"""
        self.canvas.ai_assist_mode = enabled
        self.update_legend()