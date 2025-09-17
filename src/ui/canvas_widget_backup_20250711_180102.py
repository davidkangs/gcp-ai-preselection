import numpy as np
from pathlib import Path
from ..core.user_action_tracker import UserActionTracker
import logging
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import requests
import math
from io import BytesIO

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsEllipseItem, QGraphicsRectItem,
    QGraphicsPolygonItem, QGraphicsPathItem, QMessageBox,
    QCheckBox, QPushButton, QSlider, QGroupBox
)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QPen, QBrush, QColor, QPainterPath, QPolygonF, QWheelEvent,
    QFont, QTransform, QPainter, QPixmap, QImage, QCursor
)

logger = logging.getLogger(__name__)

class InteractiveCanvas(QGraphicsView):
    point_added = pyqtSignal(str, float, float)
    point_removed = pyqtSignal(str, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.action_tracker = UserActionTracker()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.TextAntialiasing, False)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)

        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.zoom_factor = 1.15
        
        # Panning 관련 변수
        self.panning_mode = False
        self.last_mouse_pos = None
        self.space_pressed = False
        
        # 디버깅을 위한 마우스 추적 활성화
        self.setMouseTracking(True)

        self.show_satellite_layer = False
        self.show_road_layer = True
        self.show_skeleton_layer = True
        self.show_points_layer = True
        self.show_background_layer = True
        self.background_opacity = 0.2
        self.satellite_opacity = 0.8

        self.colors = {
            'road': QColor(200,200,200),
            'road_stroke': QColor(150,150,150),
            'skeleton': QColor(50,50,200),
            'background': QColor(230,230,230),
            'background_stroke': QColor(200,200,200),
            'intersection': QColor(255,0,0),
            'curve': QColor(0,0,255),
            'endpoint': QColor(0,255,0),
            'ai_intersection': QColor(128,0,128),
            'ai_curve': QColor(255,165,0),
            'ai_endpoint': QColor(139,69,19),
        }

        self.road_opacity = 0.3
        self.skeleton_opacity = 1.0

        self.road_geometry = None
        self.background_geometry = None
        self.skeleton_points = None
        self.points = {'intersection': [], 'curve': [], 'endpoint': []}
        self.deleted_points = {'intersection': [], 'curve': [], 'endpoint': []}
        self.ai_points = {'intersection': [], 'curve': [], 'endpoint': [], 'delete': []}

        # 멀티폴리곤 관리
        self.current_polygon_index = 1
        self.total_polygons = 1
        self.polygon_info = {}
        
        self.ai_assist_mode = False
        self.show_ai_predictions = True
        self.ai_confidence_threshold = 0.7

        self.road_items = []
        self.background_items = []
        self.skeleton_items = []
        self.point_items = []
        self.ai_point_items = []
        self.satellite_items = []
        self.distance_items = []  # 네트워크 연결선 및 거리 텍스트 저장

        self.max_skeleton_points = 5000
        self.original_crs = None
        
        self.transformer = None
        # 초기 transformer 설정 제거 - 도로/배경 데이터가 로드될 때 설정됨
        # self.setup_coordinate_transformer("EPSG:5186")
        
        self.satellite_tiles = {}

        # 거리 정보 저장
        self.distance_info = None
        
        # 네트워크 연결 정보 저장
        self.network_connections = []
        
        # Excel 점 저장
        self.excel_points = []
        self.excel_items = []
        
        # Excel 점 표시 여부
        self.show_excel_points = True
        
        # 점 크기 통일 (2mm = 약 8픽셀)
        self.point_size = 8

    def keyPressEvent(self, event):
        k = event.key()
        
        # 스페이스바 처리 - 팬닝 모드와 fitInView 충돌 해결
        if k == Qt.Key_Space and not event.isAutoRepeat():
            if not self.panning_mode:  # 팬닝 모드가 아닐 때만
                self.panning_mode = True
                self.space_pressed = True
                self.setCursor(Qt.OpenHandCursor)
                # 드래그 모드는 변경하지 않음 (마우스 이벤트 차단 방지)
            return
        
        # 방향키 처리 추가
        if k == Qt.Key_Left:
            self.move_view(-50, 0)  # 왼쪽으로 50픽셀 이동
            return
        elif k == Qt.Key_Right:
            self.move_view(50, 0)   # 오른쪽으로 50픽셀 이동
            return
        elif k == Qt.Key_Up:
            self.move_view(0, -50)  # 위로 50픽셀 이동
            return
        elif k == Qt.Key_Down:
            self.move_view(0, 50)   # 아래로 50픽셀 이동
            return
        
        if k == Qt.Key_D:
            cp = self.mapFromGlobal(self.cursor().pos())
            if self.rect().contains(cp):
                sp = self.mapToScene(cp)
                self.remove_nearest_point(sp.x(), -sp.y())
            self.setFocus()
            return

        if k == Qt.Key_A:
            self.show_ai_predictions = not self.show_ai_predictions
            self.update_display()
        elif k == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            self.accept_ai_suggestions()
        elif k == Qt.Key_R and event.modifiers() == Qt.ControlModifier:
            if hasattr(self.parent().parent(), 'run_ai_detection'):
                self.parent().parent().run_ai_detection()
        elif k == Qt.Key_1:
            self.toggle_road_layer(not self.show_road_layer)
        elif k == Qt.Key_2:
            self.toggle_skeleton_layer(not self.show_skeleton_layer)
        elif k == Qt.Key_3:
            self.toggle_points_layer(not self.show_points_layer)
        elif k == Qt.Key_Q:
            self.run_dqn_prediction()
        elif k == Qt.Key_T:
            self.toggle_dqn_prediction()

        super().keyPressEvent(event)

    def move_view(self, dx, dy):
        """뷰를 지정된 픽셀만큼 이동"""
        # 현재 중심점을 Scene 좌표로 가져오기
        current_center = self.mapToScene(self.viewport().rect().center())
        # 이동할 거리를 Scene 좌표로 변환
        offset = self.mapToScene(dx, dy) - self.mapToScene(0, 0)
        # 새로운 중심점으로 이동
        new_center = current_center + offset
        self.centerOn(new_center)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.panning_mode = False
            self.space_pressed = False
            self.setCursor(Qt.ArrowCursor)
            # 원래 드래그 모드로 복원하지 않음 (이미 RubberBandDrag 상태)
            self.last_mouse_pos = None
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        if self.panning_mode:
            if event.button() == Qt.LeftButton:
                self.last_mouse_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
            return
        
        if not self.skeleton_points:
            super().mousePressEvent(event)
            return

        p = self.mapToScene(event.pos())
        x, y = p.x(), -p.y()

        arr = np.array(self.skeleton_points) if isinstance(self.skeleton_points, (list, np.ndarray)) else np.array(self.skeleton_points)
        if len(arr) > 0:
            dists = np.hypot(arr[:,0]-x, arr[:,1]-y)
            mi = np.argmin(dists)
            if dists[mi] < 30:
                x, y = arr[mi]

        if event.modifiers() & Qt.ShiftModifier:
            self.remove_nearest_point(x, y)
        else:
            if event.button() == Qt.LeftButton:
                self.add_point('curve', x, y)
            elif event.button() == Qt.RightButton:
                self.add_point('endpoint', x, y)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning_mode and self.last_mouse_pos:
            # Scene 좌표계로 변환하여 처리
            old_scene_pos = self.mapToScene(self.last_mouse_pos)
            new_scene_pos = self.mapToScene(event.pos())
            delta_scene = new_scene_pos - old_scene_pos
            
            # 현재 중심점을 이동
            current_center = self.mapToScene(self.viewport().rect().center())
            new_center = current_center - delta_scene
            self.centerOn(new_center)
            
            self.last_mouse_pos = event.pos()
            return  # 팬닝 중에는 다른 이벤트 처리 안함
        
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.panning_mode and event.button() == Qt.LeftButton:
            self.setCursor(Qt.OpenHandCursor)
            self.last_mouse_pos = None
        
        super().mouseReleaseEvent(event)
        
        if not self.panning_mode and self.dragMode() == QGraphicsView.RubberBandDrag and self.show_satellite_layer:
            self.load_satellite_tiles()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(self.zoom_factor, self.zoom_factor)
        else:
            self.scale(1/self.zoom_factor, 1/self.zoom_factor)
        
        if self.show_satellite_layer:
            self.load_satellite_tiles()

    def setup_coordinate_transformer(self, source_crs='EPSG:5186'):
        try:
            import pyproj
            self.transformer = pyproj.Transformer.from_crs(
                source_crs, 'EPSG:4326', always_xy=True
            )
        except Exception as e:
            self.transformer = None

    def toggle_satellite_layer(self, visible):
        self.show_satellite_layer = visible
        if visible:
            self.load_satellite_tiles()
        else:
            for item in self.satellite_items:
                self.scene.removeItem(item)
            self.satellite_items.clear()
        self.viewport().update()

    def set_satellite_opacity(self, opacity):
        self.satellite_opacity = opacity / 100.0
        for item in self.satellite_items:
            item.setOpacity(self.satellite_opacity)

    def load_satellite_tiles(self):
        try:
            if not self.scene.items():
                return
            
            # transformer가 없으면 기본값으로 설정
            if not self.transformer:
                self.setup_coordinate_transformer('EPSG:5186')
                
            for item in self.satellite_items:
                self.scene.removeItem(item)
            self.satellite_items.clear()
            
            viewport_rect = self.viewport().rect()
            buffer = 0.5  # 각 방향으로 50% 추가 (총 2배)
            expanded_rect = viewport_rect.adjusted(
                -int(viewport_rect.width() * buffer),
                -int(viewport_rect.height() * buffer),
                int(viewport_rect.width() * buffer),
                int(viewport_rect.height() * buffer)
            )
            view_rect = self.mapToScene(expanded_rect).boundingRect()
            
            tm_min_x = view_rect.left()
            tm_min_y = -view_rect.bottom()
            tm_max_x = view_rect.right()
            tm_max_y = -view_rect.top()
            
            if not self.transformer:
                return
            
            corners_tm = [
                (tm_min_x, tm_min_y),
                (tm_max_x, tm_min_y),
                (tm_max_x, tm_max_y),
                (tm_min_x, tm_max_y),
            ]
            
            lons, lats = [], []
            for x, y in corners_tm:
                lon, lat = self.transformer.transform(x, y)
                lons.append(lon)
                lats.append(lat)
            
            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)
            
            center_lon = (min_lon + max_lon) / 2
            center_lat = (min_lat + max_lat) / 2
            
            lat_span = max_lat - min_lat
            lon_span = max_lon - min_lon
            
            viewport_width = self.viewport().width()
            viewport_height = self.viewport().height()
            
            zoom_lat = math.log2(180 / lat_span) + math.log2(viewport_height / 256)
            zoom_lon = math.log2(360 / lon_span) + math.log2(viewport_width / 256)
            zoom = min(zoom_lat, zoom_lon)
            zoom = max(1, min(19, int(zoom)))
            
            token = "pk.eyJ1Ijoia2FuZ2RhZXJpIiwiYSI6ImNtY2FtbTQyODA1Y2Iybm9ybmlhbTZrbDUifQ.dwjb3fq0FqvXDx6-OuLYHw"
            
            img_width = min(1280, viewport_width * 2)
            img_height = min(1280, viewport_height * 2)
            
            bbox = f"[{min_lon},{min_lat},{max_lon},{max_lat}]"
            url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/static/{bbox}/{img_width}x{img_height}@2x?access_token={token}"
            
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                img = QImage()
                if img.loadFromData(resp.content):
                    pixmap = QPixmap.fromImage(img)
                    pixmap_item = self.scene.addPixmap(pixmap)
                    
                    img_actual_width = pixmap.width()
                    img_actual_height = pixmap.height()
                    
                    tm_width = tm_max_x - tm_min_x
                    tm_height = tm_max_y - tm_min_y
                    
                    meters_per_lon = 111320 * math.cos(math.radians(center_lat))
                    meters_per_lat = 110540
                    wgs84_width_m = (max_lon - min_lon) * meters_per_lon
                    wgs84_height_m = (max_lat - min_lat) * meters_per_lat
                    
                    real_aspect_ratio = wgs84_width_m / wgs84_height_m
                    img_aspect_ratio = img_actual_width / img_actual_height
                    
                    if real_aspect_ratio > img_aspect_ratio:
                        effective_width = img_actual_width
                        effective_height = img_actual_width / real_aspect_ratio
                        x_padding = 0
                        y_padding = (img_actual_height - effective_height) / 2
                    else:
                        effective_height = img_actual_height
                        effective_width = img_actual_height * real_aspect_ratio
                        y_padding = 0
                        x_padding = (img_actual_width - effective_width) / 2
                    
                    scale_x = tm_width / effective_width
                    scale_y = tm_height / effective_height
                    
                    transform = QTransform()
                    
                    OFFSET_X = 12.5
                    OFFSET_Y = 0
                    
                    transform.translate(tm_min_x - OFFSET_X, -tm_max_y + OFFSET_Y)
                    transform.translate(-x_padding * scale_x, -y_padding * scale_y)
                    transform.scale(scale_x, scale_y)
                    
                    pixmap_item.setTransform(transform)
                    pixmap_item.setOpacity(self.satellite_opacity)
                    pixmap_item.setZValue(-10)
                    
                    self.satellite_items.append(pixmap_item)
                    
        except Exception as e:
            import traceback
            traceback.print_exc()

    def set_background_geometry(self, geometry, crs=None):
        self.background_geometry = geometry
        if crs:
            crs_code = None
            if hasattr(crs, 'to_epsg'):
                crs_code = crs.to_epsg()
            elif hasattr(crs, 'to_authority'):
                auth = crs.to_authority()
                if auth and auth[0] == 'EPSG':
                    crs_code = int(auth[1])
        self.draw_background_layer()
        
    def draw_background_layer(self):
        self.setUpdatesEnabled(False)
        try:
            # 안전하게 아이템 제거
            for item in self.background_items[:]:  # 리스트 복사본 사용
                try:
                    if item.scene() == self.scene:  # 아이템이 여전히 scene에 있는지 확인
                        self.scene.removeItem(item)
                except RuntimeError:
                    # 이미 삭제된 경우 무시
                    pass
            self.background_items.clear()
            
            if not self.background_geometry or not self.show_background_layer:
                return
            
            if hasattr(self.background_geometry, '__iter__'):
                for geom in self.background_geometry:
                    self._draw_background_geometry(geom)
            else:
                self._draw_background_geometry(self.background_geometry)
        except Exception as e:
            print(f"배경 레이어 그리기 오류: {e}")
        finally:
            self.setUpdatesEnabled(True)
            
    def _draw_background_geometry(self, geom):
        if geom.geom_type == 'Polygon':
            path = QPainterPath()
            exterior = list(geom.exterior.coords)
            path.moveTo(exterior[0][0], -exterior[0][1])
            for x, y in exterior[1:]:
                path.lineTo(x, -y)
            path.closeSubpath()
            
            item = QGraphicsPathItem(path)
            item.setPen(QPen(self.colors['background_stroke'], 1))
            item.setBrush(QBrush(self.colors['background']))
            item.setOpacity(self.background_opacity)
            item.setZValue(-5)
            self.scene.addItem(item)
            self.background_items.append(item)
        
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                self._draw_background_geometry(poly)

    def toggle_background_layer(self, visible):
        self.show_background_layer = visible
        self.draw_background_layer()

    def set_background_opacity(self, opacity):
        self.background_opacity = opacity / 100.0
        for item in self.background_items:
            item.setOpacity(self.background_opacity)

    def clear_canvas(self):
        self.scene.clear()
        self.road_geometry = None
        self.background_geometry = None
        self.skeleton_points = None
        self.points = {'intersection': [], 'curve': [], 'endpoint': []}
        self.ai_points = {'intersection': [], 'curve': [], 'endpoint': [], 'delete': []}
        self.deleted_points = {'intersection': [], 'curve': [], 'endpoint': []}
        self.road_items.clear()
        self.background_items.clear()
        self.skeleton_items.clear()
        self.point_items.clear()
        self.ai_point_items.clear()
        self.satellite_items.clear()
        # distance_items도 안전하게 정리
        if hasattr(self, 'distance_items'):
            self.distance_items.clear()
        self.resetTransform()

    def set_road_geometry(self, geometry, crs=None):
        self.road_geometry = geometry
        if crs:
            self.original_crs = crs
            crs_code = None
            if hasattr(crs, 'to_epsg'):
                crs_code = crs.to_epsg()
            elif hasattr(crs, 'to_authority'):
                auth = crs.to_authority()
                if auth and auth[0] == 'EPSG':
                    crs_code = int(auth[1])
            
            if crs_code:
                try:
                    import pyproj
                    source_crs = f'EPSG:{crs_code}'
                    self.transformer = pyproj.Transformer.from_crs(
                        source_crs,
                        'EPSG:4326',
                        always_xy=True
                    )
                except Exception as e:
                    pass
        self.draw_road_layer()

    def draw_road_layer(self):
        self.setUpdatesEnabled(False)
        try:
            # 안전하게 아이템 제거
            for item in self.road_items[:]:  # 리스트 복사본 사용
                try:
                    if item.scene() == self.scene:  # 아이템이 여전히 scene에 있는지 확인
                        self.scene.removeItem(item)
                except RuntimeError:
                    # 이미 삭제된 경우 무시
                    pass
            self.road_items.clear()
            
            if not self.road_geometry or not self.show_road_layer:
                return
            if hasattr(self.road_geometry, '__iter__'):
                for geom in self.road_geometry:
                    self._draw_single_geometry(geom)
            else:
                self._draw_single_geometry(self.road_geometry)
        finally:
            self.setUpdatesEnabled(True)

    def _draw_single_geometry(self, geom):
        if geom.geom_type == 'LineString':
            path = QPainterPath()
            coords = list(geom.coords)
            path.moveTo(coords[0][0], -coords[0][1])
            for x, y in coords[1:]:
                path.lineTo(x, -y)
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
            path.moveTo(exterior[0][0], -exterior[0][1])
            for x, y in exterior[1:]:
                path.lineTo(x, -y)
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
        if skeleton_points is None:
            logger.warning("스켈레톤 포인트가 None입니다")
            self.skeleton_points = None
            return
            
        if not isinstance(skeleton_points, (list, np.ndarray)):
            logger.error(f"잘못된 스켈레톤 데이터 타입: {type(skeleton_points)}")
            return
            
        self.skeleton_points = skeleton_points
        if len(skeleton_points) > 0:
            if len(skeleton_points) > self.max_skeleton_points:
                self.skeleton_sampling_rate = (
                    len(skeleton_points) // self.max_skeleton_points
                )
                logger.info(f"스켈레톤 포인트 샘플링: {len(skeleton_points)} -> {len(skeleton_points)//self.skeleton_sampling_rate}")
            else:
                self.skeleton_sampling_rate = 1
        else:
            logger.warning("빈 스켈레톤 포인트 배열입니다")
        self.draw_skeleton_layer()
        if self.skeleton_items:
            min_x = float('inf')
            max_x = float('-inf')
            min_y = float('inf')
            max_y = float('-inf')
            
            for x, y in skeleton_points:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, -y)
                max_y = max(max_y, -y)
            
            rect = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)
            self.scene.setSceneRect(rect.adjusted(-50, -50, 50, 50))
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.scale(0.5, 0.5)  # 50% 축소하여 더 넓은 영역 보이게

    def draw_skeleton_layer(self):
        self.setUpdatesEnabled(False)
        try:
            # 안전하게 아이템 제거
            for item in self.skeleton_items[:]:  # 리스트 복사본 사용
                try:
                    if item.scene() == self.scene:  # 아이템이 여전히 scene에 있는지 확인
                        self.scene.removeItem(item)
                except RuntimeError:
                    # 이미 삭제된 경우 무시
                    pass
            self.skeleton_items.clear()
            
            if (
                self.skeleton_points is None or
                len(self.skeleton_points) == 0 or
                not self.show_skeleton_layer
            ):
                return
            pts = self.skeleton_points[::self.skeleton_sampling_rate]
            if len(pts) < 1000:
                self._draw_skeleton_as_lines(pts)
            else:
                self._draw_skeleton_as_dots(pts)
        finally:
            self.setUpdatesEnabled(True)

    def _draw_skeleton_as_lines(self, points):
        path = QPainterPath()
        rem = list(range(len(points)))
        while rem:
            idx = rem.pop(0)
            path.moveTo(points[idx][0], -points[idx][1])
            while True:
                cur = points[idx]
                nxt, dist = None, 50
                for j in rem:
                    d = np.linalg.norm(np.array([cur[0], -cur[1]]) - np.array([points[j][0], -points[j][1]]))
                    if d < dist:
                        dist, nxt = d, j
                if nxt is None:
                    break
                path.lineTo(points[nxt][0], -points[nxt][1])
                rem.remove(nxt)
                idx = nxt
        item = QGraphicsPathItem(path)
        pen = QPen(self.colors['skeleton'], 2)
        item.setPen(pen)
        item.setOpacity(self.skeleton_opacity)
        item.setZValue(0)
        self.scene.addItem(item)
        self.skeleton_items.append(item)

    def _draw_skeleton_as_dots(self, points):
        for i in range(0, len(points), 100):
            path = QPainterPath()
            for x, y in points[i:i+100]:
                path.addEllipse(x-1, -y-1, 2, 2)
            item = QGraphicsPathItem(path)
            item.setPen(QPen(Qt.NoPen))
            item.setBrush(QBrush(self.colors['skeleton']))
            item.setOpacity(self.skeleton_opacity)
            item.setZValue(0)
            self.scene.addItem(item)
            self.skeleton_items.append(item)

    def toggle_road_layer(self, visible):
        self.show_road_layer = visible
        self.update_display()

    def toggle_skeleton_layer(self, visible):
        self.show_skeleton_layer = visible
        self.update_display()

    def toggle_points_layer(self, visible):
        self.show_points_layer = visible
        self.update_display()

    def set_road_opacity(self, opacity):
        self.road_opacity = opacity / 100.0
        for item in self.road_items:
            item.setOpacity(self.road_opacity)

    def set_skeleton_opacity(self, opacity):
        self.skeleton_opacity = opacity / 100.0
        for item in self.skeleton_items:
            item.setOpacity(self.skeleton_opacity)

    def update_display(self):
        self.setUpdatesEnabled(False)
        try:
            self.draw_background_layer()
            self.draw_road_layer()
            self.draw_skeleton_layer()
            self.draw_points_layer()
            if self.show_ai_predictions and self.ai_points:
                self.draw_ai_predictions()
            self.draw_distance_lines()  # 거리 표시 추가
        finally:
            self.setUpdatesEnabled(True)
            self.scene.update()

    def draw_points_layer(self):
        for item in self.point_items:
            self.scene.removeItem(item)
        self.point_items.clear()
        
        # Excel 점 아이템도 제거
        if hasattr(self, 'excel_items'):
            for item in self.excel_items:
                self.scene.removeItem(item)
            self.excel_items.clear()
        
        if not self.show_points_layer:
            return

        # 모든 점을 동일한 파란색 원으로 통일 (2mm = 8픽셀)
        unified_color = QColor(100, 100, 255)  # 파란색 통일
        point_size = self.point_size if hasattr(self, 'point_size') else 8
        half_size = point_size / 2
        
        # 모든 카테고리의 점을 동일하게 그리기
        for category in ['intersection', 'curve', 'endpoint']:
            for point in self.points.get(category, []):
                if len(point) >= 2:
                    x, y = float(point[0]), float(point[1])
                else:
                    x, y = float(point), 0.0
                    
                # 파란색 원으로 통일
                itm = self.scene.addEllipse(
                    x - half_size, -y - half_size, point_size, point_size,
                    QPen(unified_color, 2),
                    QBrush(unified_color)
                )
                itm.setZValue(10)
                self.point_items.append(itm)
        
        # Excel 점들을 다이아몬드로 표시 (표시 설정이 켜져 있을 때만)
        if hasattr(self, 'excel_points') and self.excel_points and self.show_excel_points:
            excel_color = QColor(255, 100, 100)  # 빨간색
            diamond_size = point_size * 1.2  # 다이아몬드는 약간 크게
            
            for point in self.excel_points:
                if len(point) >= 2:
                    x, y = float(point[0]), float(point[1])
                    
                    # 다이아몬드 모양 그리기
                    path = QPainterPath()
                    path.moveTo(x, -y - diamond_size)  # 상단
                    path.lineTo(x + diamond_size, -y)  # 오른쪽
                    path.lineTo(x, -y + diamond_size)  # 하단
                    path.lineTo(x - diamond_size, -y)  # 왼쪽
                    path.closeSubpath()
                    
                    item = QGraphicsPathItem(path)
                    item.setPen(QPen(excel_color, 2))
                    item.setBrush(QBrush(excel_color))
                    item.setZValue(11)  # AI 점보다 위에 표시
                    self.scene.addItem(item)
                    self.excel_items.append(item)

    def add_point(self, category, x, y):
        if category not in self.points:
            return
        if not self.is_point_exists(category, x, y):
            self.points[category].append((x, y))
            self.action_tracker.record_action('add', category, x, y)
            self.point_added.emit(category, x, y)
            self.draw_points_layer()

    def remove_nearest_point(self, x, y):
        md, cat, idx = float('inf'), None, None
        for c, pts in self.points.items():
            for i_, point in enumerate(pts):
                # point가 tuple/list인 경우 첫 2개 값만 사용
                if len(point) >= 2:
                    px, py = float(point[0]), float(point[1])
                else:
                    px, py = float(point), 0.0  # 안전장치
                d = np.hypot(px - x, py - y)
                if d < md:
                    md, cat, idx = d, c, i_
        if cat and idx is not None and md < 30:
            removed = self.points[cat].pop(idx)
            self.deleted_points[cat].append(removed)
            # removed가 tuple/list인지 확인
            if len(removed) >= 2:
                removed_x, removed_y = float(removed[0]), float(removed[1])
            else:
                removed_x, removed_y = float(removed), 0.0
            self.action_tracker.record_action('remove', cat, removed_x, removed_y)
            self.point_removed.emit(cat, removed_x, removed_y)
            self.draw_points_layer()
            return True
        return False

    def draw_ai_predictions(self):
        for item in self.ai_point_items:
            self.scene.removeItem(item)
        self.ai_point_items.clear()

        # AI 예측도 모든 점을 동일한 주황색 원으로 통일
        ai_unified_color = QColor(255, 140, 0, 150)  # 주황색 반투명
        
        # AI 교차점 - 주황색 원
        for point in self.ai_points['intersection']:
            if len(point) >= 2:
                x, y = float(point[0]), float(point[1])
            else:
                x, y = float(point), 0.0
            itm = self.scene.addEllipse(
                x-6, -y-6, 12, 12,
                QPen(ai_unified_color, 2),
                QBrush(ai_unified_color)
            )
            itm.setZValue(8)
            self.ai_point_items.append(itm)

        # AI 커브 - 주황색 원 (통일)
        for point in self.ai_points['curve']:
            if len(point) >= 2:
                x, y = float(point[0]), float(point[1])
            else:
                x, y = float(point), 0.0
            itm = self.scene.addEllipse(
                x-6, -y-6, 12, 12,
                QPen(ai_unified_color, 2),
                QBrush(ai_unified_color)
            )
            itm.setZValue(8)
            self.ai_point_items.append(itm)

        # AI 끝점 - 주황색 원 (통일)
        for point in self.ai_points['endpoint']:
            if len(point) >= 2:
                x, y = float(point[0]), float(point[1])
            else:
                x, y = float(point), 0.0
            itm = self.scene.addEllipse(
                x-6, -y-6, 12, 12,
                QPen(ai_unified_color, 2),
                QBrush(ai_unified_color)
            )
            itm.setZValue(8)
            self.ai_point_items.append(itm)

        # AI 삭제 - 빨간색 X 표시 (구분을 위해 유지)
        for point in self.ai_points.get('delete', []):
            if len(point) >= 2:
                x, y = float(point[0]), float(point[1])
            else:
                x, y = float(point), 0.0
            path = QPainterPath()
            size = 8
            path.moveTo(x - size, -y - size)
            path.lineTo(x + size, -y + size)
            path.moveTo(x - size, -y + size)
            path.lineTo(x + size, -y - size)
            
            itm = QGraphicsPathItem(path)
            itm.setPen(QPen(QColor(255, 0, 0), 3))
            itm.setZValue(9)
            self.ai_point_items.append(itm)

    def draw_distance_lines(self):
        """네트워크 연결선 및 거리 정보 표시 (검은색 점선)"""
        if not hasattr(self, 'network_connections') or not self.network_connections:
            return
            
        try:
            # 기존 거리 선 안전하게 제거
            if hasattr(self, 'distance_items'):
                for item in self.distance_items:
                    try:
                        # Qt 객체가 아직 유효한지 확인
                        if item is not None and hasattr(item, 'scene') and item.scene() is not None:
                            self.scene.removeItem(item)
                    except (RuntimeError, AttributeError):
                        # "wrapped C/C++ object has been deleted" 오류 방지
                        pass
            
            self.distance_items = []
            
            logger.info(f"거리 선 그리기 시작: {len(self.network_connections)}개 연결")
            
            # 🌐 네트워크 연결선 그리기 (검은색 점선)
            for connection in self.network_connections:
                p1 = connection['point1']
                p2 = connection['point2']
                distance = connection['distance']
                
                # 검은색 점선으로 거리 연결선 그리기 (원래 스타일)
                path = QPainterPath()
                path.moveTo(p1[0], -p1[1])
                path.lineTo(p2[0], -p2[1])
                
                line_item = QGraphicsPathItem(path)
                pen = QPen(QColor(0, 0, 0), 1)  # 검은색
                pen.setStyle(Qt.DashLine)  # 점선
                line_item.setPen(pen)
                line_item.setZValue(5)
                self.scene.addItem(line_item)
                self.distance_items.append(line_item)
                
                # 거리 텍스트 추가 (중앙에 m 단위)
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = -(p1[1] + p2[1]) / 2
                
                text_item = self.scene.addText(f"{distance:.1f}m", QFont("Arial", 8))
                text_item.setPos(mid_x, mid_y)
                text_item.setZValue(6)
                # 텍스트 배경 (가독성을 위해)
                text_item.setDefaultTextColor(QColor(0, 0, 0))
                self.distance_items.append(text_item)
                
        except Exception as e:
            logger.error(f"네트워크 연결선 그리기 오류: {e}")

    def find_nearest_ai_suggestion(self, x, y, threshold=30):
        md, nearest = float('inf'), None
        for c in ['intersection','curve','endpoint']:
            for px, py in self.ai_points[c]:
                d = np.hypot(px-x, py-y)
                if d < md and d < threshold:
                    md, nearest = d, (px,py)
        return nearest

    def accept_ai_suggestions(self):
        cnt = 0
    
        for x, y in self.ai_points.get('delete', []):
            if self.remove_nearest_point(x, y):
                cnt += 1
    
        for c in ['intersection','curve','endpoint']:
            for x, y in self.ai_points[c]:
                if not self.is_point_exists(c, x, y):
                    self.add_point(c, x, y)
                    cnt += 1
        self.ai_points = {'intersection': [], 'curve': [], 'endpoint': [], 'delete': []}
        self.deleted_points = {'intersection': [], 'curve': [], 'endpoint': []}
        self.update_display()
        if hasattr(self.parent().parent(), 'statusBar'):
            self.parent().parent().statusBar().showMessage(
                f"{cnt}개의 AI 제안이 수락되었습니다.", 3000
            )

    def is_point_exists(self, category, x, y, threshold=5):
        for point in self.points[category]:
            # point가 tuple/list인 경우 첫 2개 값만 사용
            if len(point) >= 2:
                px, py = float(point[0]), float(point[1])
            else:
                px, py = float(point), 0.0  # 안전장치
            if np.hypot(px-x, py-y) < threshold:
                return True
        return False

    def clear_all_points(self):
        self.points = {'intersection': [], 'curve': [], 'endpoint': []}
        self.ai_points = {'intersection': [], 'curve': [], 'endpoint': [], 'delete': []}
        self.deleted_points = {'intersection': [], 'curve': [], 'endpoint': []}
        self.update_display()

    def get_deleted_points(self):
        return self.deleted_points.copy()

    def clear_deleted_points(self):
        self.deleted_points = {'intersection': [], 'curve': [], 'endpoint': []}

    def handle_deletion_prediction(self, predictions):
        if 'deletion' in predictions:
            for x, y in predictions['deletion']:
                if self.remove_nearest_point(x, y):
                    pass

    def run_dqn_prediction(self, confidence_threshold=0.7):
        if not self.skeleton_points:
            logger.warning("스켈레톤 포인트가 없어서 DQN 예측을 실행할 수 없습니다")
            return
        
        try:
            from ..learning.session_predictor import get_predictor
            
            predictor = get_predictor()
            if not predictor.is_loaded:
                logger.warning("DQN 모델이 로드되지 않았습니다")
                return
            
            predictions = predictor.predict_points(self.skeleton_points, confidence_threshold)
            self.ai_points = predictions
            self.draw_ai_predictions()
            logger.info(f"DQN 예측 완료: {len(predictions.get('intersection', []))} 교차점, {len(predictions.get('curve', []))} 커브, {len(predictions.get('endpoint', []))} 끝점")
            
        except ImportError as e:
            logger.error(f"DQN 모듈을 찾을 수 없습니다: {e}")
        except AttributeError as e:
            logger.error(f"DQN 모델 속성 오류: {e}")
        except Exception as e:
            logger.error(f"DQN 예측 중 예상치 못한 오류: {e}")

    def toggle_dqn_prediction(self):
        if hasattr(self, 'show_ai_predictions'):
            self.show_ai_predictions = not self.show_ai_predictions
            if self.show_ai_predictions:
                self.run_dqn_prediction()
            else:
                self.ai_points = {'intersection': [], 'curve': [], 'endpoint': [], 'delete': []}
                self.draw_ai_predictions()

    def get_dqn_model_info(self):
        try:
            from ..learning.session_predictor import get_predictor
            predictor = get_predictor()
            info = predictor.get_model_info()
            return info
        except ImportError as e:
            logger.error(f"DQN 모듈을 찾을 수 없습니다: {e}")
            return {}
        except Exception as e:
            logger.error(f"DQN 모델 정보 가져오기 실패: {e}")
            return {}

    def set_dqn_epsilon(self, epsilon):
        try:
            from ..learning.session_predictor import get_predictor
            predictor = get_predictor()
            predictor.set_epsilon(epsilon)
            logger.info(f"DQN epsilon 값 설정: {epsilon}")
        except ImportError as e:
            logger.error(f"DQN 모듈을 찾을 수 없습니다: {e}")
        except Exception as e:
            logger.error(f"DQN epsilon 설정 실패: {e}")


class CanvasWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_file = None
        self.skeleton = None
        self.road_geometry = None
        self.background_geometry = None
        self.processing_time = 0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar_layout = QHBoxLayout()
        self.file_label = QLabel("파일: 없음")
        self.file_label.setStyleSheet("padding:5px;")
        toolbar_layout.addWidget(self.file_label)
        toolbar_layout.addStretch()
        self.stats_label = QLabel("통계: -")
        self.stats_label.setStyleSheet("padding:5px;")
        toolbar_layout.addWidget(self.stats_label)
        layout.addLayout(toolbar_layout)

        self.canvas = InteractiveCanvas(self)
        self.canvas.point_added.connect(self.on_point_added)
        self.canvas.point_removed.connect(self.on_point_removed)
        layout.addWidget(self.canvas)

        self.layer_control = self.create_layer_control()
        self.layer_control.setParent(self.canvas)
        self.layer_control.move(10, 60)

        # 메뉴얼 삭제 바 추가
        self.manual_deletion_control = self.create_manual_deletion_bar()
        self.manual_deletion_control.setParent(self.canvas)
        self.manual_deletion_control.move(10, 250)  # 레이어 컨트롤 아래

        self.legend_label = QLabel()
        self.legend_label.setStyleSheet("""
            background-color: rgba(255,255,255,200);
            padding:10px; border:1px solid #ccc;
            border-radius:5px;
        """)
        self.update_legend()
        self.legend_label.setParent(self.canvas)
        self.legend_label.move(10, 420)  # 메뉴얼 바 아래로 이동

        self.setLayout(layout)

    def create_layer_control(self):
        control = QGroupBox("레이어 컨트롤")
        control.setStyleSheet("""
            QGroupBox {background-color: rgba(255,255,255,200);
                        border: 1px solid #ccc; border-radius:5px;
                        padding:5px; font-weight:bold;}
            QGroupBox::title {padding:0 5px;}
        """)
        v = QVBoxLayout()

        h_sat = QHBoxLayout()
        self.sat_cb = QCheckBox("위성영상")
        self.sat_cb.toggled.connect(self.canvas.toggle_satellite_layer)
        self.sat_cb.setChecked(False)
        h_sat.addWidget(self.sat_cb)
        self.sat_op = QSlider(Qt.Horizontal)
        self.sat_op.setRange(0,100)
        self.sat_op.setValue(80)
        self.sat_op.setMaximumWidth(100)
        self.sat_op.valueChanged.connect(self.canvas.set_satellite_opacity)
        h_sat.addWidget(self.sat_op)
        v.addLayout(h_sat)

        # 배경 레이어 컨트롤 추가
        h_bg = QHBoxLayout()
        self.bg_cb = QCheckBox("지구계 경계")
        self.bg_cb.setChecked(True)
        self.bg_cb.toggled.connect(self.canvas.toggle_background_layer)
        h_bg.addWidget(self.bg_cb)
        self.bg_op = QSlider(Qt.Horizontal)
        self.bg_op.setRange(0, 100)
        self.bg_op.setValue(20)  # 기본 20% 투명도
        self.bg_op.setMaximumWidth(100)
        self.bg_op.valueChanged.connect(self.canvas.set_background_opacity)
        h_bg.addWidget(self.bg_op)
        v.addLayout(h_bg)

        h_rd = QHBoxLayout()
        self.rd_cb = QCheckBox("도로망")
        self.rd_cb.setChecked(True)
        self.rd_cb.toggled.connect(self.canvas.toggle_road_layer)
        h_rd.addWidget(self.rd_cb)
        self.rd_op = QSlider(Qt.Horizontal)
        self.rd_op.setRange(0,100)
        self.rd_op.setValue(30)
        self.rd_op.setMaximumWidth(100)
        self.rd_op.valueChanged.connect(self.canvas.set_road_opacity)
        h_rd.addWidget(self.rd_op)
        v.addLayout(h_rd)

        h_sk = QHBoxLayout()
        self.sk_cb = QCheckBox("중심선")
        self.sk_cb.setChecked(True)
        self.sk_cb.toggled.connect(self.canvas.toggle_skeleton_layer)
        h_sk.addWidget(self.sk_cb)
        self.sk_op = QSlider(Qt.Horizontal)
        self.sk_op.setRange(0,100)
        self.sk_op.setValue(100)
        self.sk_op.setMaximumWidth(100)
        self.sk_op.valueChanged.connect(self.canvas.set_skeleton_opacity)
        h_sk.addWidget(self.sk_op)
        v.addLayout(h_sk)

        self.pt_cb = QCheckBox("분석 포인트")
        self.pt_cb.setChecked(True)
        self.pt_cb.toggled.connect(self.canvas.toggle_points_layer)
        v.addWidget(self.pt_cb)

        control.setLayout(v)
        control.setFixedSize(200,180)  # 높이 조정
        return control

    def create_manual_deletion_bar(self):
        """메뉴얼 삭제 바 생성"""
        control = QGroupBox("수동 삭제 도구")
        control.setStyleSheet("""
            QGroupBox {background-color: rgba(255,255,255,200);
                        border: 1px solid #ccc; border-radius:5px;
                        padding:5px; font-weight:bold;}
            QGroupBox::title {padding:0 5px;}
        """)
        v = QVBoxLayout()
        
        # AI 삭제 강도 슬라이더
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("AI 삭제 정도:"))
        
        self.deletion_confidence = QSlider(Qt.Horizontal)
        self.deletion_confidence.setRange(1, 10)
        self.deletion_confidence.setValue(5)
        self.deletion_confidence.setMaximumWidth(100)
        confidence_layout.addWidget(self.deletion_confidence)
        
        self.confidence_label = QLabel("5")
        self.deletion_confidence.valueChanged.connect(self.update_deletion_label)
        confidence_layout.addWidget(self.confidence_label)
        v.addLayout(confidence_layout)
        
        # 추가 삭제 실행 버튼
        self.manual_delete_btn = QPushButton("추가 삭제 실행")
        self.manual_delete_btn.clicked.connect(self.perform_manual_deletion)
        v.addWidget(self.manual_delete_btn)
        
        # 거리 정보 표시
        distance_layout = QVBoxLayout()
        distance_layout.addWidget(QLabel("거리 정보:"))
        self.distance_stats_label = QLabel("계산 중...")
        self.distance_stats_label.setStyleSheet("font-size: 10px;")
        distance_layout.addWidget(self.distance_stats_label)
        v.addLayout(distance_layout)
        
        control.setLayout(v)
        control.setFixedSize(200, 150)
        return control
    
    def update_deletion_label(self, value):
        """AI 삭제 정도 라벨 업데이트"""
        self.confidence_label.setText(str(value))

    def perform_manual_deletion(self):
        """스켈레톤 버퍼 확장 기반 삭제 실행"""
        try:
            buffer_level = self.deletion_confidence.value()
            
            # 슬라이더 값에 따른 버퍼 픽셀 매핑
            buffer_map = {
                1: 50, 2: 60, 3: 70, 4: 80, 5: 85,
                6: 90, 7: 95, 8: 100, 9: 110, 10: 120
            }
            buffer_pixels = buffer_map.get(buffer_level, 85)
            
            # 🔥 스켈레톤 기반 버퍼 영역 생성
            if not hasattr(self, 'skeleton') or not self.skeleton:
                logger.warning("스켈레톤 데이터가 없어 삭제를 수행할 수 없습니다")
                return
            
            from shapely.geometry import LineString, Point
            from shapely.ops import unary_union
            
            # 스켈레톤을 LineString으로 변환
            skeleton_lines = []
            skeleton_points = []
            
            for i, point in enumerate(self.skeleton):
                if len(point) >= 2:
                    skeleton_points.append((float(point[0]), float(point[1])))
            
            # 연속된 점들로 LineString 생성
            if len(skeleton_points) > 1:
                for i in range(len(skeleton_points) - 1):
                    try:
                        line = LineString([skeleton_points[i], skeleton_points[i + 1]])
                        if line.length > 1:  # 너무 짧은 선분 제외
                            skeleton_lines.append(line)
                    except:
                        continue
            
            if not skeleton_lines:
                logger.warning("유효한 스켈레톤 라인이 없습니다")
                return
            
            # 🌟 스켈레톤 라인들을 통합하고 지정된 버퍼 적용
            lines_union = unary_union(skeleton_lines)
            expanded_buffer = lines_union.buffer(buffer_pixels)
            
            # 📍 현재 점들 중에서 확장된 버퍼 영역 내부에 있는 점들 찾기
            points = self.canvas.points
            points_to_remove = []
            
            for category in ['intersection', 'curve', 'endpoint']:
                for i, point in enumerate(points[category]):
                    if len(point) >= 2:
                        x, y = float(point[0]), float(point[1])
                        point_geom = Point(x, y)
                        
                        # 확장된 버퍼 내부에 있는지 확인
                        if expanded_buffer.contains(point_geom) or expanded_buffer.touches(point_geom):
                            # 원본 스켈레톤 라인에서 너무 멀리 떨어진 점만 삭제 대상
                            min_dist_to_skeleton = min(
                                line.distance(point_geom) for line in skeleton_lines
                            )
                            
                            # 기본 스켈레톤 버퍼(20px) 밖에 있는 점들만 삭제
                            if min_dist_to_skeleton > 20:
                                points_to_remove.append({
                                    'category': category,
                                    'index': i,
                                    'distance': min_dist_to_skeleton
                                })
            
            # 거리 기준으로 정렬 (멀리 있는 점부터 삭제)
            points_to_remove.sort(key=lambda x: x['distance'], reverse=True)
            
            # 🗑️ 삭제 실행 (인덱스 역순 정렬하여 인덱스 문제 방지)
            removed_count = 0
            for remove_info in sorted(points_to_remove, key=lambda x: x['index'], reverse=True):
                try:
                    category = remove_info['category']
                    index = remove_info['index']
                    
                    if index < len(points[category]):
                        points[category].pop(index)
                        removed_count += 1
                        
                except (IndexError, KeyError):
                    continue
            
            # 화면 업데이트
            self.canvas.update_display()
            
            # 실시간 표 갱신
            if hasattr(self.parent().parent(), 'update_point_count_table'):
                self.parent().parent().update_point_count_table()
            
            # 결과 표시
            if hasattr(self.parent().parent(), 'statusBar'):
                self.parent().parent().statusBar().showMessage(
                    f"스켈레톤 버퍼 삭제 완료: {removed_count}개 점 제거 (버퍼: {buffer_pixels}px)", 3000
                )
            
            logger.info(f"스켈레톤 버퍼 삭제 완료: {removed_count}개 점 제거, 버퍼: {buffer_pixels}px")
            
        except Exception as e:
            logger.error(f"스켈레톤 버퍼 삭제 오류: {e}")
            if hasattr(self.parent().parent(), 'statusBar'):
                self.parent().parent().statusBar().showMessage(f"삭제 오류: {str(e)}", 3000)

    def update_distance_stats(self):
        """거리 통계 업데이트"""
        try:
            if hasattr(self.canvas, 'distance_info') and self.canvas.distance_info:
                distances = self.canvas.distance_info.get('distances', [])
                if distances:
                    # 거리 통계 계산
                    dist_values = [d['distance'] for d in distances]
                    min_dist = min(dist_values)
                    avg_dist = sum(dist_values) / len(dist_values)
                    max_dist = max(dist_values)
                    
                    text = f"최소: {min_dist:.1f}m\n평균: {avg_dist:.1f}m\n최대: {max_dist:.1f}m\n연결: {len(distances)}개"
                    self.distance_stats_label.setText(text)
                else:
                    self.distance_stats_label.setText("거리 정보 없음")
            else:
                self.distance_stats_label.setText("거리 정보 없음")
        except Exception as e:
            logger.error(f"거리 통계 업데이트 오류: {e}")
            self.distance_stats_label.setText("거리 계산 오류")

    def update_display(self):
        if self.current_file:
            self.file_label.setText(f"파일: {Path(self.current_file).name}")
        if self.road_geometry is not None:
            self.canvas.set_road_geometry(self.road_geometry)
        if self.skeleton is not None:
            self.canvas.set_skeleton(self.skeleton)
        self.update_stats()
        self.update_distance_stats()  # 거리 통계 업데이트
        self.canvas.update_display()

    def set_road_data(self, gdf):
        if gdf is not None and not gdf.empty:
            self.road_geometry = gdf.geometry.tolist()
            self.canvas.set_road_geometry(self.road_geometry, gdf.crs)
            self.canvas.draw_road_layer()
            self.canvas.scene.update()

    def set_background_data(self, gdf):
        """배경 데이터 설정 - GeoDataFrame 또는 단일 geometry 처리"""
        if gdf is None:
            return
            
        # GeoDataFrame인 경우
        if hasattr(gdf, 'geometry') and hasattr(gdf, 'crs'):
            self.background_geometry = gdf.geometry.tolist()
            self.canvas.set_background_geometry(self.background_geometry, gdf.crs)
        # 단일 geometry인 경우
        else:
            self.background_geometry = gdf
            self.canvas.set_background_geometry(self.background_geometry)

    def update_stats(self):
        if self.skeleton is None:
            self.stats_label.setText("통계: -")
            return
        i = len(self.canvas.points['intersection'])
        c = len(self.canvas.points['curve'])
        e = len(self.canvas.points['endpoint'])
        txt = f"교차점: {i} | 커브: {c} | 끝점: {e}"
        if self.processing_time > 0:
            txt += f" | 처리시간: {self.processing_time:.2f}s"
        if hasattr(self.canvas, 'skeleton_sampling_rate') and self.canvas.skeleton_sampling_rate > 1:
            txt += f" | 샘플링:1/{self.canvas.skeleton_sampling_rate}"
        self.stats_label.setText(txt)

    def update_legend(self):
        """범례 제거 - 빈 함수로 유지"""
        if hasattr(self, 'legend_label'):
            self.legend_label.setVisible(False)

    def on_point_added(self, category, x, y):
        self.update_stats()
        logger.info(f"포인트 추가: {category} at ({x:.2f}, {y:.2f})")

    def on_point_removed(self, category, x, y):
        self.update_stats()
        logger.info(f"포인트 제거: {category} at ({x:.2f}, {y:.2f})")

    def clear_all(self):
        self.current_file = None
        self.skeleton = None
        self.road_geometry = None
        self.background_geometry = None
        self.processing_time = 0
        self.canvas.clear_canvas()
        self.file_label.setText("파일: 없음")
        self.update_stats()

    def set_ai_mode(self, enabled):
        self.canvas.ai_assist_mode = enabled
        self.update_legend()

    def handle_deletion_prediction(self, predictions):
        if 'deletion' in predictions:
            for x, y in predictions['deletion']:
                if self.remove_nearest_point(x, y):
                    pass

    def run_dqn_prediction(self, confidence_threshold=0.7):
        return self.canvas.run_dqn_prediction(confidence_threshold)

    def toggle_dqn_prediction(self):
        return self.canvas.toggle_dqn_prediction()

    def get_dqn_model_info(self):
        return self.canvas.get_dqn_model_info()

    def set_dqn_epsilon(self, epsilon):
        return self.canvas.set_dqn_epsilon(epsilon)