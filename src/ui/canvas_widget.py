import numpy as np
from pathlib import Path
from ..core.user_action_tracker import UserActionTracker
import logging
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import requests
import math
from io import BytesIO
from typing import Optional, Dict, List, Any, Union

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsEllipseItem, QGraphicsRectItem,
    QGraphicsPolygonItem, QGraphicsPathItem, QMessageBox,
    QCheckBox, QPushButton, QSlider, QGroupBox
)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QPen, QBrush, QColor, QPainterPath, QPolygonF, QWheelEvent,
    QFont, QTransform, QPainter, QPixmap, QImage, QCursor,
    QKeyEvent, QMouseEvent
)

# Qt 열거형들과 상수들 정의 (타입 체크 무시)
from PyQt5.QtCore import QCoreApplication, QEvent, QTimer

# Qt 상수들 - 하드코딩된 값 사용 (타입 안전)
RoundCap = 0x10      # Qt.RoundCap
RoundJoin = 0x80     # Qt.RoundJoin  
KeepAspectRatio = 0x01  # Qt.KeepAspectRatio
NoPen = 0            # Qt.NoPen
DashLine = 2         # Qt.DashLine
Horizontal = 0x01    # Qt.Horizontal

logger = logging.getLogger(__name__)

class InteractiveCanvas(QGraphicsView):
    point_added = pyqtSignal(str, float, float)
    point_removed = pyqtSignal(str, float, float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.action_tracker = UserActionTracker()
        # QGraphicsScene 인스턴스 생성하고 설정
        graphics_scene = QGraphicsScene()
        self.setScene(graphics_scene)
        # scene 속성을 명시적으로 저장 (린터 오류 방지)
        self._graphics_scene = graphics_scene

        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.TextAntialiasing, False)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)

        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.zoom_factor = 1.15
        
        # 좌표계 정보 추가
        self.crs = 'EPSG:5186'  # 기본값: 중부원점
        
        # Panning 관련 변수
        self.panning_mode = False
        self.last_mouse_pos: Optional[QPointF] = None
        self.space_pressed = False
        
        # 디버깅을 위한 마우스 추적 활성화
        self.setMouseTracking(True)

        self.show_satellite_layer = False
        self.show_road_layer = True
        self.show_skeleton_layer = True
        self.show_points_layer = True
        self.show_background_layer = True
        self.show_distance_layer = True  # 거리선 표시 플래그 추가
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
        
        # 사용자 추가 점들 (얕은 빨간색으로 표시)
        self.user_added_points = []
        self.deleted_points_count = 0
        
        # 선택된 점
        self.selected_point = None
        self.selected_category = None
        
        # 점들의 속성 정보 저장 (보존 기준점 참조 등)
        self.point_attributes = {}  # {(x, y): {'reference': '보존기준점 참조', ...}}

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
        
        # 거리 분석 연결 정보 저장
        self.distance_connections = []
        
        # Excel 점 저장
        self.reference_points = []
        self.reference_points_display = []
        self.excel_items = []
        
        # 보존 기준점 표시 여부
        self.show_reference_points = True
        
        # 점 크기 통일 (2mm = 약 8픽셀)
        self.point_size = 8

    def keyPressEvent(self, event: QKeyEvent):
        k = event.key()
        
        # D키로 선택된 점 삭제
        if k == 0x44:  # Qt.Key_D
            if self.selected_point is not None and self.selected_category is not None:
                self.remove_selected_point()
                return
        
        # 스페이스바 처리 - 팬닝 모드와 fitInView 충돌 해결
        if k == 0x01000020 and not event.isAutoRepeat():  # Qt.Key_Space
            if not self.panning_mode:  # 팬닝 모드가 아닐 때만
                self.panning_mode = True
                self.space_pressed = True
                self.setCursor(Qt.OpenHandCursor)  # type: ignore
                # 드래그 모드는 변경하지 않음 (마우스 이벤트 차단 방지)
            return
        
        # 방향키 처리 추가
        if k == 0x01000012:  # Qt.Key_Left
            self.move_view(-50, 0)  # 왼쪽으로 50픽셀 이동
            return
        elif k == 0x01000014:  # Qt.Key_Right
            self.move_view(50, 0)   # 오른쪽으로 50픽셀 이동
            return
        elif k == 0x01000013:  # Qt.Key_Up
            self.move_view(0, -50)  # 위로 50픽셀 이동
            return
        elif k == 0x01000015:  # Qt.Key_Down
            self.move_view(0, 50)   # 아래로 50픽셀 이동
            return
        
        if k == 0x44:  # Qt.Key_D
            cp = self.mapFromGlobal(QCursor.pos())
            if self.rect().contains(cp):
                sp = self.mapToScene(cp)
                self.remove_nearest_point(sp.x(), -sp.y())
            self.setFocus()
            return

        if k == 0x41:  # Qt.Key_A
            self.show_ai_predictions = not self.show_ai_predictions
            self.update_display()
        elif k == 0x53 and event.modifiers() == 0x02000000:  # Qt.Key_S and Qt.ControlModifier
            self.accept_ai_suggestions()
        elif k == 0x52 and event.modifiers() == 0x02000000:  # Qt.Key_R and Qt.ControlModifier
            parent = self.parent()
            if parent and hasattr(parent.parent(), 'run_ai_detection'):
                parent.parent().run_ai_detection()  # type: ignore
        elif k == 0x31:  # Qt.Key_1
            self.toggle_road_layer(not self.show_road_layer)
        elif k == 0x32:  # Qt.Key_2
            self.toggle_skeleton_layer(not self.show_skeleton_layer)
        elif k == 0x33:  # Qt.Key_3
            self.toggle_points_layer(not self.show_points_layer)
        elif k == 0x51:  # Qt.Key_Q
            self.run_dqn_prediction()
        elif k == 0x54:  # Qt.Key_T
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

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == 0x01000020 and not event.isAutoRepeat():  # Qt.Key_Space
            self.panning_mode = False
            self.space_pressed = False
            self.setCursor(Qt.ArrowCursor)  # type: ignore
            # 원래 드래그 모드로 복원하지 않음 (이미 RubberBandDrag 상태)
            self.last_mouse_pos = None
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if self.panning_mode:
            if event.button() == 0x01:  # Qt.LeftButton
                self.last_mouse_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)  # type: ignore
            return
        
        p = self.mapToScene(event.pos())
        x, y = p.x(), -p.y()
        
        # 먼저 기존 점 선택 시도
        if self.try_select_point(x, y):
            # 점이 선택됨
            super().mousePressEvent(event)
            return
        
        # 점 선택이 안 되면 새 점 추가
        if not self.skeleton_points:
            super().mousePressEvent(event)
            return

        arr = np.array(self.skeleton_points) if isinstance(self.skeleton_points, (list, np.ndarray)) else np.array(self.skeleton_points)
        if len(arr) > 0:
            dists = np.hypot(arr[:,0]-x, arr[:,1]-y)
            mi = np.argmin(dists)
            if dists[mi] < 30:
                x, y = arr[mi]

        if event.modifiers() & 0x02000000:  # Qt.ShiftModifier
            self.remove_nearest_point(x, y)
        else:
            if event.button() == 0x01:  # Qt.LeftButton
                # 사용자가 추가한 점으로 기록
                self.add_user_point(x, y)
            elif event.button() == 0x02:  # Qt.RightButton
                self.add_point('endpoint', x, y)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.panning_mode and self.last_mouse_pos is not None:
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

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.panning_mode and event.button() == 0x01:  # Qt.LeftButton
            self.setCursor(Qt.OpenHandCursor)  # type: ignore
            self.last_mouse_pos = None
        
        super().mouseReleaseEvent(event)
        
        if not self.panning_mode and self.dragMode() == QGraphicsView.RubberBandDrag and self.show_satellite_layer:
            self.load_satellite_tiles()

    def wheelEvent(self, event: QWheelEvent):
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
                self.scene().removeItem(item)
            self.satellite_items.clear()
        self.viewport().update()

    def set_satellite_opacity(self, opacity):
        self.satellite_opacity = opacity / 100.0
        for item in self.satellite_items:
            item.setOpacity(self.satellite_opacity)

    def load_satellite_tiles(self):
        try:
            if not self.scene().items():
                return
            
            # transformer가 없거나 좌표계가 변경되었으면 재설정
            if not self.transformer or not hasattr(self, '_last_crs') or self._last_crs != self.crs:
                self.setup_coordinate_transformer(self.crs)
                self._last_crs = self.crs
                
            for item in self.satellite_items:
                self.scene().removeItem(item)
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
            
            # 네트워크 오류에 대한 강화된 처리
            try:
                resp = requests.get(url, timeout=15, stream=True)
            except (requests.exceptions.ChunkedEncodingError, 
                    requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout) as e:
                logger.warning(f"Satellite 타일 로딩 실패 (네트워크 오류): {e}")
                return
            except Exception as e:
                logger.warning(f"Satellite 타일 로딩 실패 (일반 오류): {e}")
                return
            
            if resp.status_code == 200:
                try:
                    # 안전하게 응답 데이터 읽기
                    content = resp.content
                    img = QImage()
                    if img.loadFromData(content):
                        pixmap = QPixmap.fromImage(img)
                        pixmap_item = self.scene().addPixmap(pixmap)
                        
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
                        else:
                            effective_width = img_actual_height * real_aspect_ratio
                            effective_height = img_actual_height
                        
                        scale_x = tm_width / effective_width
                        scale_y = tm_height / effective_height
                        
                        offset_x = (img_actual_width - effective_width) / 2
                        offset_y = (img_actual_height - effective_height) / 2
                        
                        final_x = tm_min_x - offset_x * scale_x
                        final_y = tm_max_y + offset_y * scale_y
                        
                        pixmap_item.setPos(final_x, -final_y)
                        pixmap_item.setScale(scale_x)
                        pixmap_item.setOpacity(self.satellite_opacity)
                        pixmap_item.setZValue(-10)
                        
                        self.satellite_items.append(pixmap_item)
                        logger.info(f"Satellite 타일 로딩 완료: {img_width}x{img_height} @ zoom {zoom}")
                    else:
                        logger.warning("Satellite 이미지 데이터 로딩 실패")
                        
                except (requests.exceptions.ChunkedEncodingError, 
                        requests.exceptions.ConnectionError) as e:
                    logger.warning(f"Satellite 이미지 데이터 읽기 실패: {e}")
                except Exception as e:
                    logger.warning(f"Satellite 이미지 처리 실패: {e}")
            else:
                logger.warning(f"Satellite 타일 로딩 실패: HTTP {resp.status_code}")
                
        except Exception as e:
            logger.error(f"위성 타일 로딩 전체 오류: {e}")

    def set_background_geometry(self, geometry, crs=None):
        self.background_geometry = geometry
        if crs:
            # CRS를 문자열로 직접 전달한 경우 (예: 'EPSG:5186')
            if isinstance(crs, str):
                self.original_crs = crs
                self.crs = crs
                self.setup_coordinate_transformer(crs)
                logger.info(f"배경 geometry 좌표계 설정 (문자열): {crs}")
            else:
                # CRS 객체인 경우
                crs_code = None
                if hasattr(crs, 'to_epsg'):
                    crs_code = crs.to_epsg()
                elif hasattr(crs, 'to_authority'):
                    auth = crs.to_authority()
                    if auth and auth[0] == 'EPSG':
                        crs_code = int(auth[1])
                
                # 좌표계 업데이트
                if crs_code:
                    self.original_crs = f'EPSG:{crs_code}'
                    self.crs = self.original_crs
                    self.setup_coordinate_transformer(self.original_crs)
                    logger.info(f"배경 geometry 좌표계 설정 (EPSG): {self.crs}")
                else:
                    self.original_crs = str(crs)
                    self.crs = self.original_crs
                    self.setup_coordinate_transformer(self.original_crs)
                    logger.info(f"배경 geometry 좌표계 설정 (기타): {self.crs}")
        self.draw_background_layer()
        
    def draw_background_layer(self):
        self.setUpdatesEnabled(False)
        try:
            # 안전하게 아이템 제거
            for item in self.background_items[:]:  # 리스트 복사본 사용
                try:
                    if item.scene() == self.scene():  # 아이템이 여전히 scene에 있는지 확인
                        self.scene().removeItem(item)
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
            self.scene().addItem(item)
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
        # 모든 아이템을 안전하게 제거
        try:
            self.scene().clear()
        except (RuntimeError, AttributeError):
            pass
            
        self.road_geometry = None
        self.background_geometry = None
        self.skeleton_points = None
        self.points = {'intersection': [], 'curve': [], 'endpoint': []}
        self.ai_points = {'intersection': [], 'curve': [], 'endpoint': [], 'delete': []}
        self.deleted_points = {'intersection': [], 'curve': [], 'endpoint': []}
        
        # 아이템 리스트들을 안전하게 정리
        self.road_items.clear()
        self.background_items.clear()
        self.skeleton_items.clear()
        self.point_items.clear()
        self.ai_point_items.clear()
        self.satellite_items.clear()
        
        # distance_items도 안전하게 정리
        if hasattr(self, 'distance_items'):
            for item in self.distance_items:
                try:
                    if item is not None and hasattr(item, 'scene') and item.scene() is not None:
                        self.scene().removeItem(item)
                except (RuntimeError, AttributeError):
                    pass
            self.distance_items.clear()
            
        # excel_items도 안전하게 정리
        if hasattr(self, 'excel_items'):
            self.excel_items.clear()
            
        try:
            self.resetTransform()
        except (RuntimeError, AttributeError):
            pass

    def set_road_geometry(self, geometry, crs=None):
        self.road_geometry = geometry
        if crs:
            # CRS를 문자열로 직접 전달한 경우 (예: 'EPSG:5186')
            if isinstance(crs, str):
                self.original_crs = crs
                self.crs = crs
                self.setup_coordinate_transformer(crs)
                logger.info(f"도로 geometry 좌표계 설정 (문자열): {crs}")
            else:
                # CRS 객체인 경우
                self.original_crs = crs
                self.crs = crs  # 캔버스의 좌표계 업데이트
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
                        self.crs = source_crs
                        self.original_crs = source_crs
                        logger.info(f"도로 geometry 좌표계 설정 (EPSG): {source_crs}")
                    except Exception as e:
                        logger.error(f"좌표계 변환기 설정 실패: {e}")
                        pass
        self.draw_road_layer()

    def draw_road_layer(self):
        self.setUpdatesEnabled(False)
        try:
            # 안전하게 아이템 제거
            for item in self.road_items[:]:  # 리스트 복사본 사용
                try:
                    if item.scene() == self.scene():  # 아이템이 여전히 scene에 있는지 확인
                        self.scene().removeItem(item)
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
            pen.setCapStyle(RoundCap)  # type: ignore  # Qt.RoundCap
            pen.setJoinStyle(RoundJoin)  # type: ignore  # Qt.RoundJoin
            item.setPen(pen)
            item.setOpacity(self.road_opacity)
            item.setZValue(-2)
            self.scene().addItem(item)
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
            self.scene().addItem(item)
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
            self.scene().setSceneRect(rect.adjusted(-50, -50, 50, 50))
            self.fitInView(self.scene().sceneRect(), KeepAspectRatio)  # type: ignore  # Qt.KeepAspectRatio
            self.scale(0.5, 0.5)  # 50% 축소하여 더 넓은 영역 보이게

    def draw_skeleton_layer(self):
        self.setUpdatesEnabled(False)
        try:
            # 안전하게 아이템 제거
            for item in self.skeleton_items[:]:  # 리스트 복사본 사용
                try:
                    if item.scene() == self.scene():  # 아이템이 여전히 scene에 있는지 확인
                        self.scene().removeItem(item)
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
        item.setZValue(5)  # 스켈레톤 레이어 Z-Order
        self.scene().addItem(item)
        self.skeleton_items.append(item)

    def _draw_skeleton_as_dots(self, points):
        for i in range(0, len(points), 100):
            path = QPainterPath()
            for x, y in points[i:i+100]:
                path.addEllipse(x-1, -y-1, 2, 2)
            item = QGraphicsPathItem(path)
            item.setPen(QPen(NoPen))  # type: ignore  # Qt.NoPen
            item.setBrush(QBrush(self.colors['skeleton']))
            item.setOpacity(self.skeleton_opacity)
            item.setZValue(5)  # 스켈레톤 레이어 Z-Order
            self.scene().addItem(item)
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
            # 보존기준점을 스켈레톤 바로 위에 그리기
            self.draw_reference_points()
            # AI/사용자 점들은 그 위에
            self.draw_points_layer()
            if self.show_ai_predictions and self.ai_points:
                self.draw_ai_predictions()
            # self.draw_distance_lines()  # 거리 표시 자동 호출 제거 - 버튼으로만 수동 실행
        finally:
            self.setUpdatesEnabled(True)
            self.scene().update()

    def draw_reference_points(self):
        """보존 기준점 그리기 (스켈레톤 바로 위에)"""
        # Excel 점 아이템 안전하게 제거
        if hasattr(self, 'excel_items'):
            for item in self.excel_items[:]:  # 리스트 복사본 사용
                try:
                    if item is not None and hasattr(item, 'scene') and item.scene() is not None:
                        self.scene().removeItem(item)
                except (RuntimeError, AttributeError):
                    pass
            self.excel_items.clear()
        else:
            self.excel_items = []
        
        # 보존 기준점 그리기
        if hasattr(self, 'reference_points_display') and self.reference_points_display and self.show_reference_points:
            point_size = self.point_size if hasattr(self, 'point_size') else 8
            diamond_size = point_size * 1.2  # 다이아몬드는 약간 크게
            
            for ref_display in self.reference_points_display:
                point = ref_display['point']
                color_name = ref_display.get('color', 'black')
                
                # 색상 설정 (연한 회색으로 변경)
                if color_name == 'purple':
                    ref_color = QColor(128, 0, 128)  # 보라색 (통합된 보존점)
                elif color_name == 'black':
                    ref_color = QColor(0, 0, 0)     # 검은색 (독립 보존점)
                elif color_name == 'gray':
                    ref_color = QColor(192, 192, 192)  # 연한 회색 (참고용 보존점)
                else:
                    ref_color = QColor(255, 0, 0)   # 기본 빨간색
                
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
                    item.setPen(QPen(ref_color.darker(120), 1))  # 테두리도 연하게
                    item.setBrush(QBrush(ref_color))
                    item.setZValue(7)  # 스켈레톤(5) 위, AI 점(10) 아래
                    self.scene().addItem(item)
                    self.excel_items.append(item)
    
    def draw_points_layer(self):
        # 안전하게 포인트 아이템 제거
        for item in self.point_items[:]:  # 리스트 복사본 사용
            try:
                if item is not None and hasattr(item, 'scene') and item.scene() is not None:
                    self.scene().removeItem(item)
            except (RuntimeError, AttributeError):
                # "wrapped C/C++ object has been deleted" 오류 방지
                pass
        self.point_items.clear()
        
        # 분석기준점 레이어가 꺼져있으면 여기서 return (Excel 점은 이미 위에서 처리됨)
        if not self.show_points_layer:
            return

        point_size = self.point_size if hasattr(self, 'point_size') else 8
        half_size = point_size / 2
        
        # 1. AI 점들 그리기 (파란색)
        ai_color = QColor(100, 100, 255)
        for category in ['intersection', 'curve', 'endpoint']:
            for point in self.points.get(category, []):
                if len(point) >= 2:
                    x, y = float(point[0]), float(point[1])
                else:
                    x, y = float(point), 0.0
                
                # 선택된 점인지 확인
                is_selected = (self.selected_point == (x, y) and self.selected_category == category)
                
                if is_selected:
                    # 선택된 점은 노란색 테두리
                    itm = self.scene().addEllipse(
                        x - half_size - 2, -y - half_size - 2, point_size + 4, point_size + 4,
                        QPen(QColor(255, 255, 0), 3),  # 노란색 테두리
                        QBrush(ai_color)
                    )
                else:
                    # 일반 AI 점
                    itm = self.scene().addEllipse(
                        x - half_size, -y - half_size, point_size, point_size,
                        QPen(ai_color, 2),
                        QBrush(ai_color)
                    )
                itm.setZValue(10)
                self.point_items.append(itm)
        
        # 2. 사용자가 추가한 점들 그리기 (얕은 빨간색)
        user_color = QColor(255, 150, 150)  # 얕은 빨간색
        for point in self.user_added_points:
            if len(point) >= 2:
                x, y = float(point[0]), float(point[1])
            else:
                continue
            
            # 선택된 점인지 확인
            is_selected = (self.selected_point == (x, y) and self.selected_category == 'user')
            
            if is_selected:
                # 선택된 점은 노란색 테두리
                itm = self.scene().addEllipse(
                    x - half_size - 2, -y - half_size - 2, point_size + 4, point_size + 4,
                    QPen(QColor(255, 255, 0), 3),  # 노란색 테두리
                    QBrush(user_color)
                )
            else:
                # 일반 사용자 추가 점
                itm = self.scene().addEllipse(
                    x - half_size, -y - half_size, point_size, point_size,
                    QPen(user_color.darker(120), 2),
                    QBrush(user_color)
                )
            itm.setZValue(11)  # AI 점보다 위에
            self.point_items.append(itm)

    def add_point(self, category, x, y):
        if category not in self.points:
            return
        if not self.is_point_exists(category, x, y):
            self.points[category].append((x, y))
            self.action_tracker.record_action('add', category, x, y)
            self.point_added.emit(category, x, y)
            # 화면 업데이트
            self.draw_points_layer()
    
    def add_user_point(self, x, y):
        """사용자가 추가한 점 (얕은 빨간색으로 표시)"""
        self.user_added_points.append((x, y))
        
        # 보존 기준점 근처인지 확인하고 속성 추가
        if hasattr(self, 'reference_points_display') and self.reference_points_display:
            import math
            min_distance = float('inf')
            closest_ref = None
            
            for ref_display in self.reference_points_display:
                ref_point = ref_display['point']
                if len(ref_point) >= 2:
                    ref_x, ref_y = float(ref_point[0]), float(ref_point[1])
                    distance = math.sqrt((x - ref_x)**2 + (y - ref_y)**2)
                    if distance <= 20.0 and distance < min_distance:
                        min_distance = distance
                        closest_ref = ref_display
            
            # 20m 이내에 보존 기준점이 있으면 속성 추가
            if closest_ref:
                point_key = (x, y)
                if not hasattr(self, 'point_attributes'):
                    self.point_attributes = {}
                
                self.point_attributes[point_key] = {
                    'reference': "보존기준점 참고",
                    'distance': round(min_distance, 2)
                }
        
        self.action_tracker.record_action('add', 'user', x, y)
        self.point_added.emit('user', x, y)
        self.draw_points_layer()
    
    def try_select_point(self, x, y):
        """점 선택 시도"""
        min_dist = 15  # 선택 반경
        
        # 모든 점 검사 (AI 점, 사용자 추가 점, 보존 점)
        for category, points in self.points.items():
            for i, (px, py) in enumerate(points):
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                if dist < min_dist:
                    self.selected_point = (px, py)
                    self.selected_category = category
                    self.draw_points_layer()
                    return True
        
        # 사용자 추가 점 검사
        for i, (px, py) in enumerate(self.user_added_points):
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < min_dist:
                self.selected_point = (px, py)
                self.selected_category = 'user'
                self.draw_points_layer()
                return True
        
        # 선택 해제
        self.selected_point = None
        self.selected_category = None
        self.draw_points_layer()
        return False
    
    def remove_selected_point(self):
        """선택된 점 삭제"""
        if self.selected_point is None:
            return
        
        x, y = self.selected_point
        
        if self.selected_category == 'user':
            # 사용자 추가 점 삭제
            if self.selected_point in self.user_added_points:
                self.user_added_points.remove(self.selected_point)
                self.deleted_points_count += 1
        elif self.selected_category in self.points:
            # AI 점 삭제
            if self.selected_point in self.points[self.selected_category]:
                self.points[self.selected_category].remove(self.selected_point)
                self.deleted_points_count += 1
        
        self.selected_point = None
        self.selected_category = None
        self.action_tracker.record_action('remove', self.selected_category, x, y)
        self.point_removed.emit(self.selected_category, x, y)
        self.draw_points_layer()

    def remove_point(self, category, x, y):
        if category not in self.points:
            return
        for i, (px, py) in enumerate(self.points[category]):
            if abs(px - x) < 5 and abs(py - y) < 5:
                removed_point = self.points[category].pop(i)
                self.deleted_points[category].append(removed_point)
                self.action_tracker.record_action('remove', category, x, y)
                self.point_removed.emit(category, x, y)
                # 화면 업데이트
                self.draw_points_layer()
                break

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
        
        # 제거 (30 픽셀 임계값 내에서만)
        if remove_category and remove_idx is not None and min_dist < 30:
            removed = self.points[remove_category].pop(remove_idx)
            # 삭제된 포인트 추적
            self.deleted_points[remove_category].append(removed)
            self.action_tracker.record_action('remove', remove_category, removed[0], removed[1])
            self.point_removed.emit(remove_category, removed[0], removed[1])
            # 화면 업데이트
            self.draw_points_layer()
            return True
        
        return False

    def draw_ai_predictions(self):
        """AI 예측 결과 표시"""
        # 안전하게 AI 아이템 제거
        for item in self.ai_point_items[:]:  # 리스트 복사본 사용
            try:
                if item is not None and hasattr(item, 'scene') and item.scene() is not None:
                    self.scene().removeItem(item)
            except (RuntimeError, AttributeError):
                # "wrapped C/C++ object has been deleted" 오류 방지
                pass
        self.ai_point_items.clear()

        if not self.ai_points or not self.show_ai_predictions:
            return

        # AI 점 통일 색상 (주황색)
        ai_unified_color = QColor(255, 165, 0)

        # AI 교차점 - 주황색 원 (통일)
        for point in self.ai_points['intersection']:
            if len(point) >= 2:
                x, y = float(point[0]), float(point[1])
            else:
                x, y = float(point), 0.0
            itm = self.scene().addEllipse(
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
            itm = self.scene().addEllipse(
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
            itm = self.scene().addEllipse(
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
            self.scene().addItem(itm)
            self.ai_point_items.append(itm)

    def draw_distance_lines(self):
        """네트워크 연결선 및 거리 정보 표시 (검은색 점선)"""
        # 거리 레이어가 숨겨져 있으면 바로 거리선 제거
        if not self.show_distance_layer:
            self.clear_distance_lines()
            return
            
        # 네트워크 연결과 거리 분석 연결을 모두 확인
        has_network = hasattr(self, 'network_connections') and self.network_connections
        has_distance = hasattr(self, 'distance_connections') and self.distance_connections
        
        logger.info(f"🔍 draw_distance_lines 호출됨 - has_network: {has_network}, has_distance: {has_distance}")
        if has_distance:
            logger.info(f"🔍 distance_connections 개수: {len(self.distance_connections)}")
            # 각 연결의 세부 정보 출력
            for i, conn in enumerate(self.distance_connections[:3]):  # 처음 3개만 출력
                logger.info(f"🔍 연결 {i+1}: {conn}")
        
        # distance_connections가 없으면 빈 리스트로 초기화
        if not hasattr(self, 'distance_connections'):
            self.distance_connections = []
            logger.info("🔍 distance_connections 속성이 없어서 빈 리스트로 초기화")
        
        if not has_network and not has_distance:
            logger.info("🔍 연결 정보가 없어서 거리 선 그리기 중단")
            return
            
        try:
            # 기존 거리 선 안전하게 제거
            if hasattr(self, 'distance_items'):
                for item in self.distance_items:
                    try:
                        # Qt 객체가 아직 유효한지 확인
                        if item is not None and hasattr(item, 'scene') and item.scene() is not None:
                            self.scene().removeItem(item)
                    except (RuntimeError, AttributeError):
                        # "wrapped C/C++ object has been deleted" 오류 방지
                        pass
            
            self.distance_items = []
            
            # 연결선 그리기 - 우선 거리 분석 연결을 그리고, 없으면 네트워크 연결
            connections_to_draw = []
            if has_distance:
                connections_to_draw = self.distance_connections
                logger.info(f"📏 거리 분석 선 그리기 시작: {len(self.distance_connections)}개 연결")
            elif has_network:
                connections_to_draw = self.network_connections
                logger.info(f"🌐 네트워크 연결선 그리기 시작: {len(self.network_connections)}개 연결")
            
            # 🚨 중요: connections_to_draw가 비어있는지 확인
            if not connections_to_draw:
                logger.warning(f"❌ connections_to_draw가 비어있습니다! has_distance: {has_distance}, has_network: {has_network}")
                # 실제 속성 값들도 확인
                if hasattr(self, 'distance_connections'):
                    logger.warning(f"❌ self.distance_connections 실제값: {self.distance_connections}")
                if hasattr(self, 'network_connections'):
                    logger.warning(f"❌ self.network_connections 실제값: {getattr(self, 'network_connections', 'None')}")
                return
            
            logger.info(f"✅ 실제 그리기 시작: {len(connections_to_draw)}개 연결")
            
            # 🌐 연결선 그리기 (검은색 점선)
            drawn_count = 0
            for i, connection in enumerate(connections_to_draw):
                try:
                    p1 = connection['point1']
                    p2 = connection['point2']
                    distance = connection['distance']
                    
                    logger.info(f"📏 연결 {i+1} 그리기: {p1} -> {p2}, {distance:.1f}m")
                    
                    # 검은색 점선으로 거리 연결선 그리기 (원래 스타일)
                    path = QPainterPath()
                    path.moveTo(p1[0], -p1[1])
                    path.lineTo(p2[0], -p2[1])
                    
                    line_item = QGraphicsPathItem(path)
                    pen = QPen(QColor(0, 0, 0), 1)  # 검은색
                    pen.setStyle(DashLine)  # type: ignore  # Qt.DashLine 점선
                    line_item.setPen(pen)
                    line_item.setZValue(5)
                    self.scene().addItem(line_item)
                    self.distance_items.append(line_item)
                    
                    # 거리 텍스트 추가 (중앙에 m 단위)
                    mid_x = (p1[0] + p2[0]) / 2
                    mid_y = -(p1[1] + p2[1]) / 2
                    
                    text_item = self.scene().addText(f"{distance:.1f}m", QFont("Arial", 8))
                    text_item.setPos(mid_x, mid_y)
                    text_item.setZValue(6)
                    # 텍스트 배경 (가독성을 위해)
                    text_item.setDefaultTextColor(QColor(0, 0, 0))
                    self.distance_items.append(text_item)
                    
                    drawn_count += 1
                    
                except Exception as e:
                    logger.error(f"연결 {i+1} 그리기 실패: {e}")
                    continue
            
            logger.info(f"🎉 거리선 그리기 완료: {drawn_count}개 연결 그려짐")
            
        except Exception as e:
            logger.error(f"네트워크 연결선 그리기 오류: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")

    def clear_distance_lines(self):
        """거리선과 텍스트 제거"""
        try:
            if hasattr(self, 'distance_items'):
                for item in self.distance_items:
                    try:
                        if item is not None and hasattr(item, 'scene') and item.scene() is not None:
                            self.scene().removeItem(item)
                    except (RuntimeError, AttributeError):
                        pass
                self.distance_items = []
        except Exception as e:
            logger.error(f"거리선 제거 오류: {e}")

    def toggle_distance_layer(self, visible):
        """거리 레이어 표시/숨김 토글"""
        self.show_distance_layer = visible
        if visible:
            self.draw_distance_lines()
        else:
            self.clear_distance_lines()

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
        try:
            parent = self.parent()
            if parent and hasattr(parent.parent(), 'statusBar'):
                status_bar = getattr(parent.parent(), 'statusBar', None)
                if status_bar and callable(status_bar):
                    status_bar().showMessage(f"{cnt}개의 AI 제안이 수락되었습니다.", 3000)  # type: ignore
        except AttributeError:
            pass  # statusBar가 없는 경우 무시

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
        # self.stats_label = QLabel("통계: -")  # 통계 라벨 숨김
        # self.stats_label.setStyleSheet("padding:5px;")
        # toolbar_layout.addWidget(self.stats_label)
        layout.addLayout(toolbar_layout)

        self.canvas = InteractiveCanvas(self)
        self.canvas.point_added.connect(self.on_point_added)
        self.canvas.point_removed.connect(self.on_point_removed)
        layout.addWidget(self.canvas)

        self.layer_control = self.create_layer_control()
        self.layer_control.setParent(self.canvas)
        self.layer_control.move(10, 60)

        # 메뉴얼 삭제 바 추가
        # 수동 삭제 도구 제거됨

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
        self.sat_op = QSlider(Horizontal)  # type: ignore  # Qt.Horizontal
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
        self.bg_op = QSlider(Horizontal)  # type: ignore  # Qt.Horizontal
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
        self.rd_op = QSlider(Horizontal)  # type: ignore  # Qt.Horizontal
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
        self.sk_op = QSlider(Horizontal)  # type: ignore  # Qt.Horizontal
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

    # create_manual_deletion_bar 메서드 제거됨 - 수동 삭제 도구 완전 제거
    
    # update_deletion_label 메서드 제거됨

    # perform_manual_deletion 메서드 제거됨 - 수동 삭제 도구 완전 제거

    def update_distance_stats(self):
        """거리 통계 업데이트 (로그 출력만)"""
        try:
            # 거리 정보 안전하게 처리 (타입 오류 방지)
            canvas = getattr(self, 'canvas', None)
            if canvas and hasattr(canvas, 'distance_info'):
                distance_info = getattr(canvas, 'distance_info', None)
                if distance_info and isinstance(distance_info, dict):
                    distances = distance_info.get('distances', [])
                    # 타입 체크를 통한 안전한 처리
                    if distances and hasattr(distances, '__iter__'):
                        # 거리 통계 계산 (타입 안전성 보장)
                        dist_values: List[float] = []
                        for d in distances:  # type: ignore
                            if isinstance(d, dict) and 'distance' in d:
                                dist_values.append(float(d['distance']))
                        
                        if dist_values:
                            min_dist = min(dist_values)
                            avg_dist = sum(dist_values) / len(dist_values)
                            max_dist = max(dist_values)
                            logger.info(f"📏 거리 통계 - 최소: {min_dist:.1f}m, 평균: {avg_dist:.1f}m, 최대: {max_dist:.1f}m, 연결: {len(distances)}개")
                        else:
                            logger.info("📏 거리 정보: 유효한 거리 데이터가 없습니다")
                    else:
                        logger.info("📏 거리 정보: 연결된 점이 없습니다")
                else:
                    logger.info("📏 거리 정보: 계산된 거리가 없습니다")
            else:
                logger.info("📏 거리 정보: 캔버스 정보가 없습니다")
        except Exception as e:
            logger.error(f"거리 통계 업데이트 오류: {e}")

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
            self.road_gdf = gdf  # ← 도로 폴리곤 기반 거리 계산용으로 추가
            self.road_geometry = gdf.geometry.tolist()
            # 캔버스의 현재 좌표계를 사용
            target_crs = self.canvas.crs if hasattr(self.canvas, 'crs') else 'EPSG:5186'
            # 좌표계가 다르면 변환
            if gdf.crs and str(gdf.crs) != target_crs:
                gdf = gdf.to_crs(target_crs)
                self.road_gdf = gdf
                self.road_geometry = gdf.geometry.tolist()
            self.canvas.set_road_geometry(self.road_geometry, target_crs)
            self.canvas.draw_road_layer()
            self.canvas.scene().update()

    def set_background_data(self, gdf):
        """배경 데이터 설정 - GeoDataFrame 또는 단일 geometry 처리"""
        if gdf is None:
            return
            
        # 캔버스의 현재 좌표계를 사용
        target_crs = self.canvas.crs if hasattr(self.canvas, 'crs') else 'EPSG:5186'
        logger.info(f"set_background_data - 캔버스 좌표계: {target_crs}")
        
        # GeoDataFrame인 경우
        if hasattr(gdf, 'geometry') and hasattr(gdf, 'crs'):
            logger.info(f"set_background_data - GeoDataFrame CRS: {gdf.crs}")
            # 좌표계가 다르면 변환
            if gdf.crs and str(gdf.crs) != target_crs:
                logger.info(f"set_background_data - 좌표계 변환: {gdf.crs} → {target_crs}")
                gdf = gdf.to_crs(target_crs)
            self.background_geometry = gdf.geometry.tolist()
            self.canvas.set_background_geometry(self.background_geometry, target_crs)
        # 단일 geometry인 경우
        else:
            logger.info(f"set_background_data - 단일 geometry (타입: {type(gdf)})")
            self.background_geometry = gdf
            self.canvas.set_background_geometry(self.background_geometry, target_crs)

    def update_stats(self):
        # 통계 표시 기능 비활성화 - UI 간소화
        # if self.skeleton is None:
        #     self.stats_label.setText("통계: -")
        #     return
        # i = len(self.canvas.points['intersection'])
        # c = len(self.canvas.points['curve'])
        # e = len(self.canvas.points['endpoint'])
        # txt = f"교차점: {i} | 커브: {c} | 끝점: {e}"
        # if self.processing_time > 0:
        #     txt += f" | 처리시간: {self.processing_time:.2f}s"
        # if hasattr(self.canvas, 'skeleton_sampling_rate') and self.canvas.skeleton_sampling_rate > 1:
        #     txt += f" | 샘플링:1/{self.canvas.skeleton_sampling_rate}"
        # self.stats_label.setText(txt)
        pass  # 통계 업데이트 비활성화

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

    def toggle_distance_display(self, checked):
        """거리 표시 토글"""
        self.canvas.toggle_distance_layer(checked)