#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
인터렉티브 커브점 선택 도구 - 미분 기반 곡률 계산
"""

import sys
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *

import geopandas as gpd
from shapely.geometry import Point, LineString
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter1d
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveCurveSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("인터렉티브 커브점 선택 도구 (미분 기반)")
        self.setGeometry(100, 100, 1600, 900)
        
        # 데이터 저장
        self.road_geometry = None
        self.curvature_data = []  # 각 도로의 곡률 데이터
        self.selected_curves = []  # 사용자가 선택한 커브점
        self.current_road_idx = 0
        
        # 시각화 상태
        self.curvature_points = []
        self.road_lines = []
        
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # 왼쪽 컨트롤 패널
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 오른쪽 시각화 영역
        viz_layout = QVBoxLayout()
        
        # 상단: 도로망 시각화
        self.road_figure = Figure(figsize=(12, 6))
        self.road_canvas = FigureCanvas(self.road_figure)
        self.road_ax = self.road_figure.add_subplot(111)
        self.road_ax.set_title("도로망 및 커브점 (클릭으로 선택)")
        self.road_ax.set_aspect('equal')
        viz_layout.addWidget(self.road_canvas)
        
        # 하단: 곡률 그래프
        self.curve_figure = Figure(figsize=(12, 3))
        self.curve_canvas = FigureCanvas(self.curve_figure)
        self.curve_ax = self.curve_figure.add_subplot(111)
        self.curve_ax.set_title("곡률 변화 그래프 (미분 기반)")
        viz_layout.addWidget(self.curve_canvas)
        
        viz_widget = QWidget()
        viz_widget.setLayout(viz_layout)
        layout.addWidget(viz_widget, stretch=1)
        
        # 마우스 클릭 이벤트 연결
        self.road_canvas.mpl_connect('button_press_event', self.on_road_click)
        self.curve_canvas.mpl_connect('button_press_event', self.on_curve_click)
    
    def create_control_panel(self):
        panel = QWidget()
        panel.setFixedWidth(350)
        panel.setStyleSheet("background-color: #f0f0f0; padding: 10px;")
        
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 1. 파일 로드
        file_group = QGroupBox("1. 도로망 파일")
        file_layout = QVBoxLayout()
        
        self.file_label = QLabel("파일: 선택 안됨")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        load_btn = QPushButton("📁 Shapefile 선택")
        load_btn.clicked.connect(self.load_road_file)
        load_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        file_layout.addWidget(load_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 2. 미분 기반 곡률 설정
        curve_group = QGroupBox("2. 미분 기반 곡률 설정")
        curve_layout = QVBoxLayout()
        
        # 샘플링 거리
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("샘플링 거리:"))
        self.sample_distance = QSpinBox()
        self.sample_distance.setRange(5, 50)
        self.sample_distance.setValue(10)
        self.sample_distance.setSuffix("m")
        sample_layout.addWidget(self.sample_distance)
        curve_layout.addLayout(sample_layout)
        
        # 스무딩 강도
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("스무딩 강도:"))
        self.smooth_sigma = QDoubleSpinBox()
        self.smooth_sigma.setRange(0.5, 5.0)
        self.smooth_sigma.setValue(1.5)
        self.smooth_sigma.setSingleStep(0.1)
        smooth_layout.addWidget(self.smooth_sigma)
        curve_layout.addLayout(smooth_layout)
        
        # 미분 차수
        diff_layout = QHBoxLayout()
        diff_layout.addWidget(QLabel("미분 차수:"))
        self.diff_order = QSpinBox()
        self.diff_order.setRange(1, 3)
        self.diff_order.setValue(2)
        diff_layout.addWidget(self.diff_order)
        curve_layout.addLayout(diff_layout)
        
        # 곡률 계산 버튼
        calc_btn = QPushButton("📊 곡률 계산")
        calc_btn.clicked.connect(self.calculate_curvature)
        calc_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        curve_layout.addWidget(calc_btn)
        
        curve_group.setLayout(curve_layout)
        layout.addWidget(curve_group)
        
        # 3. 도로 선택
        road_group = QGroupBox("3. 도로 선택")
        road_layout = QVBoxLayout()
        
        road_select_layout = QHBoxLayout()
        road_select_layout.addWidget(QLabel("도로 번호:"))
        self.road_selector = QSpinBox()
        self.road_selector.setRange(0, 0)
        self.road_selector.valueChanged.connect(self.change_road)
        road_select_layout.addWidget(self.road_selector)
        road_layout.addLayout(road_select_layout)
        
        # 자동 다음 도로
        self.auto_next = QCheckBox("자동 다음 도로")
        self.auto_next.setChecked(True)
        road_layout.addWidget(self.auto_next)
        
        road_group.setLayout(road_layout)
        layout.addWidget(road_group)
        
        # 4. 선택된 커브점
        selected_group = QGroupBox("4. 선택된 커브점")
        selected_layout = QVBoxLayout()
        
        self.selected_list = QListWidget()
        self.selected_list.setMaximumHeight(150)
        selected_layout.addWidget(self.selected_list)
        
        # 버튼들
        btn_layout = QHBoxLayout()
        
        clear_btn = QPushButton("🗑️ 전체 삭제")
        clear_btn.clicked.connect(self.clear_selected)
        clear_btn.setStyleSheet("background-color: #f44336; color: white; padding: 6px;")
        btn_layout.addWidget(clear_btn)
        
        undo_btn = QPushButton("↶ 마지막 취소")
        undo_btn.clicked.connect(self.undo_last)
        undo_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 6px;")
        btn_layout.addWidget(undo_btn)
        
        selected_layout.addLayout(btn_layout)
        selected_group.setLayout(selected_layout)
        layout.addWidget(selected_group)
        
        # 5. 결과 저장
        save_group = QGroupBox("5. 결과 저장")
        save_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(100)
        self.result_text.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        save_layout.addWidget(self.result_text)
        
        save_btn = QPushButton("💾 커브점 저장")
        save_btn.clicked.connect(self.save_curves)
        save_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        save_layout.addWidget(save_btn)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        layout.addStretch()
        return panel
    
    def load_road_file(self):
        """도로망 파일 로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "도로망 Shapefile 선택", "", "Shapefiles (*.shp)"
        )
        
        if file_path:
            try:
                gdf = gpd.read_file(file_path)
                
                # 모든 도로 geometry 수집
                all_geoms = []
                geom_stats = {'LineString': 0, 'MultiLineString': 0, 'Polygon': 0, 'MultiPolygon': 0, 'None': 0, 'Other': 0}
                
                for geom in gdf.geometry:
                    if geom is None:
                        geom_stats['None'] += 1
                        continue
                    elif geom.geom_type == 'LineString':
                        all_geoms.append(geom)
                        geom_stats['LineString'] += 1
                    elif geom.geom_type == 'MultiLineString':
                        all_geoms.extend(geom.geoms)
                        geom_stats['MultiLineString'] += 1
                    elif geom.geom_type == 'Polygon':
                        all_geoms.append(geom.exterior)
                        geom_stats['Polygon'] += 1
                    elif geom.geom_type == 'MultiPolygon':
                        for poly in geom.geoms:
                            all_geoms.append(poly.exterior)
                        geom_stats['MultiPolygon'] += 1
                    else:
                        geom_stats['Other'] += 1
                
                self.road_geometry = all_geoms
                self.file_label.setText(f"파일: {Path(file_path).name}")
                
                # 도로 선택기 업데이트
                self.road_selector.setRange(0, len(all_geoms) - 1)
                
                # 도로 그리기
                self.plot_roads()
                
                # 결과 표시
                self.result_text.append(f"✅ 도로망 로드 완료: {len(all_geoms)}개 도로")
                for geom_type, count in geom_stats.items():
                    if count > 0:
                        self.result_text.append(f"   - {geom_type}: {count}개")
                
            except Exception as e:
                self.result_text.append(f"❌ 로드 실패: {str(e)}")
    
    def plot_roads(self):
        """도로망 그리기"""
        self.road_ax.clear()
        self.road_lines.clear()
        
        if not self.road_geometry:
            return
        
        # 모든 도로 그리기
        for i, geom in enumerate(self.road_geometry):
            try:
                if geom.geom_type in ['LineString', 'LinearRing']:
                    x, y = geom.xy
                    if len(x) > 0 and len(y) > 0:
                        color = 'red' if i == self.current_road_idx else 'gray'
                        alpha = 1.0 if i == self.current_road_idx else 0.3
                        line = self.road_ax.plot(x, y, color=color, linewidth=3 if i == self.current_road_idx else 1, alpha=alpha)[0]
                        self.road_lines.append(line)
                elif hasattr(geom, 'coords'):
                    coords = list(geom.coords)
                    if len(coords) > 1:
                        x_coords = [coord[0] for coord in coords]
                        y_coords = [coord[1] for coord in coords]
                        color = 'red' if i == self.current_road_idx else 'gray'
                        alpha = 1.0 if i == self.current_road_idx else 0.3
                        line = self.road_ax.plot(x_coords, y_coords, color=color, linewidth=3 if i == self.current_road_idx else 1, alpha=alpha)[0]
                        self.road_lines.append(line)
            except Exception as e:
                continue
        
        # 선택된 커브점 표시
        if self.selected_curves:
            x_coords = [curve[0] for curve in self.selected_curves]
            y_coords = [curve[1] for curve in self.selected_curves]
            self.road_ax.scatter(x_coords, y_coords, c='blue', s=100, alpha=0.8, 
                               marker='o', edgecolors='darkblue', linewidth=2, label='선택된 커브점')
        
        # 현재 도로의 곡률 지점 표시
        if self.curvature_points:
            x_coords = [cp['point'][0] for cp in self.curvature_points]
            y_coords = [cp['point'][1] for cp in self.curvature_points]
            self.road_ax.scatter(x_coords, y_coords, c='orange', s=30, alpha=0.6, label='곡률 변화 지점')
        
        self.road_ax.set_title(f"도로망 및 커브점 (현재: 도로 {self.current_road_idx})")
        self.road_ax.set_aspect('equal')
        self.road_ax.grid(True, alpha=0.3)
        self.road_ax.legend()
        self.road_canvas.draw()
    
    def calculate_curvature(self):
        """미분 기반 곡률 계산"""
        if not self.road_geometry:
            self.result_text.append("❌ 먼저 도로망을 로드하세요")
            return
        
        sample_distance = self.sample_distance.value()
        smooth_sigma = self.smooth_sigma.value()
        diff_order = self.diff_order.value()
        
        self.result_text.append(f"🔍 미분 기반 곡률 계산 시작...")
        self.result_text.append(f"   - 샘플링 거리: {sample_distance}m")
        self.result_text.append(f"   - 스무딩 강도: {smooth_sigma}")
        self.result_text.append(f"   - 미분 차수: {diff_order}")
        
        # 모든 도로에 대해 곡률 계산
        self.curvature_data = []
        
        for i, geom in enumerate(self.road_geometry):
            try:
                if geom.geom_type not in ['LineString', 'LinearRing']:
                    continue
                
                # 도로 경계선 추출
                road_polygon = geom.buffer(6.0)
                if road_polygon.geom_type != 'Polygon':
                    continue
                
                boundary = road_polygon.exterior
                
                # 경계선을 따라 균등 샘플링
                total_length = boundary.length
                num_samples = max(10, int(total_length / sample_distance))
                
                distances = np.linspace(0, total_length, num_samples)
                points = []
                
                for dist in distances:
                    point = boundary.interpolate(dist)
                    points.append((point.x, point.y))
                
                # 미분 기반 곡률 계산
                curvature_values = self.calculate_differential_curvature(
                    points, smooth_sigma, diff_order
                )
                
                # 곡률 데이터 저장
                road_curvature = {
                    'road_idx': i,
                    'points': points,
                    'distances': distances,
                    'curvatures': curvature_values,
                    'boundary': boundary
                }
                self.curvature_data.append(road_curvature)
                
            except Exception as e:
                continue
        
        self.result_text.append(f"   → 곡률 계산 완료: {len(self.curvature_data)}개 도로")
        
        # 현재 도로의 곡률 표시
        self.change_road(self.current_road_idx)
    
    def calculate_differential_curvature(self, points, smooth_sigma, diff_order):
        """미분을 이용한 곡률 계산"""
        if len(points) < 5:
            return np.zeros(len(points))
        
        # 좌표 분리
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])
        
        # 가우시안 스무딩
        x_smooth = gaussian_filter1d(x_coords, sigma=smooth_sigma)
        y_smooth = gaussian_filter1d(y_coords, sigma=smooth_sigma)
        
        # 1차 미분 (속도)
        dx = np.gradient(x_smooth)
        dy = np.gradient(y_smooth)
        
        if diff_order == 1:
            # 1차 미분만 사용: 방향 변화율
            curvature = np.abs(np.gradient(np.arctan2(dy, dx)))
        elif diff_order == 2:
            # 2차 미분 사용: 가속도 기반 곡률
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            
            # 곡률 공식: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            numerator = np.abs(dx * d2y - dy * d2x)
            denominator = np.power(dx**2 + dy**2, 1.5)
            denominator = np.where(denominator == 0, 1e-10, denominator)
            curvature = numerator / denominator
        else:
            # 3차 미분 사용: 더 민감한 곡률
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            d3x = np.gradient(d2x)
            d3y = np.gradient(d2y)
            
            # 2차 미분 곡률
            numerator = np.abs(dx * d2y - dy * d2x)
            denominator = np.power(dx**2 + dy**2, 1.5)
            denominator = np.where(denominator == 0, 1e-10, denominator)
            curvature2 = numerator / denominator
            
            # 3차 미분 성분 추가
            curvature3 = np.abs(d3x + d3y)
            
            # 가중 결합
            curvature = curvature2 + 0.1 * curvature3
        
        return curvature
    
    def change_road(self, road_idx):
        """도로 변경"""
        self.current_road_idx = road_idx
        
        # 현재 도로의 곡률 데이터 찾기
        current_curvature = None
        for data in self.curvature_data:
            if data['road_idx'] == road_idx:
                current_curvature = data
                break
        
        if current_curvature is None:
            self.result_text.append(f"⚠️ 도로 {road_idx}의 곡률 데이터가 없습니다")
            return
        
        # 곡률 그래프 그리기
        self.plot_curvature_graph(current_curvature)
        
        # 곡률 지점 업데이트
        self.update_curvature_points(current_curvature)
        
        # 도로망 다시 그리기
        self.plot_roads()
    
    def plot_curvature_graph(self, curvature_data):
        """곡률 그래프 그리기"""
        self.curve_ax.clear()
        
        distances = curvature_data['distances']
        curvatures = curvature_data['curvatures']
        
        # 곡률 그래프
        self.curve_ax.plot(distances, curvatures, 'b-', linewidth=2, label='곡률')
        
        # 평균선
        mean_curvature = np.mean(curvatures)
        self.curve_ax.axhline(y=mean_curvature, color='r', linestyle='--', alpha=0.7, label=f'평균: {mean_curvature:.4f}')
        
        # 표준편차 기준선
        std_curvature = np.std(curvatures)
        threshold = mean_curvature + 2 * std_curvature
        self.curve_ax.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, label=f'임계값: {threshold:.4f}')
        
        self.curve_ax.set_xlabel('거리 (m)')
        self.curve_ax.set_ylabel('곡률')
        self.curve_ax.set_title(f'도로 {self.current_road_idx} 곡률 변화 (클릭으로 선택)')
        self.curve_ax.grid(True, alpha=0.3)
        self.curve_ax.legend()
        self.curve_canvas.draw()
    
    def update_curvature_points(self, curvature_data):
        """곡률 지점 업데이트"""
        # 임계값 계산 (평균 + 2*표준편차)
        curvatures = curvature_data['curvatures']
        mean_curvature = np.mean(curvatures)
        std_curvature = np.std(curvatures)
        threshold = mean_curvature + 2 * std_curvature
        
        # 임계값 이상인 지점들
        self.curvature_points = []
        for i, (point, curvature) in enumerate(zip(curvature_data['points'], curvatures)):
            if curvature > threshold:
                self.curvature_points.append({
                    'point': point,
                    'curvature': curvature,
                    'distance': curvature_data['distances'][i]
                })
    
    def on_road_click(self, event):
        """도로망에서 클릭 이벤트"""
        if event.inaxes != self.road_ax:
            return
        
        if event.button == 1:  # 왼쪽 클릭
            # 클릭한 위치에 가장 가까운 곡률 지점 찾기
            click_point = (event.xdata, event.ydata)
            
            if self.curvature_points:
                min_dist = float('inf')
                closest_point = None
                
                for cp in self.curvature_points:
                    dist = np.sqrt((cp['point'][0] - click_point[0])**2 + 
                                 (cp['point'][1] - click_point[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = cp
                
                # 50m 이내의 점만 선택
                if closest_point and min_dist < 50:
                    self.add_selected_curve(closest_point)
    
    def on_curve_click(self, event):
        """곡률 그래프에서 클릭 이벤트"""
        if event.inaxes != self.curve_ax:
            return
        
        if event.button == 1:  # 왼쪽 클릭
            # 클릭한 거리에 해당하는 곡률 지점 찾기
            click_distance = event.xdata
            
            # 현재 도로의 곡률 데이터 찾기
            current_curvature = None
            for data in self.curvature_data:
                if data['road_idx'] == self.current_road_idx:
                    current_curvature = data
                    break
            
            if current_curvature:
                # 가장 가까운 거리의 점 찾기
                distances = current_curvature['distances']
                idx = np.argmin(np.abs(distances - click_distance))
                
                selected_point = {
                    'point': current_curvature['points'][idx],
                    'curvature': current_curvature['curvatures'][idx],
                    'distance': distances[idx]
                }
                
                self.add_selected_curve(selected_point)
    
    def add_selected_curve(self, curve_point):
        """선택된 커브점 추가"""
        # 중복 확인 (20m 이내)
        for existing in self.selected_curves:
            dist = np.sqrt((existing[0] - curve_point['point'][0])**2 + 
                         (existing[1] - curve_point['point'][1])**2)
            if dist < 20:
                return
        
        # 커브점 추가
        self.selected_curves.append(curve_point['point'])
        
        # 리스트 업데이트
        self.update_selected_list()
        
        # 시각화 업데이트
        self.plot_roads()
        
        # 자동 다음 도로
        if self.auto_next.isChecked() and self.current_road_idx < len(self.road_geometry) - 1:
            self.road_selector.setValue(self.current_road_idx + 1)
    
    def update_selected_list(self):
        """선택된 커브점 리스트 업데이트"""
        self.selected_list.clear()
        for i, curve in enumerate(self.selected_curves):
            self.selected_list.addItem(f"{i+1}. ({curve[0]:.1f}, {curve[1]:.1f})")
    
    def clear_selected(self):
        """선택된 커브점 전체 삭제"""
        self.selected_curves.clear()
        self.update_selected_list()
        self.plot_roads()
    
    def undo_last(self):
        """마지막 선택 취소"""
        if self.selected_curves:
            self.selected_curves.pop()
            self.update_selected_list()
            self.plot_roads()
    
    def save_curves(self):
        """커브점 저장"""
        if not self.selected_curves:
            self.result_text.append("❌ 선택된 커브점이 없습니다")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "커브점 저장", "selected_curves.txt", "Text files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("# 선택된 커브점 좌표\n")
                    f.write("# X, Y\n")
                    for curve in self.selected_curves:
                        f.write(f"{curve[0]:.6f}, {curve[1]:.6f}\n")
                
                self.result_text.append(f"✅ 커브점 저장 완료: {len(self.selected_curves)}개")
                self.result_text.append(f"   파일: {Path(file_path).name}")
                
            except Exception as e:
                self.result_text.append(f"❌ 저장 실패: {str(e)}")


def main():
    app = QApplication(sys.argv)
    
    widget = InteractiveCurveSelector()
    widget.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main() 