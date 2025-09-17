#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
도로 경계선 기반 커브점 추출 테스트 도구
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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurveDetectionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("도로 경계선 기반 커브점 추출 테스트")
        self.setGeometry(100, 100, 1400, 800)
        
        # 데이터 저장
        self.road_geometry = None
        self.curvature_points = []
        self.final_curves = []
        
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # 왼쪽 컨트롤 패널
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 오른쪽 matplotlib 캔버스
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("도로 경계선 기반 커브점 추출")
        self.ax.set_aspect('equal')
        
    def create_control_panel(self):
        panel = QWidget()
        panel.setFixedWidth(300)
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
        
        # 2. 설정
        settings_group = QGroupBox("2. 설정")
        settings_layout = QVBoxLayout()
        
        # 샘플링 거리
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("샘플링 거리:"))
        self.sample_distance = QSpinBox()
        self.sample_distance.setRange(5, 50)
        self.sample_distance.setValue(15)
        self.sample_distance.setSuffix("m")
        sample_layout.addWidget(self.sample_distance)
        settings_layout.addLayout(sample_layout)
        
        # 곡률 임계값
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("곡률 임계값:"))
        self.curvature_threshold = QDoubleSpinBox()
        self.curvature_threshold.setRange(0.01, 1.0)
        self.curvature_threshold.setValue(0.08)  # 0.15 → 0.08로 낮춤
        self.curvature_threshold.setSingleStep(0.01)
        threshold_layout.addWidget(self.curvature_threshold)
        settings_layout.addLayout(threshold_layout)
        
        # 도로 버퍼 크기
        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("도로 버퍼:"))
        self.road_buffer = QSpinBox()
        self.road_buffer.setRange(3, 15)
        self.road_buffer.setValue(6)
        self.road_buffer.setSuffix("m")
        buffer_layout.addWidget(self.road_buffer)
        settings_layout.addLayout(buffer_layout)
        
        # 군집 반경
        cluster_layout = QHBoxLayout()
        cluster_layout.addWidget(QLabel("군집 반경:"))
        self.cluster_radius = QSpinBox()
        self.cluster_radius.setRange(10, 100)
        self.cluster_radius.setValue(30)
        self.cluster_radius.setSuffix("m")
        cluster_layout.addWidget(self.cluster_radius)
        settings_layout.addLayout(cluster_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # 3. 실행
        exec_group = QGroupBox("3. 실행")
        exec_layout = QVBoxLayout()
        
        detect_btn = QPushButton("🔍 커브점 검출")
        detect_btn.clicked.connect(self.detect_curves)
        detect_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        exec_layout.addWidget(detect_btn)
        
        clear_btn = QPushButton("🗑️ 초기화")
        clear_btn.clicked.connect(self.clear_plot)
        clear_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        exec_layout.addWidget(clear_btn)
        
        exec_group.setLayout(exec_layout)
        layout.addWidget(exec_group)
        
        # 4. 결과
        result_group = QGroupBox("4. 결과")
        result_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(150)
        self.result_text.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        result_layout.addWidget(self.result_text)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
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
                
                # 모든 도로 geometry 수집 및 통계
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
                        # 폴리곤의 경우 경계선을 LineString으로 변환
                        all_geoms.append(geom.exterior)
                        geom_stats['Polygon'] += 1
                    elif geom.geom_type == 'MultiPolygon':
                        # 멀티폴리곤의 경우 각 폴리곤의 경계선을 추출
                        for poly in geom.geoms:
                            all_geoms.append(poly.exterior)
                        geom_stats['MultiPolygon'] += 1
                    else:
                        geom_stats['Other'] += 1
                
                self.road_geometry = all_geoms
                self.file_label.setText(f"파일: {Path(file_path).name}")
                
                # 도로 그리기
                self.plot_roads()
                
                # 상세 로드 결과 표시
                self.result_text.append(f"✅ 도로망 로드 완료: {len(all_geoms)}개 도로")
                self.result_text.append(f"   📊 Geometry 타입별 통계:")
                for geom_type, count in geom_stats.items():
                    if count > 0:
                        self.result_text.append(f"      - {geom_type}: {count}개")
                
            except Exception as e:
                self.result_text.append(f"❌ 로드 실패: {str(e)}")
    
    def plot_roads(self):
        """도로망 그리기"""
        self.ax.clear()
        
        if not self.road_geometry:
            self.result_text.append("❌ 도로 geometry가 없습니다")
            return
        
        # 도로 그리기 (디버깅 정보 추가)
        plotted_count = 0
        geom_types = {}
        
        for i, geom in enumerate(self.road_geometry):
            geom_type = geom.geom_type
            geom_types[geom_type] = geom_types.get(geom_type, 0) + 1
            
            try:
                # LineString, LinearRing 모두 처리
                if geom_type in ['LineString', 'LinearRing']:
                    x, y = geom.xy
                    if len(x) > 0 and len(y) > 0:
                        self.ax.plot(x, y, 'gray', linewidth=2, alpha=0.7)
                        plotted_count += 1
                elif hasattr(geom, 'coords'):
                    # 좌표가 있는 다른 타입들 처리
                    coords = list(geom.coords)
                    if len(coords) > 1:
                        x_coords = [coord[0] for coord in coords]
                        y_coords = [coord[1] for coord in coords]
                        self.ax.plot(x_coords, y_coords, 'gray', linewidth=2, alpha=0.7)
                        plotted_count += 1
            except Exception as e:
                self.result_text.append(f"⚠️ 도로 {i} 그리기 실패: {str(e)}")
                continue
        
        # 디버깅 정보 출력
        self.result_text.append(f"🎨 시각화 완료: {plotted_count}개 도로 그리기")
        self.result_text.append(f"   📊 그려진 Geometry 타입:")
        for geom_type, count in geom_types.items():
            self.result_text.append(f"      - {geom_type}: {count}개")
        
        self.ax.set_title("도로 경계선 기반 커브점 추출")
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)  # 격자 추가로 확인
        self.canvas.draw()
    
    def detect_curves(self):
        """커브점 검출"""
        if not self.road_geometry:
            self.result_text.append("❌ 먼저 도로망을 로드하세요")
            return
        
        # 설정값 가져오기
        sample_distance = self.sample_distance.value()
        curvature_threshold = self.curvature_threshold.value()
        road_buffer = self.road_buffer.value()
        cluster_radius = self.cluster_radius.value()
        
        self.result_text.append(f"🔍 커브점 검출 시작...")
        self.result_text.append(f"   - 샘플링 거리: {sample_distance}m")
        self.result_text.append(f"   - 곡률 임계값: {curvature_threshold}")
        self.result_text.append(f"   - 도로 버퍼: {road_buffer}m")
        
        # 1단계: 도로 경계선에서 곡률 변화 검출
        self.curvature_points = []
        processed_count = 0
        
        for i, geom in enumerate(self.road_geometry):
            # LineString과 LinearRing 모두 처리
            if geom.geom_type not in ['LineString', 'LinearRing']:
                continue
            
            processed_count += 1
            
            try:
                # 도로를 버퍼로 확장
                road_polygon = geom.buffer(road_buffer)
                if road_polygon.geom_type != 'Polygon':
                    self.result_text.append(f"⚠️ 도로 {i}: 버퍼 결과가 Polygon이 아님 ({road_polygon.geom_type})")
                    continue
                
                # 경계선 추출
                boundary = road_polygon.exterior
                total_length = boundary.length
                
                # 경계선을 따라 샘플링
                num_samples = max(5, int(total_length / sample_distance))
                curvature_found = 0
                
                for j in range(num_samples):
                    distance = (j * sample_distance) % total_length
                    
                    # 곡률 계산
                    curvature = self.calculate_curvature_at_distance(
                        boundary, distance, sample_distance
                    )
                    
                    if curvature > curvature_threshold:
                        point = boundary.interpolate(distance)
                        self.curvature_points.append({
                            'point': (point.x, point.y),
                            'curvature': curvature
                        })
                        curvature_found += 1
                
                # 디버깅: 첫 번째 도로의 곡률 정보 출력
                if i == 0:
                    self.result_text.append(f"🔍 도로 0 분석:")
                    self.result_text.append(f"   - 길이: {total_length:.1f}m")
                    self.result_text.append(f"   - 샘플 수: {num_samples}개")
                    self.result_text.append(f"   - 발견된 곡률점: {curvature_found}개")
                        
            except Exception as e:
                self.result_text.append(f"⚠️ 도로 {i} 처리 실패: {str(e)}")
                continue
        
        self.result_text.append(f"   → 처리된 도로: {processed_count}개")
        self.result_text.append(f"   → 곡률 변화 지점: {len(self.curvature_points)}개")
        
        # 2단계: 군집화
        if len(self.curvature_points) > 1:
            points = np.array([cp['point'] for cp in self.curvature_points])
            
            # DBSCAN 군집화
            clustering = DBSCAN(eps=cluster_radius, min_samples=2)
            labels = clustering.fit_predict(points)
            
            # 군집별 중심점 계산
            self.final_curves = []
            unique_labels = set(labels)
            
            for label in unique_labels:
                if label == -1:  # 노이즈 제외
                    continue
                
                cluster_mask = labels == label
                cluster_points = points[cluster_mask]
                
                # 중심점 계산
                center_x = np.mean(cluster_points[:, 0])
                center_y = np.mean(cluster_points[:, 1])
                self.final_curves.append((center_x, center_y))
            
            self.result_text.append(f"   → 군집화 후 최종 커브점: {len(self.final_curves)}개")
        
        # 3단계: 시각화
        self.plot_results()
    
    def calculate_curvature_at_distance(self, boundary, distance, window=20.0):
        """특정 거리에서의 곡률 계산"""
        try:
            # 앞뒤 점들 구하기
            d1 = max(0, distance - window)
            d2 = min(boundary.length, distance + window)
            
            # 윈도우 크기를 더 관대하게 설정
            if d2 - d1 < window * 0.5:  # 원래 window에서 절반으로 완화
                return 0.0
            
            p1 = boundary.interpolate(d1)
            p2 = boundary.interpolate(distance)
            p3 = boundary.interpolate(d2)
            
            # 벡터 계산
            v1 = np.array([p2.x - p1.x, p2.y - p1.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            # 각도 변화 계산
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 > 0 and len2 > 0:
                v1_norm = v1 / len1
                v2_norm = v2 / len2
                
                cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                return angle
            
            return 0.0
            
        except:
            return 0.0
    
    def plot_results(self):
        """결과 시각화"""
        # 기존 도로 다시 그리기
        self.plot_roads()
        
        # 곡률 변화 지점 (작은 빨간 점)
        if self.curvature_points:
            x_coords = [cp['point'][0] for cp in self.curvature_points]
            y_coords = [cp['point'][1] for cp in self.curvature_points]
            self.ax.scatter(x_coords, y_coords, c='red', s=10, alpha=0.6, label='곡률 변화 지점')
        
        # 최종 커브점 (큰 파란 원)
        if self.final_curves:
            x_coords = [curve[0] for curve in self.final_curves]
            y_coords = [curve[1] for curve in self.final_curves]
            self.ax.scatter(x_coords, y_coords, c='blue', s=100, alpha=0.8, 
                          marker='o', edgecolors='darkblue', linewidth=2, label='최종 커브점')
        
        self.ax.legend()
        self.ax.set_title(f"커브점 검출 결과 (총 {len(self.final_curves)}개)")
        self.canvas.draw()
    
    def clear_plot(self):
        """플롯 초기화"""
        self.ax.clear()
        self.ax.set_title("도로 경계선 기반 커브점 추출")
        self.ax.set_aspect('equal')
        self.canvas.draw()
        
        self.curvature_points.clear()
        self.final_curves.clear()
        self.result_text.clear()


def main():
    app = QApplication(sys.argv)
    
    widget = CurveDetectionWidget()
    widget.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main() 