import os
import sys
import time
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from functools import lru_cache
import numpy as np
import geopandas as gpd

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QProgressBar, QFileDialog, QMessageBox,
    QAction, QToolBar, QCheckBox, QDialog, QTableWidget, QTableWidgetItem,
    QTextEdit, QListWidgetItem, QGroupBox, QTabWidget, QProgressDialog,
    QApplication, QSpinBox, QComboBox, QDoubleSpinBox, QSplitter,
    QRadioButton, QInputDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor

from ..core.skeleton_extractor import SkeletonExtractor
from ..core.district_road_clipper import DistrictRoadClipper
from ..config import *
from .canvas_widget import CanvasWidget
from ..utils import save_session, load_session, list_sessions, extract_point_features
from ..learning.dqn_model import create_agent
from ..learning import DQNDataCollector
from ..core.batch_processor import BatchProcessor, PerformanceMonitor
import torch

# RL 관련 추가 import
from src.learning.rl_dqn.environment import SurveyPointEnvironment
from src.learning.rl_dqn.agent import RLDQNAgent
from src.core.visibility_checker import extract_road_polygons_from_gdf
from src.core.coverage_analyzer import CoverageAnalyzer

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import logging
logger = logging.getLogger(__name__)

# 🚀 통합된 캐시 시스템 import
from ..utils.cache_manager import get_cache


class CacheWorkerThread(QThread):
    finished = pyqtSignal(str, object)
    
    def __init__(self, func, args, cache_key):
        super().__init__()
        self.func = func
        self.args = args
        self.cache_key = cache_key
    
    def run(self):
        try:
            result = self.func(*self.args)
            self.finished.emit(self.cache_key, result)
        except Exception as e:
            logger.error(f"캐시 작업 실패: {e}")
            self.finished.emit(self.cache_key, None)


class CacheStatusDialog(QDialog):
    def __init__(self, cache_manager, parent=None):
        super().__init__(parent)
        self.cache_manager = cache_manager
        self.setup_ui()
        self.update_display()
    
    def setup_ui(self):
        self.setWindowTitle("캐시 상태")
        self.setMinimumSize(500, 400)
        
        layout = QVBoxLayout()
        
        stats_group = QGroupBox("캐시 통계")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        items_group = QGroupBox("캐시 항목")
        items_layout = QVBoxLayout()
        
        self.items_table = QTableWidget()
        self.items_table.setColumnCount(3)
        self.items_table.setHorizontalHeaderLabels(["파일", "작업", "크기 (MB)"])
        items_layout.addWidget(self.items_table)
        
        items_group.setLayout(items_layout)
        layout.addWidget(items_group)
        
        button_layout = QHBoxLayout()
        refresh_btn = QPushButton("새로고침")
        refresh_btn.clicked.connect(self.update_display)
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(refresh_btn)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def update_display(self):
        """TransparentCache 기반으로 캐시 상태 표시"""
        try:
            cache_dir = self.cache_manager.cache_dir
            total_files = 0
            total_size_mb = 0
            
            # 캐시 디렉토리별 파일 수 계산
            cache_stats = {}
            for cache_type in ['skeleton', 'ai_prediction', 'clipping', 'processing']:
                type_dir = cache_dir / cache_type
                if type_dir.exists():
                    files = list(type_dir.glob("*.pkl"))
                    cache_stats[cache_type] = len(files)
                    total_files += len(files)
                    
                    # 크기 계산
                    for file in files:
                        try:
                            total_size_mb += file.stat().st_size / (1024 * 1024)
                        except:
                            pass
                else:
                    cache_stats[cache_type] = 0
            
            stats_text = f"""총 캐시 파일: {total_files}개
총 크기: {total_size_mb:.2f} MB
스켈레톤 캐시: {cache_stats.get('skeleton', 0)}개
AI 예측 캐시: {cache_stats.get('ai_prediction', 0)}개
클리핑 캐시: {cache_stats.get('clipping', 0)}개
처리 캐시: {cache_stats.get('processing', 0)}개
캐시 디렉토리: {cache_dir}"""
            
            self.stats_text.setText(stats_text)
            
            # 테이블 업데이트
            self.items_table.setRowCount(0)
            
            for cache_type in cache_stats:
                if cache_stats[cache_type] > 0:
                    row = self.items_table.rowCount()
                    self.items_table.insertRow(row)
                    
                    self.items_table.setItem(row, 0, QTableWidgetItem(cache_type))
                    self.items_table.setItem(row, 1, QTableWidgetItem(f"{cache_stats[cache_type]}개 파일"))
                    
                    # 타입별 크기 계산
                    type_size = 0
                    type_dir = cache_dir / cache_type
                    if type_dir.exists():
                        for file in type_dir.glob("*.pkl"):
                            try:
                                type_size += file.stat().st_size / (1024 * 1024)
                            except:
                                pass
                    
                    self.items_table.setItem(row, 2, QTableWidgetItem(f"{type_size:.2f}"))
            
            self.items_table.resizeColumnsToContents()
            
        except Exception as e:
            logger.error(f"캐시 상태 업데이트 실패: {e}")
            self.stats_text.setText(f"캐시 상태 로드 실패: {str(e)}")


class CacheSettingsDialog(QDialog):
    def __init__(self, cache_manager, parent=None):
        super().__init__(parent)
        self.cache_manager = cache_manager
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("캐시 설정")
        self.setMinimumSize(400, 200)
        
        layout = QVBoxLayout()
        
        info_group = QGroupBox("캐시 정보")
        info_layout = QVBoxLayout()
        
        info_text = f"""현재 캐시 시스템: TransparentCache
캐시 디렉토리: {self.cache_manager.cache_dir}
캐시 타입: 스켈레톤, AI예측, 클리핑, 처리

자동 기능:
- 파일 변경 시 자동 무효화
- 30일 후 오래된 캐시 자동 정리
- 중복 방지 해시 기반 키 생성"""
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("확인")
        ok_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)


class BatchProcessDialog(QDialog):
    def __init__(self, folder_path, parent=None):
        super().__init__(parent)
        self.folder_path = folder_path
        self.selected_files = []
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("배치 처리 설정")
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout()
        
        file_group = QGroupBox("처리할 파일 선택")
        file_layout = QVBoxLayout()
        
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.MultiSelection)
        
        if self.folder_path:
            for file in Path(self.folder_path).glob("*.shp"):
                self.file_list.addItem(str(file))
        
        file_layout.addWidget(self.file_list)
        
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("전체 선택")
        select_all_btn.clicked.connect(self.select_all)
        deselect_all_btn = QPushButton("전체 해제")
        deselect_all_btn.clicked.connect(self.deselect_all)
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        file_layout.addLayout(button_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        option_group = QGroupBox("처리 옵션")
        option_layout = QVBoxLayout()
        
        self.use_dqn_checkbox = QCheckBox("DQN 자동 검출 사용")
        self.use_dqn_checkbox.setChecked(True)
        option_layout.addWidget(self.use_dqn_checkbox)
        
        self.save_sessions_checkbox = QCheckBox("세션 자동 저장")
        self.save_sessions_checkbox.setChecked(True)
        option_layout.addWidget(self.save_sessions_checkbox)
        
        self.use_cache_checkbox = QCheckBox("캐시 사용")
        self.use_cache_checkbox.setChecked(True)
        option_layout.addWidget(self.use_cache_checkbox)
        
        option_group.setLayout(option_layout)
        layout.addWidget(option_group)
        
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("실행")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("취소")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def select_all(self):
        for i in range(self.file_list.count()):
            self.file_list.item(i).setSelected(True)
    
    def deselect_all(self):
        for i in range(self.file_list.count()):
            self.file_list.item(i).setSelected(False)
    
    def get_selected_files(self):
        return [item.text() for item in self.file_list.selectedItems()]


class BatchProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.results = []
    
    def setup_ui(self):
        self.setWindowTitle("배치 처리 진행 중...")
        self.setMinimumSize(600, 400)
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("준비 중...")
        layout.addWidget(self.status_label)
        
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels([
            "파일명", "상태", "교차점", "커브", "끝점"
        ])
        layout.addWidget(self.result_table)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        layout.addWidget(self.log_text)
        
        button_layout = QHBoxLayout()
        self.stop_btn = QPushButton("중지")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.close_btn = QPushButton("닫기")
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setEnabled(False)
        
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.close_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def add_result(self, file_path, result):
        row = self.result_table.rowCount()
        self.result_table.insertRow(row)
        
        self.result_table.setItem(row, 0, QTableWidgetItem(Path(file_path).name))
        
        if result['success']:
            self.result_table.setItem(row, 1, QTableWidgetItem("✓ 성공"))
            self.result_table.setItem(row, 2, QTableWidgetItem(
                str(result['metadata']['detected_intersections'])
            ))
            self.result_table.setItem(row, 3, QTableWidgetItem(
                str(result['metadata']['detected_curves'])
            ))
            self.result_table.setItem(row, 4, QTableWidgetItem(
                str(result['metadata']['detected_endpoints'])
            ))
        else:
            self.result_table.setItem(row, 1, QTableWidgetItem("✗ 실패"))
            self.result_table.setItem(row, 2, QTableWidgetItem("-"))
            self.result_table.setItem(row, 3, QTableWidgetItem("-"))
            self.result_table.setItem(row, 4, QTableWidgetItem("-"))
        
        self.result_table.resizeColumnsToContents()
    
    def add_error(self, error_msg):
        self.log_text.append(f"[오류] {error_msg}")
    
    def set_completed(self, summary):
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.status_label.setText("배치 처리 완료")
    
    def stop_processing(self):
        reply = QMessageBox.question(
            self, "확인", "배치 처리를 중지하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.parent().batch_processor:
                self.parent().batch_processor.stop()


class PerformanceMonitorDialog(QDialog):
    def __init__(self, performance_monitor, parent=None):
        super().__init__(parent)
        self.performance_monitor = performance_monitor
        self.setup_ui()
        self.update_display()
    
    def setup_ui(self):
        self.setWindowTitle("성능 모니터링")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        
        self.tab_widget = QTabWidget()
        
        summary_widget = self.create_summary_tab()
        self.tab_widget.addTab(summary_widget, "요약")
        
        training_widget = self.create_training_tab()
        self.tab_widget.addTab(training_widget, "학습 성능")
        
        inference_widget = self.create_inference_tab()
        self.tab_widget.addTab(inference_widget, "추론 성능")
        
        layout.addWidget(self.tab_widget)
        
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
    
    def create_summary_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)
        
        widget.setLayout(layout)
        return widget
    
    def create_training_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        figure = Figure(figsize=(8, 6))
        self.training_canvas = FigureCanvas(figure)
        layout.addWidget(self.training_canvas)
        
        widget.setLayout(layout)
        return widget
    
    def create_inference_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        figure = Figure(figsize=(8, 6))
        self.inference_canvas = FigureCanvas(figure)
        layout.addWidget(self.inference_canvas)
        
        widget.setLayout(layout)
        return widget
    
    def update_display(self):
        summary = self.performance_monitor.get_summary()
        summary_text = "=== 성능 요약 ===\n\n"
        
        if 'training' in summary:
            summary_text += f"[학습]\n"
            summary_text += f"- 총 에폭: {summary['training']['total_epochs']}\n"
            summary_text += f"- 최종 손실: {summary['training']['final_loss']:.4f}\n"
            summary_text += f"- 최소 손실: {summary['training']['min_loss']:.4f}\n\n"
        
        if 'inference' in summary:
            summary_text += f"[추론]\n"
            summary_text += f"- 총 추론 횟수: {summary['inference']['total_inferences']}\n"
            summary_text += f"- 평균 처리 시간: {summary['inference']['avg_time']:.3f}초\n"
            summary_text += f"- 평균 처리 속도: {summary['inference']['avg_points_per_second']:.1f} points/sec\n\n"
        
        if 'accuracy' in summary:
            summary_text += f"[정확도]\n"
            summary_text += f"- 현재 정확도: {summary['accuracy']['current_accuracy']:.2%}\n"
            summary_text += f"- 최고 정확도: {summary['accuracy']['best_accuracy']:.2%}\n\n"
        
        if 'file_processing' in summary:
            summary_text += f"[파일 처리]\n"
            summary_text += f"- 총 파일 수: {summary['file_processing']['total_files']}\n"
            summary_text += f"- 평균 처리 시간: {summary['file_processing']['avg_processing_time']:.2f}초\n"
        
        self.summary_text.setText(summary_text)
        
        self.plot_training_history()
        self.plot_inference_performance()
    
    def plot_training_history(self):
        figure = self.training_canvas.figure
        figure.clear()
        
        if not self.performance_monitor.metrics['training_history']:
            return
        
        ax = figure.add_subplot(111)
        
        history = self.performance_monitor.metrics['training_history']
        epochs = [h['epoch'] for h in history]
        losses = [h['loss'] for h in history]
        
        ax.plot(epochs, losses, 'b-', label='Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('학습 손실 추이')
        ax.legend()
        ax.grid(True)
        
        self.training_canvas.draw()
    
    def plot_inference_performance(self):
        figure = self.inference_canvas.figure
        figure.clear()
        
        if not self.performance_monitor.metrics['inference_times']:
            return
        
        ax = figure.add_subplot(111)
        
        times = self.performance_monitor.metrics['inference_times']
        indices = range(len(times))
        pps = [t['points_per_second'] for t in times]
        
        ax.plot(indices, pps, 'g-', label='Points/sec')
        ax.set_xlabel('추론 횟수')
        ax.set_ylabel('처리 속도 (points/sec)')
        ax.set_title('추론 성능 추이')
        ax.legend()
        ax.grid(True)
        
        self.inference_canvas.draw()


class BatchProcessorThread(QThread):
    progress_updated = pyqtSignal(int, str)
    file_processed = pyqtSignal(str, dict)
    batch_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, batch_processor, file_paths, use_dqn=True, save_sessions=True):
        super().__init__()
        self.batch_processor = batch_processor
        self.file_paths = file_paths
        self.use_dqn = use_dqn
        self.save_sessions = save_sessions
    
    def run(self):
        self.batch_processor.process_batch(
            self.file_paths, 
            self.use_dqn, 
            self.save_sessions
        )


# RL 훈련 스레드
class RLTrainingThread(QThread):
    progress = pyqtSignal(int, float, float)  # episode, reward, coverage
    finished = pyqtSignal(object, dict)  # agent, results
    
    def __init__(self, env, num_episodes):
        super().__init__()
        self.env = env
        self.num_episodes = num_episodes
        self.is_running = True
    
    def run(self):
        # 에이전트 생성
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        
        agent = RLDQNAgent(state_size, action_size)
        
        episode_rewards = []
        coverage_ratios = []
        
        for episode in range(self.num_episodes):
            if not self.is_running:
                break
            
            state = self.env.reset()
            episode_reward = 0
            
            while not self.env.done:
                action = agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                agent.remember(state, action, reward, next_state, done)
                
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
                
                state = next_state
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            coverage_ratios.append(info.get('coverage_ratio', 0))
            
            self.progress.emit(episode + 1, episode_reward, coverage_ratios[-1])
        
        results = {
            'episode_rewards': episode_rewards,
            'coverage_ratios': coverage_ratios,
            'final_coverage': coverage_ratios[-1] if coverage_ratios else 0
        }
        
        self.finished.emit(agent, results)
    
    def stop(self):
        self.is_running = False


# RL 결과 다이얼로그
class RLResultsDialog(QDialog):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("RL 훈련 결과")
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout()
        
        # 그래프
        figure = Figure(figsize=(8, 6))
        canvas = FigureCanvas(figure)
        
        ax1 = figure.add_subplot(121)
        ax1.plot(self.results['episode_rewards'])
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards')
        ax1.grid(True)
        
        ax2 = figure.add_subplot(122)
        ax2.plot(self.results['coverage_ratios'])
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Coverage Ratio')
        ax2.set_title('Coverage Progress')
        ax2.grid(True)
        ax2.set_ylim([0, 1])
        
        figure.tight_layout()
        layout.addWidget(canvas)
        
        # 통계
        stats_text = f"""훈련 완료!
        
에피소드: {len(self.results['episode_rewards'])}
최종 보상: {self.results['episode_rewards'][-1]:.1f}
평균 보상: {np.mean(self.results['episode_rewards']):.1f}
최종 커버리지: {self.results['final_coverage']:.1%}"""
        
        stats_label = QLabel(stats_text)
        layout.addWidget(stats_label)
        
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)


# 배치 시뮬레이션 다이얼로그
class PlacementSimulationDialog(QDialog):
    def __init__(self, env, canvas_widget, parent=None):
        super().__init__(parent)
        self.env = env
        self.canvas_widget = canvas_widget
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("기준점 배치 시뮬레이션")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        
        # 컨트롤
        control_layout = QHBoxLayout()
        
        self.step_btn = QPushButton("한 단계 진행")
        self.step_btn.clicked.connect(self.step_simulation)
        control_layout.addWidget(self.step_btn)
        
        self.auto_btn = QPushButton("자동 진행")
        self.auto_btn.setCheckable(True)
        self.auto_btn.toggled.connect(self.toggle_auto)
        control_layout.addWidget(self.auto_btn)
        
        self.reset_btn = QPushButton("초기화")
        self.reset_btn.clicked.connect(self.reset_simulation)
        control_layout.addWidget(self.reset_btn)
        
        layout.addLayout(control_layout)
        
        # 정보 표시
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        layout.addWidget(self.info_text)
        
        # 닫기 버튼
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        
        # 타이머
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_simulation)
        
        # 초기화
        self.reset_simulation()
    
    def reset_simulation(self):
        self.state = self.env.reset()
        self.update_display()
    
    def step_simulation(self):
        if self.env.done:
            self.auto_btn.setChecked(False)
            QMessageBox.information(self, "완료", "시뮬레이션이 완료되었습니다.")
            return
        
        # 랜덤 액션 (실제로는 학습된 에이전트 사용)
        action = self.env.action_space.sample()
        self.state, reward, done, info = self.env.step(action)
        
        self.update_display()
    
    def toggle_auto(self, checked):
        if checked:
            self.timer.start(500)  # 0.5초마다
        else:
            self.timer.stop()
    
    def update_display(self):
        # 캔버스 업데이트
        self.canvas_widget.canvas.points = {
            'intersection': [(x, y) for x, y in self.env.placed_points],
            'curve': [],
            'endpoint': []
        }
        self.canvas_widget.canvas.update_display()
        
        # 정보 업데이트
        coverage_info = self.env.coverage_analyzer.calculate_coverage(self.env.placed_points)
        
        info_text = f"""단계: {self.env.current_step}
배치된 점: {len(self.env.placed_points)}개
커버리지: {coverage_info['coverage_ratio']:.1%}
평균 간격: {coverage_info['avg_spacing']:.1f}m
중복률: {coverage_info['overlap_ratio']:.1%}"""
        
        self.info_text.setText(info_text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.region_files = {}
        
        # 지구계-도로망 처리
        self.district_clipper = DistrictRoadClipper()
        self.current_polygon_data = None
        self.file_mode = 'district'  # 'district' or 'road'
        self.selected_crs = 'EPSG:5186' 
        
        self.setWindowTitle("도로망 AI 분석 시스템")
        self.setGeometry(100, 100, 1200, 800)
        
        self.folder_path = None
        self.current_file = None
        
        self.cache_manager = get_cache()
        
        self.skeleton_extractor = SkeletonExtractor()
        
        self.agent = create_agent()
        self.model_path = Path("models/true_dqn_model.pth")
        self.model_path.parent.mkdir(exist_ok=True)
        
        self.batch_processor = BatchProcessor(self.skeleton_extractor, self.agent)
        self.performance_monitor = PerformanceMonitor()
        self.performance_monitor.load_metrics()
        
        self.auto_save_timer = None
        self.batch_thread = None
        self.background_workers = []
        
        self.init_ui()
        self.setup_ai_ui()
        self.setup_batch_ui()
        self.setup_cache_ui()
        
        # RL 관련 초기화 추가
        self.setup_rl_ui()
        self.rl_agent = None
        self.rl_env = None
        
        self.load_model()
        self.update_cache_status()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        self.canvas_widget = CanvasWidget(self)
        main_layout.addWidget(self.canvas_widget, 3)
        
        self.statusBar().showMessage("준비")
        
        self.cache_status_label = QLabel()
        self.statusBar().addPermanentWidget(self.cache_status_label)
        
        self.collector = DQNDataCollector()
        self.collector.connect_to_canvas(self.canvas_widget.canvas)
        
    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 파일 모드 선택
        mode_group = QGroupBox("파일 모드")
        mode_layout = QVBoxLayout()
        
        self.district_radio = QRadioButton("지구계 파일 (자동 도로망 추출)")
        self.district_radio.setChecked(True)
        self.district_radio.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.district_radio)
        
        self.road_radio = QRadioButton("도로망 파일 (직접 업로드)")
        self.road_radio.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.road_radio)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # 좌표계 선택
        crs_group = QGroupBox("좌표계")
        crs_layout = QVBoxLayout()
        
        self.crs_5186_radio = QRadioButton("EPSG:5186 (중부원점)")
        self.crs_5186_radio.setChecked(True)
        self.crs_5186_radio.toggled.connect(lambda: self.on_crs_changed('EPSG:5186'))
        crs_layout.addWidget(self.crs_5186_radio)
        
        self.crs_5187_radio = QRadioButton("EPSG:5187 (동부원점)")
        self.crs_5187_radio.toggled.connect(lambda: self.on_crs_changed('EPSG:5187'))
        crs_layout.addWidget(self.crs_5187_radio)
        
        crs_group.setLayout(crs_layout)
        layout.addWidget(crs_group)
        
        folder_btn = QPushButton("폴더 선택")
        folder_btn.clicked.connect(self.select_folder)
        layout.addWidget(folder_btn)
        
        self.folder_label = QLabel("폴더를 선택하세요")
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)
        
        layout.addWidget(QLabel("Shapefile 목록:"))
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        layout.addWidget(self.file_list)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.info_label = QLabel("파일을 선택하세요")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        # 멀티폴리곤 네비게이션
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("이전")
        self.prev_btn.clicked.connect(self.prev_polygon)
        self.prev_btn.setEnabled(False)
        self.next_btn = QPushButton("다음")
        self.next_btn.clicked.connect(self.next_polygon)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        layout.addLayout(nav_layout)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
        
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Shapefile 폴더 선택")
        if folder:
            self.folder_path = folder
            self.folder_label.setText(f"폴더: {folder}")
            self.load_shapefiles()
            
    def load_shapefiles(self):
        self.file_list.clear()
        self.region_files.clear()
        
        if not self.folder_path:
            return
        
        if self.file_mode == 'district':
            # 지구계 모드: 모든 shp 파일 (도로망 제외)
            shp_files = [f for f in Path(self.folder_path).glob("*.shp") 
                        if not f.stem.endswith("_road")]
            
            for shp_file in shp_files:
                self.file_list.addItem(shp_file.stem)
                self.region_files[shp_file.stem] = {'district': str(shp_file)}
            
            self.info_label.setText(f"{len(shp_files)}개의 지구계 파일 발견")
        else:
            # 기존 도로망 모드 로직
            region_groups = {}
            
            for shp_file in Path(self.folder_path).glob("*.shp"):
                file_name = shp_file.stem
                
                if file_name.endswith("_road"):
                    region_name = file_name[:-5]
                    if region_name not in region_groups:
                        region_groups[region_name] = {}
                    region_groups[region_name]['road'] = str(shp_file)
                else:
                    region_name = file_name
                    if region_name not in region_groups:
                        region_groups[region_name] = {}
                    region_groups[region_name]['background'] = str(shp_file)
            
            for region_name, files in region_groups.items():
                if 'road' in files:
                    self.file_list.addItem(f"{region_name}_road")
                    self.region_files[f"{region_name}_road"] = files
            
            self.info_label.setText(f"{len(self.region_files)}개의 도로망 파일 발견")

    def on_file_selected(self, item):
        region_name = item.text()
        
        if self.file_mode == 'district':
            # 지구계 모드
            if region_name in self.region_files:
                self.current_file = region_name
                self.process_district_file(self.region_files[region_name]['district'])
            else:
                QMessageBox.warning(self, "경고", f"파일을 찾을 수 없습니다: {region_name}")
        else:
            # 기존 도로망 모드
            if region_name in self.region_files:
                self.current_file = region_name
                self.process_region_files(region_name)
            else:
                QMessageBox.warning(self, "경고", f"지역 파일을 찾을 수 없습니다: {region_name}")
            
    def process_region_files(self, region_name):
        try:
            self.canvas_widget.clear_all()
            
            self.progress_bar.setValue(0)
            self.info_label.setText("처리 중...")
            
            files = self.region_files[region_name]
            
            progress = QProgressDialog("지역 파일 처리 중...", "취소", 0, 5, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            if 'background' in files:
                progress.setLabelText("배경 지구계 읽는 중...")
                progress.setValue(1)
                QApplication.processEvents()
                
                background_gdf = gpd.read_file(files['background'])
                self.canvas_widget.set_background_data(background_gdf)
            
            if 'road' in files:
                progress.setLabelText("도로망 데이터 읽는 중...")
                progress.setValue(2)
                QApplication.processEvents()
                
                road_file = files['road']
                
                # TransparentCache 사용
                skeleton_data = self.cache_manager.get(road_file, "skeleton")
                
                if skeleton_data is None:
                    progress.setLabelText("스켈레톤 추출 중...")
                    progress.setValue(3)
                    QApplication.processEvents()
                    
                    start_time = time.time()
                    # 현재 좌표계 가져오기 (기본값 EPSG:5186)
                    target_crs = getattr(self, 'selected_crs', 'EPSG:5186')
                    skeleton, intersections = self.skeleton_extractor.process_shapefile(road_file, target_crs)
                    processing_time = time.time() - start_time
                    
                    skeleton_data = {
                        'skeleton': skeleton,
                        'intersections': intersections,
                        'processing_time': processing_time
                    }
                    self.cache_manager.set(road_file, "skeleton", skeleton_data)
                else:
                    skeleton = skeleton_data['skeleton']
                    intersections = skeleton_data['intersections']
                    processing_time = skeleton_data['processing_time']
                
                road_gdf = gpd.read_file(road_file)
                self.canvas_widget.set_road_data(road_gdf)
                
                progress.setLabelText("화면 업데이트 중...")
                progress.setValue(4)
                QApplication.processEvents()
                
                self.canvas_widget.current_file = road_file
                self.canvas_widget.skeleton = skeleton
                self.canvas_widget.processing_time = processing_time
                
                self.canvas_widget.canvas.points = {
                    'intersection': [(float(x), float(y)) for x, y in intersections],
                    'curve': [],
                    'endpoint': []
                }
                
                self.canvas_widget.canvas.ai_points = {
                    'intersection': [],
                    'curve': [],
                    'endpoint': [],
                    'delete': []
                }
                
                self.collector.start_session(road_file, skeleton, intersections)
            
            self.canvas_widget.update_display()
            progress.setValue(5)
            
            info_text = f"지역: {region_name}\n"
            if 'background' in files:
                info_text += f"배경: ✓\n"
            if 'road' in files:
                info_text += f"도로망: ✓\n"
                info_text += f"스켈레톤: {len(skeleton)}점\n"
                info_text += f"교차점: {len(intersections)}개\n"
                info_text += f"처리시간: {processing_time:.2f}초"
            
            self.info_label.setText(info_text)
            self.progress_bar.setValue(100)
            
            progress.close()
            
            self.update_cache_status()
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self, "오류", f"지역 파일 처리 실패:\n{str(e)}")
            self.info_label.setText(f"오류: {str(e)}")
            import traceback
            traceback.print_exc()

    def setup_ai_ui(self):
        ai_menu = self.menuBar().addMenu("AI 학습")
        
        save_session_action = QAction("세션 저장", self)
        save_session_action.triggered.connect(self.save_current_session)
        ai_menu.addAction(save_session_action)
        
        load_session_action = QAction("세션 불러오기", self)
        load_session_action.triggered.connect(self.load_session_dialog)
        ai_menu.addAction(load_session_action)
        
        ai_menu.addSeparator()
        
        train_action = QAction("현재 데이터로 학습", self)
        train_action.triggered.connect(self.train_on_current_data)
        ai_menu.addAction(train_action)
        
        batch_train_action = QAction("모든 세션으로 학습", self)
        batch_train_action.triggered.connect(self.batch_train)
        ai_menu.addAction(batch_train_action)
        
        ai_menu.addSeparator()
        
        predict_action = QAction("DQN 자동 검출", self)
        predict_action.triggered.connect(self.run_dqn_detection)
        ai_menu.addAction(predict_action)
        
        ai_toolbar = self.addToolBar("AI Tools")
        
        self.ai_assist_btn = QPushButton("DQN 보조 OFF")
        self.ai_assist_btn.setCheckable(True)
        self.ai_assist_btn.toggled.connect(self.toggle_ai_assist)
        ai_toolbar.addWidget(self.ai_assist_btn)
        
        self.model_status_label = QLabel("모델: 미학습")
        ai_toolbar.addWidget(self.model_status_label)
        
        self.auto_save_checkbox = QCheckBox("자동 저장")
        self.auto_save_checkbox.toggled.connect(self.toggle_auto_save)
        ai_toolbar.addWidget(self.auto_save_checkbox)
    
    def save_current_session(self):
        if not hasattr(self.canvas_widget, 'current_file') or not self.canvas_widget.current_file:
            QMessageBox.warning(self, "경고", "저장할 파일이 없습니다.")
            return
        
        labels = self.canvas_widget.canvas.points
        skeleton = self.canvas_widget.skeleton
        
        metadata = {
            'total_points': len(skeleton) if skeleton is not None else 0,
            'processing_time': getattr(self.canvas_widget, 'processing_time', 0)
        }
        
        user_actions = self.collector.end_session()
        session_path = save_session(
            self.canvas_widget.current_file,
            labels,
            skeleton,
            metadata,
            user_actions
        )
        
        if session_path:
            QMessageBox.information(self, "성공", f"세션이 저장되었습니다:\n{session_path}")
            self.collector.start_session(
                self.canvas_widget.current_file,
                skeleton,
                labels.get('intersection', [])
            )
    
    def load_session_dialog(self):
        sessions = list_sessions()
        
        if not sessions:
            QMessageBox.information(self, "정보", "저장된 세션이 없습니다.")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("세션 불러오기")
        dialog.setMinimumWidth(600)
        
        layout = QVBoxLayout()
        
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["파일명", "저장 시간", "교차점", "커브", "끝점"])
        table.setRowCount(len(sessions))
        
        for i, session in enumerate(sessions):
            table.setItem(i, 0, QTableWidgetItem(Path(session['file_path']).name))
            table.setItem(i, 1, QTableWidgetItem(session['timestamp']))
            table.setItem(i, 2, QTableWidgetItem(str(session['label_counts']['intersection'])))
            table.setItem(i, 3, QTableWidgetItem(str(session['label_counts']['curve'])))
            table.setItem(i, 4, QTableWidgetItem(str(session['label_counts']['endpoint'])))
        
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        button_layout = QHBoxLayout()
        load_btn = QPushButton("불러오기")
        cancel_btn = QPushButton("취소")
        
        button_layout.addWidget(load_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        def load_selected():
            row = table.currentRow()
            if row >= 0:
                session_data = load_session(sessions[row]['path'])
                if session_data:
                    self.apply_session_data(session_data)
                    dialog.accept()
        
        load_btn.clicked.connect(load_selected)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec_()
    
    def apply_session_data(self, session_data):
        file_path = session_data['file_path']
        if Path(file_path).exists():
            self.canvas_widget.canvas.points = session_data['labels']
            self.canvas_widget.skeleton = session_data['skeleton']
            self.canvas_widget.update_display()
            
            QMessageBox.information(self, "성공", "세션을 불러왔습니다.")
        else:
            QMessageBox.warning(self, "경고", f"파일을 찾을 수 없습니다:\n{file_path}")
    
    def train_on_current_data(self):
        if not hasattr(self.canvas_widget, 'skeleton') or self.canvas_widget.skeleton is None:
            QMessageBox.warning(self, "경고", "학습할 데이터가 없습니다.")
            return
        
        current_session = {
            'skeleton': self.canvas_widget.skeleton,
            'labels': self.canvas_widget.canvas.points
        }
        
        features, labels = self.prepare_training_data([current_session])
        
        if len(features) == 0:
            QMessageBox.warning(self, "경고", "학습 데이터가 부족합니다.")
            return
        
        progress = QProgressDialog("모델 학습 중...", "취소", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        try:
            self.agent.train_on_batch(features, labels, epochs=10)
            self.agent.save(self.model_path)
            self.model_status_label.setText("모델: 학습됨")
            
            progress.setValue(100)
            QMessageBox.information(self, "성공", "모델 학습이 완료되었습니다.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"학습 중 오류 발생:\n{str(e)}")
        finally:
            progress.close()
    
    def batch_train(self):
        sessions = list_sessions()
        
        if not sessions:
            QMessageBox.warning(self, "경고", "학습할 세션이 없습니다.")
            return
        
        reply = QMessageBox.question(
            self, "확인", 
            f"{len(sessions)}개의 세션으로 학습하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        progress = QProgressDialog("세션 데이터 로드 중...", "취소", 0, len(sessions), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        all_sessions = []
        for i, session_info in enumerate(sessions):
            if progress.wasCanceled():
                return
            
            session_data = load_session(session_info['path'])
            if session_data:
                all_sessions.append(session_data)
            
            progress.setValue(i + 1)
        
        progress.setLabelText("학습 데이터 준비 중...")
        features, labels = self.prepare_training_data(all_sessions)
        
        if len(features) == 0:
            QMessageBox.warning(self, "경고", "학습 데이터가 부족합니다.")
            progress.close()
            return
        
        progress.setLabelText("모델 학습 중...")
        progress.setMaximum(100)
        
        try:
            self.agent.train_on_batch(features, labels, epochs=20)
            self.agent.save(self.model_path)
            self.model_status_label.setText("모델: 학습됨")
            
            progress.setValue(100)
            QMessageBox.information(
                self, "성공", 
                f"{len(all_sessions)}개 세션으로 학습 완료\n"
                f"총 {len(features)}개 포인트 학습"
            )
        except Exception as e:
            QMessageBox.critical(self, "오류", f"학습 중 오류 발생:\n{str(e)}")
        finally:
            progress.close()
    
    def prepare_training_data(self, sessions):
        """3-액션 시스템 학습 데이터 준비"""
        all_features = []
        all_labels = []
        
        for session in sessions:
            skeleton = session.get('skeleton', [])
            if not skeleton:
                continue
            labels = session.get('labels', {})
            deleted_points = session.get('deleted_points', {})
            
            # 사용자가 추가한 커브 점들
            curve_points = set()
            for px, py in labels.get('curve', []):
                curve_points.add((float(px), float(py)))
            
            # 사용자가 삭제한 점들
            delete_points = set()
            for category in ['intersection', 'curve', 'endpoint']:
                for px, py in deleted_points.get(category, []):
                    delete_points.add((float(px), float(py)))
            
            for i, point in enumerate(skeleton):
                if len(point) < 2:
                    continue
                features = extract_point_features(point, skeleton, i)
                x, y = float(point[0]), float(point[1])
                
                # 라벨 결정 (AI 담당 영역만)
                label = 0  # 기본: keep
                
                # 삭제된 점 확인
                min_delete_dist = min([np.sqrt((x - dx)**2 + (y - dy)**2) 
                                     for dx, dy in delete_points], default=float('inf'))
                if min_delete_dist < 5:
                    label = 2  # delete
                else:
                    # 커브 점 확인  
                    min_curve_dist = min([np.sqrt((x - cx)**2 + (y - cy)**2) 
                                        for cx, cy in curve_points], default=float('inf'))
                    if min_curve_dist < 5:
                        label = 1  # add_curve
                
                all_features.append(features)
                all_labels.append(label)
        
        return np.array(all_features), np.array(all_labels)
    
    def run_dqn_detection(self):
        if not hasattr(self.canvas_widget, 'skeleton') or self.canvas_widget.skeleton is None:
            QMessageBox.warning(self, "경고", "분석할 데이터가 없습니다.")
            return
        
        if not self.model_path.exists():
            QMessageBox.warning(self, "경고", "학습된 DQN 모델이 없습니다. 먼저 학습을 수행하세요.")
            return
        
        file_path = self.canvas_widget.current_file
        # TransparentCache 사용
        dqn_predictions = self.cache_manager.get(file_path, "dqn_predictions")
        
        if dqn_predictions is not None:
            self.canvas_widget.canvas.ai_points = dqn_predictions
            self.canvas_widget.update_display()
            
            QMessageBox.information(
                self, "DQN 검출 완료 (캐시)",
                f"검출 결과:\n"
                f"- 교차점: {len(dqn_predictions['intersection'])}개\n"
                f"- 커브: {len(dqn_predictions['curve'])}개\n"
                f"- 끝점: {len(dqn_predictions['endpoint'])}개\n"
                f"- 삭제: {len(dqn_predictions['delete'])}개"
            )
            
            self.update_cache_status()
            return
        
        progress = QProgressDialog("DQN 검출 중...", "취소", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        try:
            skeleton = self.canvas_widget.skeleton
            all_features = []
            
            for i, point in enumerate(skeleton):
                if i % 100 == 0:
                    progress.setValue(int(i / len(skeleton) * 50))
                    if progress.wasCanceled():
                        return
                
                features = extract_point_features(point, skeleton, i)
                all_features.append(features)
            
            progress.setLabelText("DQN 예측 중...")
            predictions = self.agent.predict(np.array(all_features))
            
            dqn_points = {
                'intersection': [],
                'curve': [],
                'endpoint': [],
                'delete': []
            }
            
            for i, pred in enumerate(predictions):
                point = tuple(skeleton[i])
                if pred == 1:
                    dqn_points['intersection'].append(point)
                elif pred == 2:
                    dqn_points['curve'].append(point)
                elif pred == 3:
                    dqn_points['endpoint'].append(point)
                elif pred == 4:
                    dqn_points['delete'].append(point)
            
            # TransparentCache 사용
            # TransparentCache 사용
            self.cache_manager.set(file_path, "dqn_predictions", dqn_points)
            
            self.canvas_widget.canvas.ai_points = dqn_points
            self.canvas_widget.update_display()
            
            progress.setValue(100)
            
            QMessageBox.information(
                self, "DQN 검출 완료",
                f"검출 결과:\n"
                f"- 교차점: {len(dqn_points['intersection'])}개\n"
                f"- 커브: {len(dqn_points['curve'])}개\n"
                f"- 끝점: {len(dqn_points['endpoint'])}개\n"
                f"- 삭제: {len(dqn_points['delete'])}개"
            )
            
            self.update_cache_status()
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"DQN 검출 중 오류 발생:\n{str(e)}")
        finally:
            progress.close()
    
    def toggle_ai_assist(self, checked):
        if checked:
            self.ai_assist_btn.setText("DQN 보조 ON")
            self.canvas_widget.canvas.ai_assist_mode = True
        else:
            self.ai_assist_btn.setText("DQN 보조 OFF")
            self.canvas_widget.canvas.ai_assist_mode = False
    
    def toggle_auto_save(self, checked):
        if checked:
            if not self.auto_save_timer:
                self.auto_save_timer = QTimer()
                self.auto_save_timer.timeout.connect(self.auto_save_session)
                self.auto_save_timer.start(300000)
        else:
            if self.auto_save_timer:
                self.auto_save_timer.stop()
                self.auto_save_timer = None
    
    def auto_save_session(self):
        if hasattr(self.canvas_widget, 'current_file') and self.canvas_widget.current_file:
            try:
                self.save_current_session()
            except Exception as e:
                logger.error(f"자동 저장 실패: {e}")
    
    def load_model(self):
        if self.model_path.exists():
            try:
                self.agent.load(self.model_path)
                self.model_status_label.setText("DQN 모델: 로드됨")
            except Exception as e:
                print(f"모델 로드 실패: {e}")
                self.model_status_label.setText("DQN 모델: 미학습")
    
    def setup_batch_ui(self):
        batch_menu = self.menuBar().addMenu("배치 처리")
        
        batch_process_action = QAction("배치 처리 실행", self)
        batch_process_action.triggered.connect(self.show_batch_dialog)
        batch_menu.addAction(batch_process_action)
        
        batch_menu.addSeparator()
        
        performance_action = QAction("성능 모니터링", self)
        performance_action.triggered.connect(self.show_performance_monitor)
        batch_menu.addAction(performance_action)
        
        export_metrics_action = QAction("메트릭 내보내기", self)
        export_metrics_action.triggered.connect(self.export_metrics)
        batch_menu.addAction(export_metrics_action)
    
    def setup_cache_ui(self):
        cache_menu = self.menuBar().addMenu("캐시")
        
        cache_status_action = QAction("캐시 상태", self)
        cache_status_action.triggered.connect(self.show_cache_dialog)
        cache_menu.addAction(cache_status_action)
        
        clear_cache_action = QAction("캐시 비우기", self)
        clear_cache_action.triggered.connect(self.clear_cache)
        cache_menu.addAction(clear_cache_action)
        
        cache_menu.addSeparator()
        
        cache_settings_action = QAction("캐시 설정", self)
        cache_settings_action.triggered.connect(self.show_cache_settings)
        cache_menu.addAction(cache_settings_action)
        
        self.cache_timer = QTimer()
        self.cache_timer.timeout.connect(self.update_cache_status)
        self.cache_timer.start(5000)
    
    def update_cache_status(self):
        """TransparentCache 기반 상태 업데이트"""
        try:
            cache_dir = self.cache_manager.cache_dir
            total_files = 0
            total_size_mb = 0
            
            # 캐시 파일 개수와 크기 계산
            for cache_type in ['skeleton', 'ai_prediction', 'clipping', 'processing']:
                type_dir = cache_dir / cache_type
                if type_dir.exists():
                    files = list(type_dir.glob("*.pkl"))
                    total_files += len(files)
                    
                    for file in files:
                        try:
                            total_size_mb += file.stat().st_size / (1024 * 1024)
                        except:
                            pass
            
            status_text = f"캐시: {total_files}개 파일 | {total_size_mb:.1f}MB"
            self.cache_status_label.setText(status_text)
            
        except Exception as e:
            logger.error(f"캐시 상태 업데이트 실패: {e}")
            self.cache_status_label.setText("캐시: 상태 불명")
    
    def show_cache_dialog(self):
        dialog = CacheStatusDialog(self.cache_manager, self)
        dialog.exec_()
    
    def clear_cache(self):
        reply = QMessageBox.question(
            self, "확인",
            "모든 캐시를 삭제하시겠습니까?\n" \
            "다음 작업 시 처리 시간이 길어질 수 있습니다.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # TransparentCache 디렉토리 전체 정리
                cache_dir = self.cache_manager.cache_dir
                for cache_type in ['skeleton', 'ai_prediction', 'clipping', 'processing']:
                    type_dir = cache_dir / cache_type
                    if type_dir.exists():
                        for cache_file in type_dir.glob("*.pkl"):
                            try:
                                cache_file.unlink()
                            except:
                                pass
                
                self.update_cache_status()
                QMessageBox.information(self, "완료", "캐시가 비워졌습니다.")
                logger.info("모든 캐시 파일 삭제 완료")
                
            except Exception as e:
                logger.error(f"캐시 삭제 실패: {e}")
                QMessageBox.warning(self, "오류", f"캐시 삭제 중 오류 발생: {str(e)}")
    
    def show_cache_settings(self):
        dialog = CacheSettingsDialog(self.cache_manager, self)
        dialog.exec_()  # TransparentCache는 설정 변경이 필요 없음
    
    def show_batch_dialog(self):
        dialog = BatchProcessDialog(self.folder_path, self)
        if dialog.exec_():
            selected_files = dialog.get_selected_files()
            use_dqn = dialog.use_dqn_checkbox.isChecked()
            save_sessions = dialog.save_sessions_checkbox.isChecked()
            
            if selected_files:
                self.run_batch_processing(selected_files, use_dqn, save_sessions)
    
    def run_batch_processing(self, file_paths, use_dqn, save_sessions):
        self.progress_dialog = BatchProgressDialog(self)
        self.progress_dialog.show()
        
        self.batch_processor.progress_updated.connect(self.progress_dialog.update_progress)
        self.batch_processor.file_processed.connect(self.on_file_processed)
        self.batch_processor.batch_completed.connect(self.on_batch_completed)
        self.batch_processor.error_occurred.connect(self.on_batch_error)
        
        self.batch_thread = BatchProcessorThread(
            self.batch_processor, file_paths, use_dqn, save_sessions
        )
        self.batch_thread.start()
    
    def on_file_processed(self, file_path, result):
        if result['success']:
            self.performance_monitor.log_file_processing(
                file_path,
                result['processing_time'],
                sum([
                    len(result['labels']['intersection']),
                    len(result['labels']['curve']),
                    len(result['labels']['endpoint'])
                ])
            )
            
            self.progress_dialog.add_result(file_path, result)
    
    def on_batch_completed(self, summary):
        self.progress_dialog.set_completed(summary)
        
        QMessageBox.information(
            self, "배치 처리 완료",
            f"전체: {summary['total_files']}개\n"
            f"성공: {summary['successful']}개\n"
            f"실패: {summary['failed']}개\n"
            f"소요 시간: {summary['total_time']:.1f}초"
        )
    
    def on_batch_error(self, error_msg):
        self.progress_dialog.add_error(error_msg)
    
    def show_performance_monitor(self):
        monitor_dialog = PerformanceMonitorDialog(self.performance_monitor, self)
        monitor_dialog.exec_()
    
    def export_metrics(self):
        export_dir = QFileDialog.getExistingDirectory(
            self, "메트릭 내보내기 폴더 선택"
        )
        
        if export_dir:
            try:
                self.performance_monitor.export_to_csv(export_dir)
                QMessageBox.information(
                    self, "성공", 
                    f"메트릭이 내보내졌습니다:\n{export_dir}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "오류",
                    f"메트릭 내보내기 실패:\n{str(e)}"
                )
    
    # RL 관련 메서드 추가
    def setup_rl_ui(self):
        """RL 훈련 UI 추가"""
        rl_menu = self.menuBar().addMenu("강화학습")
        
        # RL 환경 생성
        create_env_action = QAction("RL 환경 생성", self)
        create_env_action.triggered.connect(self.create_rl_environment)
        rl_menu.addAction(create_env_action)
        
        # RL 훈련
        train_rl_action = QAction("RL 훈련 시작", self)
        train_rl_action.triggered.connect(self.start_rl_training)
        rl_menu.addAction(train_rl_action)
        
        # RL 예측
        predict_rl_action = QAction("RL로 기준점 배치", self)
        predict_rl_action.triggered.connect(self.predict_with_rl)
        rl_menu.addAction(predict_rl_action)
        
        rl_menu.addSeparator()
        
        # 시뮬레이션
        simulate_action = QAction("배치 시뮬레이션", self)
        simulate_action.triggered.connect(self.simulate_placement)
        rl_menu.addAction(simulate_action)

    def create_rl_environment(self):
        """현재 지구로 RL 환경 생성"""
        if not self.current_polygon_data:
            QMessageBox.warning(self, "경고", "먼저 지구계 파일을 로드하세요.")
            return
        
        try:
            # 현재 폴리곤 데이터
            poly_info = self.current_polygon_data['polygons'][self.current_polygon_index]
            district_polygon = poly_info['geometry'].iloc[0].geometry
            road_gdf = poly_info.get('clipped_road')
            
            if road_gdf is None:
                QMessageBox.warning(self, "경고", "도로망이 없습니다.")
                return
            
            # 도로 폴리곤 추출
            road_polygons = extract_road_polygons_from_gdf(road_gdf)
            
            # 스켈레톤과 휴리스틱 교차점
            skeleton_points = self.canvas_widget.skeleton
            heuristic_intersections = [(x, y) for x, y in self.canvas_widget.canvas.points.get('intersection', [])]
            
            # 환경 생성
            env_config = {
                'max_points': 150,
                'target_distance': 50.0,
                'coverage_radius': 50.0,
                'coverage_target': 0.95
            }
            
            self.rl_env = SurveyPointEnvironment(
                district_polygon,
                road_polygons,
                skeleton_points,
                heuristic_intersections,
                env_config
            )
            
            QMessageBox.information(
                self, "성공",
                f"RL 환경 생성 완료!\n"
                f"- 후보점: {self.rl_env.num_candidates}개\n"
                f"- 초기 교차점: {len(self.rl_env.placed_points)}개"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"환경 생성 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def start_rl_training(self):
        """RL 훈련 시작"""
        if not self.rl_env:
            reply = QMessageBox.question(
                self, "확인",
                "RL 환경이 없습니다. 생성하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.create_rl_environment()
            else:
                return
        
        # 훈련 설정
        num_episodes, ok = QInputDialog.getInt(
            self, "RL 훈련", "에피소드 수:", 100, 10, 1000
        )
        
        if not ok:
            return
        
        # 훈련 스레드
        self.rl_thread = RLTrainingThread(self.rl_env, num_episodes)
        self.rl_thread.progress.connect(self.on_rl_progress)
        self.rl_thread.finished.connect(self.on_rl_finished)
        
        # 진행 다이얼로그
        self.rl_progress = QProgressDialog("RL 훈련 중...", "취소", 0, num_episodes, self)
        self.rl_progress.setWindowModality(Qt.WindowModal)
        self.rl_progress.canceled.connect(self.rl_thread.stop)
        self.rl_progress.show()
        
        self.rl_thread.start()

    def on_rl_progress(self, episode, reward, coverage):
        """RL 훈련 진행 상황"""
        self.rl_progress.setValue(episode)
        self.rl_progress.setLabelText(
            f"에피소드 {episode} - 보상: {reward:.1f}, 커버리지: {coverage:.1%}"
        )

    def on_rl_finished(self, agent, results):
        """RL 훈련 완료"""
        self.rl_progress.close()
        self.rl_agent = agent
        
        # 결과 다이얼로그
        dialog = RLResultsDialog(results, self)
        dialog.exec_()
        
        # 모델 저장
        reply = QMessageBox.question(
            self, "모델 저장",
            "훈련된 모델을 저장하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "RL 모델 저장", "models/rl_dqn/rl_model.pth", "PyTorch Models (*.pth)"
            )
            if file_path:
                self.rl_agent.save(file_path)
                QMessageBox.information(self, "성공", "모델이 저장되었습니다.")

    def predict_with_rl(self):
        """RL로 기준점 예측"""
        if not self.rl_agent:
            # 모델 로드
            file_path, _ = QFileDialog.getOpenFileName(
                self, "RL 모델 선택", "models/rl_dqn", "PyTorch Models (*.pth)"
            )
            if not file_path:
                return
            
            # 에이전트 생성 (임시)
            self.rl_agent = RLDQNAgent(1000, 1000)
            self.rl_agent.load(file_path)
        
        if not self.rl_env:
            self.create_rl_environment()
        
        # 예측 실행
        state = self.rl_env.reset()
        predicted_points = []
        
        while not self.rl_env.done:
            action = self.rl_agent.act(state, training=False)
            state, reward, done, info = self.rl_env.step(action)
            
            if 'new_point' in info:
                predicted_points.append(info['new_point'])
        
        # 캔버스에 표시
        self.canvas_widget.canvas.ai_points = {
            'intersection': self.rl_env.placed_points[:len(self.rl_env.heuristic_intersections)],
            'curve': predicted_points,
            'endpoint': [],
            'delete': []
        }
        self.canvas_widget.canvas.update_display()
        
        QMessageBox.information(
            self, "RL 예측 완료",
            f"총 {len(predicted_points)}개 기준점 추가\n"
            f"최종 커버리지: {self.rl_env._get_step_info(None)['coverage_ratio']:.1%}"
        )

    def simulate_placement(self):
        """기준점 배치 시뮬레이션"""
        if not self.rl_env:
            QMessageBox.warning(self, "경고", "먼저 RL 환경을 생성하세요.")
            return
        
        # 시뮬레이션 다이얼로그
        dialog = PlacementSimulationDialog(self.rl_env, self.canvas_widget, self)
        dialog.exec_()
    
    def on_mode_changed(self):
        self.file_mode = 'district' if self.district_radio.isChecked() else 'road'
        self.load_shapefiles()
    
    def on_crs_changed(self, crs):
        self.selected_crs = crs
    
    def process_district_file(self, district_file):
        try:
            self.canvas_widget.clear_all()
            self.progress_bar.setValue(0)
            self.info_label.setText("지구계 처리 중...")
            
            # 지구계 파일 처리
            results = self.district_clipper.process_district_file(
                district_file,
                target_crs=self.selected_crs,
                auto_find_road=True
            )
            
            if not results['success']:
                if results['error'] == "도로망 파일을 찾을 수 없음":
                    # 수동 선택 다이얼로그
                    folder = QFileDialog.getExistingDirectory(
                        self, "도로망 폴더 선택",
                        str(self.district_clipper.road_base_path)
                    )
                    if folder:
                        self.process_district_with_manual_road(results['polygons'], folder)
                else:
                    QMessageBox.critical(self, "오류", f"처리 실패: {results['error']}")
                return
            
            # 멀티폴리곤 처리
            self.current_polygon_data = results
            self.current_polygon_index = 0
            self.load_polygon(0)
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"지구계 처리 실패:\n{str(e)}")
    
    def process_district_with_manual_road(self, polygons, road_folder):
        try:
            for poly_info in polygons:
                clipped = self.district_clipper.clip_with_manual_road(
                    poly_info['geometry'],
                    road_folder,
                    self.selected_crs
                )
                poly_info['clipped_road'] = clipped
            
            self.current_polygon_data = {
                'success': True,
                'polygons': polygons,
                'total_polygons': len(polygons),
                'target_crs': self.selected_crs
            }
            self.current_polygon_index = 0
            self.load_polygon(0)
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"도로망 클리핑 실패:\n{str(e)}")
    
    def load_polygon(self, index):
        if not self.current_polygon_data or not self.current_polygon_data['polygons']:
            return
        
        poly_info = self.current_polygon_data['polygons'][index]
        
        # 지구계 표시
        self.canvas_widget.set_background_data(poly_info['geometry'])
        
        # 도로망 처리
        if poly_info.get('clipped_road') is not None:
            self.process_clipped_road(poly_info['clipped_road'])
        
        # UI 업데이트
        total = self.current_polygon_data['total_polygons']
        self.canvas_widget.set_polygon_info(index + 1, total, poly_info)
        
        # 버튼 상태
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < total - 1)
        
        self.info_label.setText(f"폴리곤 {index + 1}/{total}")
    
    def process_clipped_road(self, road_gdf):
        if road_gdf is None or road_gdf.empty:
            QMessageBox.warning(self, "경고", "클리핑된 도로망이 없습니다.")
            return
        
        try:
            # 스켈레톤 추출을 위한 임시 파일 저장
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as tmp:
                temp_path = tmp.name
            
            # GeometryCollection 처리
            from shapely.geometry import GeometryCollection
            processed_gdf = road_gdf.copy()
            
            # GeometryCollection을 개별 geometry로 분해
            if any(processed_gdf.geometry.geom_type == 'GeometryCollection'):
                new_rows = []
                for idx, row in processed_gdf.iterrows():
                    if isinstance(row.geometry, GeometryCollection):
                        for geom in row.geometry.geoms:
                            new_row = row.copy()
                            new_row['geometry'] = geom
                            new_rows.append(new_row)
                    else:
                        new_rows.append(row)
                processed_gdf = gpd.GeoDataFrame(new_rows, crs=road_gdf.crs)
            
            processed_gdf.to_file(temp_path, driver='GPKG')
            
            # 스켈레톤 추출
            # 현재 좌표계 가져오기 (기본값 EPSG:5186)
            target_crs = getattr(self, 'selected_crs', 'EPSG:5186')
            skeleton, intersections = self.skeleton_extractor.process_shapefile(temp_path, target_crs)
            
            # 임시 파일 삭제
            Path(temp_path).unlink(missing_ok=True)
            # GPKG는 단일 파일이므로 추가 확장자 처리 불필요
            
            # 캔버스 업데이트
            self.canvas_widget.set_road_data(road_gdf)
            self.canvas_widget.skeleton = skeleton
            self.canvas_widget.canvas.points = {
                'intersection': [(float(x), float(y)) for x, y in intersections],
                'curve': [],
                'endpoint': []
            }
            self.canvas_widget.update_display()
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"스켈레톤 추출 실패:\n{str(e)}")
    
    def prev_polygon(self):
        if self.current_polygon_index > 0:
            self.save_current_polygon_session()
            self.current_polygon_index -= 1
            self.load_polygon(self.current_polygon_index)
    
    def next_polygon(self):
        total = self.current_polygon_data['total_polygons']
        if self.current_polygon_index < total - 1:
            self.save_current_polygon_session()
            self.current_polygon_index += 1
            self.load_polygon(self.current_polygon_index)
    
    def save_current_polygon_session(self):
        # 현재 폴리곤의 작업 내용 저장
        if hasattr(self.canvas_widget, 'skeleton') and self.canvas_widget.skeleton:
            polygon_info = {
                'index': self.current_polygon_index + 1,
                'total': self.current_polygon_data['total_polygons']
            }
            # save_session 호출 (polygon_info 파라미터 추가됨)
            from ..utils import save_session, get_polygon_session_name
            
            base_name = Path(self.current_file).stem
            session_name = get_polygon_session_name(
                base_name,
                self.current_polygon_index + 1,
                self.current_polygon_data['total_polygons']
            )
            
            save_session(
                session_name,
                self.canvas_widget.canvas.points,
                self.canvas_widget.skeleton,
                polygon_info=polygon_info
            )

    def closeEvent(self, event):
        if self.collector.current_session:
            self.collector.end_session()
        self.cache_manager.save_metadata()
        event.accept()


def main():
    import sys
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()