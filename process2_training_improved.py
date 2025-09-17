"""
프로세스 2: 끝점만 AI 학습
- 끝점만 2클래스(DQN) 학습
- 커브/교차점은 휴리스틱
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QProgressBar, QMessageBox,
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QSplitter, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

sys.path.append(str(Path(__file__).parent.parent))
from src.learning.dqn_model import DQNAgent, create_agent
from src.utils import load_session, list_sessions, extract_point_features
from src.config import *

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== 상세 디버깅 코드 ==========
import time
import threading

def debug_print_with_time(msg):
    """시간 포함 디버깅 출력"""
    timestamp = time.strftime("%H:%M:%S")
    thread_id = threading.current_thread().ident
    print(f"[{timestamp}] [T-{thread_id}] {msg}")
    sys.stdout.flush()

def debug_timer(func_name):
    """함수 실행 시간 측정 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            debug_print_with_time(f"시작: {func_name}")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                debug_print_with_time(f"완료: {func_name} ({elapsed:.2f}초)")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                debug_print_with_time(f"오류: {func_name} ({elapsed:.2f}초) - {e}")
                raise
        return wrapper
    return decorator

# 진행상황 표시 함수
def show_progress(current, total, operation):
    """진행상황 표시"""
    if total > 0:
        percent = (current / total) * 100
        debug_print_with_time(f"{operation}: {current}/{total} ({percent:.1f}%)")
    else:
        debug_print_with_time(f"{operation}: {current}")

debug_print_with_time("=== 상세 디버깅 모드 활성화 ===")
# ======================================



# ========== 디버깅 강화 코드 ==========
import sys
import traceback

def debug_print(msg):
    """디버깅 출력 (콘솔 + UI)"""
    print(f"[DEBUG] {msg}")
    sys.stdout.flush()

def safe_execute(func, *args, **kwargs):
    """안전한 함수 실행 with 에러 캐치"""
    try:
        debug_print(f"실행 중: {func.__name__}")
        result = func(*args, **kwargs)
        debug_print(f"완료: {func.__name__}")
        return result
    except Exception as e:
        debug_print(f"오류 in {func.__name__}: {e}")
        traceback.print_exc()
        raise

# 콘솔 로깅 설정 강화
import logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

debug_print("=== Process2 Training 시작 ===")
# ======================================



def prepare_training_data(sessions):
    """
    세션 데이터에서 AI 학습용 데이터 준비
    """
    all_features = []
    all_labels = []
    
    for session in sessions:
        skeleton = session.get('skeleton', [])
        labels_dict = session.get('labels', {})
        deleted_points = session.get('deleted_points', {})
        
        if not skeleton or len(skeleton) == 0:
            continue
            
        skeleton_array = np.array(skeleton)
        
        for i, point in enumerate(skeleton_array):
            window_points = []
            window_size = 50
            start_idx = max(0, i - window_size)
            end_idx = min(len(skeleton_array), i + window_size + 1)
            
            for j in range(start_idx, end_idx):
                window_points.append(skeleton_array[j].tolist())
            
            try:
                features = extract_point_features(point.tolist(), window_points, skeleton)
                
                label = 0  # 기본값: 일반
                min_distance = float('inf')
                
                # 삭제된 포인트 확인 (최우선)
                for category in ['curve', 'endpoint', 'intersection']:
                    for deleted_point in deleted_points.get(category, []):
                        dist = np.linalg.norm(np.array(point) - np.array(deleted_point))
                        if dist < min_distance and dist < 10:
                            min_distance = dist
                            label = 3  # 삭제
                
                # 라벨된 포인트 확인
                if label == 0:
                    # 커브 포인트 확인
                    for curve_point in labels_dict.get('curve', []):
                        dist = np.linalg.norm(np.array(point) - np.array(curve_point))
                        if dist < min_distance and dist < 10:
                            min_distance = dist
                            label = 1  # 커브
                    
                    # 끝점 확인
                    for endpoint in labels_dict.get('endpoint', []):
                        dist = np.linalg.norm(np.array(point) - np.array(endpoint))
                        if dist < min_distance and dist < 10:
                            min_distance = dist
                            label = 2  # 끝점
                
                all_features.append(features)
                all_labels.append(label)
                
            except Exception as e:
                print(f"특징 추출 실패: {e}")
                continue
    
    if len(all_features) == 0:
        print("경고: 학습 데이터가 없습니다!")
        return np.array([]).reshape(0, 7), np.array([])
    
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    
    print(f"학습 데이터 준비 완료: {len(features_array)}개 샘플")
    print(f"클래스 분포 - 일반: {np.sum(labels_array==0)}, 커브: {np.sum(labels_array==1)}, 끝점: {np.sum(labels_array==2)}, 삭제: {np.sum(labels_array==3)}")
    
    return features_array, labels_array




def get_optimized_config():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    return {
        'epochs': 20,
        'batch_size': 256,
        'learning_rate': 0.003,
        'hidden_sizes': [128, 64],
    }

class TrainingWorker(QThread):
    progress = pyqtSignal(int, str)
    epoch_completed = pyqtSignal(int, float, float)
    training_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, sessions, config):
        super().__init__()
        self.sessions = sessions
        self.config = config
        self.is_running = True
        self.agent = None
    

    def run(self):
        try:
            # 1. 데이터 준비
            self.log_message.emit("학습 데이터 준비 중...")
            self.progress.emit(10, "데이터 준비 중...")

            # 모든 편집 동작을 학습
            features, labels = prepare_training_data(self.sessions)
            
            if len(features) == 0:
                raise ValueError("학습 데이터가 없습니다.")
            
            # ========== 🚀 최적화 1: 데이터 샘플링 ==========
            total_samples = len(features)
            if total_samples > 10000:  # 10,000개 이상이면 샘플링
                import random
                random.seed(42)  # 재현 가능한 결과
                indices = random.sample(range(total_samples), 10000)
                features = features[indices]
                labels = labels[indices]
                self.log_message.emit(f"데이터 샘플링: {total_samples} -> {len(features)}개")
            
            # ========== 🚀 최적화 2: 데이터 타입 변환 ==========
            import torch
            features = torch.FloatTensor(features)
            labels = torch.LongTensor(labels)
            
            self.log_message.emit(f"총 {len(features)}개의 학습 데이터 준비 완료")
            
            # 클래스 분포 확인
            unique, counts = torch.unique(labels, return_counts=True)
            class_dist = {int(u): int(c) for u, c in zip(unique, counts)}
            self.log_message.emit(f"클래스 분포: {class_dist}")
            
            # 데이터 분할 (80:20)
            split_idx = int(len(features) * 0.8)
            train_features = features[:split_idx]
            train_labels = labels[:split_idx]
            val_features = features[split_idx:]
            val_labels = labels[split_idx:]
            
            # 2. 에이전트 생성 (4클래스)
            self.progress.emit(20, "모델 초기화 중...")
            self.agent = create_agent({
                'lr': self.config['learning_rate'],
                'batch_size': self.config['batch_size'],
                'hidden_sizes': self.config['hidden_sizes'],
                'gamma': self.config['gamma'],
                'epsilon_decay': self.config['epsilon_decay'],
                'action_size': 4  # 4클래스(일반/커브/끝점/삭제)
            })
            
            # 기존 모델 로드 (있다면)
            if self.config['load_pretrained'] and Path(self.config['model_path']).exists():
                self.agent.load(self.config['model_path'])
                self.log_message.emit("기존 모델 로드 완료")
            
            self.progress.emit(30, "학습 시작...")
            start_time = time.time()
            epochs = self.config['epochs']
            best_val_accuracy = 0
            
            # ========== 🚀 최적화 3: 배치 처리 개선 ==========
            batch_size = self.config['batch_size']
            device = self.agent.device
            
            for epoch in range(epochs):
                if not self.is_running:
                    break
                
                epoch_start = time.time()
                self.agent.q_network.train()
                
                # 미니배치 처리
                n_batches = len(train_features) // batch_size
                epoch_loss = 0
                
                for batch_idx in range(min(n_batches, 20)):  # 최대 20 배치만 처리
                    if not self.is_running:
                        break
                    
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size
                    
                    batch_features = train_features[start_idx:end_idx].to(device)
                    batch_labels = train_labels[start_idx:end_idx].to(device)
                    
                    # ========== 🚀 최적화 4: 벡터화된 경험 저장 ==========
                    for i in range(len(batch_features)):
                        feat = batch_features[i].cpu().numpy()
                        label = int(batch_labels[i])
                        reward_map = {0: 0.1, 1: 0.5, 2: 1.0, 3: 1.5}
                        reward = reward_map.get(label, 0.1)
                        self.agent.remember(feat, label, reward, feat, False)
                    
                    # 경험 재생 (더 자주 실행)
                    if len(self.agent.memory) >= batch_size:
                        loss = self.agent.replay()
                        if loss is not None:
                            epoch_loss += loss
                
                # ========== 🚀 최적화 5: 빠른 검증 ==========
                self.agent.q_network.eval()
                with torch.no_grad():
                    # 검증 데이터 샘플링 (빠른 평가)
                    val_sample_size = min(1000, len(val_features))
                    val_indices = torch.randperm(len(val_features))[:val_sample_size]
                    val_sample_features = val_features[val_indices].to(device)
                    val_sample_labels = val_labels[val_indices]
                    
                    val_predictions = self.agent.predict(val_sample_features.cpu().numpy())
                    val_accuracy = np.mean(val_predictions == val_sample_labels.numpy())
                
                epoch_time = time.time() - epoch_start
                progress = 30 + int((epoch + 1) / epochs * 60)
                
                self.progress.emit(progress, f"Epoch {epoch+1}/{epochs} - Acc: {val_accuracy:.3f} ({epoch_time:.1f}s)")
                self.epoch_completed.emit(epoch+1, epoch_loss, val_accuracy)
                
                # ========== 🚀 최적화 6: 선택적 모델 저장 ==========
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    if self.config['save_best'] and (epoch + 1) % 5 == 0:  # 5 에포크마다만 저장
                        self.agent.save(self.config['model_path'])
                        self.log_message.emit(f"최고 모델 저장 (Epoch {epoch+1}, Acc: {val_accuracy:.3f})")
                
                # 조기 종료 조건
                if val_accuracy > 0.95:  # 95% 정확도 달성시 조기 종료
                    self.log_message.emit(f"목표 정확도 달성! 조기 종료 (Epoch {epoch+1})")
                    break
            
            # 4. 최종 평가
            self.progress.emit(90, "최종 평가 중...")
            
            # 전체 데이터 평가 (샘플링)
            eval_sample_size = min(5000, len(features))
            eval_indices = torch.randperm(len(features))[:eval_sample_size]
            eval_features = features[eval_indices]
            eval_labels = labels[eval_indices]
            
            all_predictions = self.agent.predict(eval_features.numpy())
            final_accuracy = np.mean(all_predictions == eval_labels.numpy())
            
            # 클래스별 정확도 계산
            class_accuracies = {}
            for class_id in range(4):  # 0: 일반, 1: 커브, 2: 끝점, 3: 삭제
                mask = eval_labels == class_id
                if torch.sum(mask) > 0:
                    class_acc = np.mean(all_predictions[mask] == eval_labels[mask].numpy())
                    class_accuracies[class_id] = class_acc

            training_time = time.time() - start_time
            results = {
                'success': True,
                'final_accuracy': final_accuracy,
                'best_val_accuracy': best_val_accuracy,
                'class_accuracies': class_accuracies,
                'training_time': training_time,
                'total_samples': len(features),
                'epochs_completed': epoch + 1
            }
            self.progress.emit(100, "학습 완료!")
            self.training_completed.emit(results)
            
        except Exception as e:
            logger.error(f"학습 오류: {e}")
            self.error_occurred.emit(str(e))

    def stop(self):
        self.is_running = False

class TrainingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sessions = []
        self.training_worker = None
        self.training_history = []
        self.init_ui()
        self.load_sessions()
    
    def init_ui(self):
        self.setWindowTitle("도로망 AI 학습 도구 - 프로세스 2")
        self.setGeometry(100, 100, 1200, 800)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        self.statusBar().showMessage("준비")
    
    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        session_group = QGroupBox("1. 학습 데이터 선택")
        session_layout = QVBoxLayout()
        self.session_list = QListWidget()
        self.session_list.setSelectionMode(QListWidget.MultiSelection)
        session_layout.addWidget(self.session_list)
        self.session_info_label = QLabel("0개 선택됨")
        session_layout.addWidget(self.session_info_label)
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("전체 선택")
        select_all_btn.clicked.connect(self.select_all_sessions)
        button_layout.addWidget(select_all_btn)
        refresh_btn = QPushButton("새로고침")
        refresh_btn.clicked.connect(self.load_sessions)
        button_layout.addWidget(refresh_btn)
        session_layout.addLayout(button_layout)
        session_group.setLayout(session_layout)
        layout.addWidget(session_group)
        config_group = QGroupBox("2. 학습 설정")
        config_layout = QVBoxLayout()
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("에폭 수:"))
        self.epoch_spinbox = QSpinBox()
        self.epoch_spinbox.setRange(1, 1000)
        self.epoch_spinbox.setValue(20)
        epoch_layout.addWidget(self.epoch_spinbox)
        config_layout.addLayout(epoch_layout)
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("학습률:"))
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.0001, 0.1)
        self.lr_spinbox.setSingleStep(0.0001)
        self.lr_spinbox.setValue(0.003)
        self.lr_spinbox.setDecimals(4)
        lr_layout.addWidget(self.lr_spinbox)
        config_layout.addLayout(lr_layout)
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("배치 크기:"))
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(16, 256)
        self.batch_spinbox.setSingleStep(16)
        self.batch_spinbox.setValue(256)
        batch_layout.addWidget(self.batch_spinbox)
        config_layout.addLayout(batch_layout)
        network_layout = QHBoxLayout()
        network_layout.addWidget(QLabel("네트워크:"))
        self.network_combo = QComboBox()
        self.network_combo.addItems([
            "Small (128, 64)",
            "Medium (256, 128, 64)",
            "Large (512, 256, 128, 64)"
        ])
        self.network_combo.setCurrentIndex(1)
        network_layout.addWidget(self.network_combo)
        config_layout.addLayout(network_layout)
        self.load_pretrained_checkbox = QCheckBox("기존 모델에서 시작")
        config_layout.addWidget(self.load_pretrained_checkbox)
        self.save_best_checkbox = QCheckBox("최고 모델 자동 저장")
        self.save_best_checkbox.setChecked(True)
        config_layout.addWidget(self.save_best_checkbox)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        control_group = QGroupBox("3. 학습 제어")
        control_layout = QVBoxLayout()
        self.start_btn = QPushButton("학습 시작")
        self.start_btn.clicked.connect(self.start_training)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
            }
        """)
        control_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("학습 중지")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("대기 중...")
        control_layout.addWidget(self.status_label)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        layout.addStretch()
        return panel
    
    def create_right_panel(self):
        tab_widget = QTabWidget()
        monitor_tab = self.create_monitor_tab()
        tab_widget.addTab(monitor_tab, "학습 모니터")
        log_tab = self.create_log_tab()
        tab_widget.addTab(log_tab, "로그")
        result_tab = self.create_result_tab()
        tab_widget.addTab(result_tab, "결과")
        return tab_widget
    
    def create_monitor_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        self.figure = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        stats_group = QGroupBox("실시간 통계")
        stats_layout = QVBoxLayout()
        self.stats_label = QLabel("학습 전...")
        self.stats_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #f0f0f0;
                font-family: monospace;
            }
        """)
        stats_layout.addWidget(self.stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        return widget
    
    def create_log_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        control_layout = QHBoxLayout()
        clear_btn = QPushButton("로그 지우기")
        clear_btn.clicked.connect(self.log_text.clear)
        control_layout.addWidget(clear_btn)
        save_log_btn = QPushButton("로그 저장")
        save_log_btn.clicked.connect(self.save_log)
        control_layout.addWidget(save_log_btn)
        control_layout.addStretch()
        layout.addLayout(control_layout)
        return widget
    
    def create_result_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["항목", "값"])
        layout.addWidget(self.result_table)
        save_model_btn = QPushButton("학습된 모델 저장")
        save_model_btn.clicked.connect(self.save_model)
        layout.addWidget(save_model_btn)
        return widget
    
    def load_sessions(self):
        self.session_list.clear()
        sessions = list_sessions()
        for session in sessions:
            # 어떤 편집이든 있으면 표시
            has_edits = (session['label_counts'].get('curve', 0) > 0 or 
                        session['label_counts'].get('endpoint', 0) > 0 or
                        session.get('deleted_count', 0) > 0)
            if has_edits:
                filename = Path(session['file_path']).name
                counts = session['label_counts']
                item_text = f"{filename} - 교:{counts['intersection']} 커:{counts['curve']} 끝:{counts['endpoint']}"
                self.session_list.addItem(item_text)
                self.session_list.item(self.session_list.count() - 1).setData(
                    Qt.UserRole, session['path']
                )
        self.update_session_info()
    
    def select_all_sessions(self):
        for i in range(self.session_list.count()):
            self.session_list.item(i).setSelected(True)
        self.update_session_info()
    
    def update_session_info(self):
        selected_count = len(self.session_list.selectedItems())
        self.session_info_label.setText(f"{selected_count}개 선택됨")
        if selected_count > 0:
            total_points = 0
            for item in self.session_list.selectedItems():
                session_path = item.data(Qt.UserRole)
                session = load_session(session_path)
                if session:
                    for points in session['labels'].values():
                        total_points += len(points)
            self.session_info_label.setText(
                f"{selected_count}개 선택됨 (총 {total_points:,}개 라벨)"
            )
    
    def get_network_config(self):
        network_map = {
            "Small (128, 64)": [128, 64],
            "Medium (256, 128, 64)": [256, 128, 64],
            "Large (512, 256, 128, 64)": [512, 256, 128, 64]
        }
        return network_map[self.network_combo.currentText()]
    
    def start_training(self):
        selected_items = self.session_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "경고", "학습할 세션을 선택하세요.")
            return
        self.sessions = []
        self.log_text.append(f"=== 학습 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        for item in selected_items:
            session_path = item.data(Qt.UserRole)
            session = load_session(session_path)
            if session:
                self.sessions.append(session)
                self.log_text.append(f"세션 로드: {Path(session_path).name}")
        opt_config = get_optimized_config() if 'get_optimized_config' in globals() else {}
        config = {
            'epochs': min(self.epoch_spinbox.value(), 30),
            'learning_rate': self.lr_spinbox.value(),
            'batch_size': max(self.batch_spinbox.value(), 256),
            'hidden_sizes': opt_config.get('hidden_sizes', self.get_network_config()),
            'gamma': 0.99,
            'epsilon_decay': 0.995,
            'load_pretrained': self.load_pretrained_checkbox.isChecked(),
            'save_best': self.save_best_checkbox.isChecked(),
            'model_path': 'models/road_detector_v2.pth'
        }
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.training_history = []
        self.training_worker = TrainingWorker(self.sessions, config)
        self.training_worker.progress.connect(self.on_progress_update)
        self.training_worker.epoch_completed.connect(self.on_epoch_completed)
        self.training_worker.training_completed.connect(self.on_training_completed)
        self.training_worker.error_occurred.connect(self.on_error)
        self.training_worker.log_message.connect(self.log_text.append)
        self.training_worker.start()
    
    def stop_training(self):
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.stop()
            self.training_worker.wait()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("학습 중지됨")
        self.log_text.append("[중지] 사용자에 의해 학습이 중지되었습니다.")
    
    def on_progress_update(self, progress, message):
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def on_epoch_completed(self, epoch, loss, accuracy):
        self.training_history.append({
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy
        })
        self.update_training_plot()
        self.update_stats()
    
    def update_training_plot(self):
        if not self.training_history:
            return
        self.figure.clear()
        epochs = [h['epoch'] for h in self.training_history]
        accuracies = [h['accuracy'] for h in self.training_history]
        ax = self.figure.add_subplot(111)
        ax.plot(epochs, accuracies, 'b-', linewidth=2, label='Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.canvas.draw()
    
    def update_stats(self):
        if not self.training_history:
            return
        latest = self.training_history[-1]
        best_accuracy = max(h['accuracy'] for h in self.training_history)
        stats_text = f"""현재 에폭: {latest['epoch']}
현재 정확도: {latest['accuracy']:.4f}
최고 정확도: {best_accuracy:.4f}
학습률: {self.lr_spinbox.value():.4f}
배치 크기: {self.batch_spinbox.value()}"""
        self.stats_label.setText(stats_text)
    
    def on_training_completed(self, results):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append(f"\n=== 학습 완료 ===")
        self.log_text.append(f"최종 정확도: {results['final_accuracy']:.4f}")
        self.log_text.append(f"학습 시간: {results['training_time']:.1f}초")
        self.update_result_table(results)
        QMessageBox.information(
            self, "학습 완료",
            f"모델 학습이 완료되었습니다.\n\n"
            f"최종 정확도: {results['final_accuracy']:.4f}\n"
            f"최고 검증 정확도: {results['best_val_accuracy']:.4f}\n"
            f"학습 시간: {results['training_time']:.1f}초"
        )
    
    def update_result_table(self, results):
        self.result_table.setRowCount(0)
        items = [
            ("최종 정확도", f"{results['final_accuracy']:.4f}"),
            ("최고 검증 정확도", f"{results['best_val_accuracy']:.4f}"),
            ("학습 시간", f"{results['training_time']:.1f}초"),
            ("총 샘플 수", f"{results['total_samples']:,}"),
            ("완료 에폭", f"{results['epochs_completed']}")
        ]
        # 클래스명 정의
        class_names = ['일반', '커브', '끝점', '삭제']
        for class_id, accuracy in results['class_accuracies'].items():
            items.append((f"{class_names[class_id]} 정확도", f"{accuracy:.4f}"))
        for item_name, value in items:
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 0, QTableWidgetItem(item_name))
            self.result_table.setItem(row, 1, QTableWidgetItem(value))
    
    def on_error(self, error_msg):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append(f"[오류] {error_msg}")
        QMessageBox.critical(self, "학습 오류", f"학습 중 오류가 발생했습니다:\n{error_msg}")
    
    def save_model(self):
        if not hasattr(self, 'training_worker') or not self.training_worker:
            QMessageBox.warning(self, "경고", "저장할 모델이 없습니다.")
            return
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "모델 저장", "models/", "PyTorch Model (*.pth)"
        )
        if file_path:
            try:
                self.training_worker.agent.save(file_path)
                QMessageBox.information(self, "성공", f"모델이 저장되었습니다:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"모델 저장 실패:\n{str(e)}")
    
    def save_log(self):
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "로그 저장", "logs/", "Text Files (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "성공", "로그가 저장되었습니다.")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"로그 저장 실패:\n{str(e)}")

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = TrainingTool()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
