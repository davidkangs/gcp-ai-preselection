"""
프로세스 3 - 재학습 다이얼로그
20차원 통일 버전
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QTextEdit, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

sys.path.append(str(Path(__file__).parent.parent))
from src.learning.dqn_model import create_agent
from src.utils import load_session

import logging
logger = logging.getLogger(__name__)


class RetrainingWorker(QThread):
    progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str)
    training_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, session_paths, base_model_path, epochs, save_path):
        super().__init__()
        self.session_paths = session_paths
        self.base_model_path = base_model_path
        self.epochs = epochs
        self.save_path = save_path
    
    def _prepare_user_actions_data(self, sessions):
        all_features = []
        all_labels = []
        
        for session in sessions:
            skeleton = session.get('skeleton', [])
            user_actions = session.get('user_actions', [])
        samples = session.get(\'samples\', [])  # DQN 샘플
            heuristic_results = session.get('metadata', {}).get('heuristic_results', {})
            
            if not skeleton:
                continue
            
        
        # DQN 샘플 처리
        if samples:
            for sample in samples:
                state = sample.get('state_vector', [])
                action = sample.get('action', 0)
                if len(state) == 20 and 0 <= action <= 4:
                    all_features.append(state)
                    all_labels.append(action)
        
            if user_actions:
                for action_data in sorted(user_actions, key=lambda x: x.get('timestamp', 0)):
                    action_type = action_data.get('action')
                    category = action_data.get('category')
                    position = action_data.get('position', [0, 0])
                    
                    if len(position) < 2:
                        continue
                    
                    x, y = float(position[0]), float(position[1])
                    skeleton_idx = self._find_nearest_skeleton_point(x, y, skeleton)
                    state_vector = self._create_state_vector_for_training(x, y, skeleton, skeleton_idx, heuristic_results)
                    
                                    
                if isinstance(action_type, int):
                    label = action_type
                else:
                    label_mapping = {
                        ('add', 'intersection'): 1,
                        ('add', 'curve'): 2,
                        ('add', 'endpoint'): 3,
                        ('remove', 'intersection'): 0,
                        ('remove', 'curve'): 0,
                        ('remove', 'endpoint'): 0
                    }
                    
                    label = label_mapping.get((action_type, category), 0)
                    all_features.append(state_vector)
                    all_labels.append(label)
            else:
                labels = session.get('labels', {})
                for category, points in labels.items():
                    category_label = {'intersection': 1, 'curve': 2, 'endpoint': 3}.get(category, 0)
                    
                    for point in points:
                        if len(point) >= 2:
                            x, y = float(point[0]), float(point[1])
                            skeleton_idx = self._find_nearest_skeleton_point(x, y, skeleton)
                            state_vector = self._create_state_vector_for_training(x, y, skeleton, skeleton_idx, heuristic_results)
                            
                            all_features.append(state_vector)
                            all_labels.append(category_label)
        
        return np.array(all_features), np.array(all_labels)
    
    def _find_nearest_skeleton_point(self, x, y, skeleton):
        if not skeleton:
            return 0
        
        distances = [((x - p[0])**2 + (y - p[1])**2)**0.5 for p in skeleton if len(p) >= 2]
        return np.argmin(distances) if distances else 0
    
    def _create_state_vector_for_training(self, x, y, skeleton, idx, heuristic_results=None):
        features = [x, y]
        
        if idx > 0 and len(skeleton[idx-1]) >= 2:
            prev_x, prev_y = skeleton[idx-1][0], skeleton[idx-1][1]
            dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            angle = np.arctan2(y - prev_y, x - prev_x)
            features.extend([dist, angle])
        else:
            features.extend([0.0, 0.0])
        
        if idx < len(skeleton) - 1 and len(skeleton[idx+1]) >= 2:
            next_x, next_y = skeleton[idx+1][0], skeleton[idx+1][1]
            dist = np.sqrt((next_x - x)**2 + (next_y - y)**2)
            angle = np.arctan2(next_y - y, next_x - x)
            features.extend([dist, angle])
        else:
            features.extend([0.0, 0.0])
        
        density = sum(1 for p in skeleton if len(p) >= 2 and np.sqrt((x-p[0])**2 + (y-p[1])**2) <= 50)
        features.append(density / len(skeleton) if skeleton else 0.0)
        
        if idx > 0 and idx < len(skeleton) - 1:
            try:
                if len(skeleton[idx-1]) >= 2 and len(skeleton[idx+1]) >= 2:
                    p1 = np.array(skeleton[idx-1][:2])
                    p2 = np.array([x, y])
                    p3 = np.array(skeleton[idx+1][:2])
                    v1 = p2 - p1
                    v2 = p3 - p2
                    angle1 = np.arctan2(v1[1], v1[0])
                    angle2 = np.arctan2(v2[1], v2[0])
                    curvature = abs(angle2 - angle1)
                    if curvature > np.pi:
                        curvature = 2 * np.pi - curvature
                    features.append(curvature)
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        if heuristic_results:
            heuristic_class = self._get_heuristic_class(x, y, heuristic_results)
            heuristic_onehot = [0, 0, 0, 0]
            heuristic_onehot[heuristic_class] = 1
            features.extend(heuristic_onehot)
            
            confidence = 0.9 if heuristic_class > 0 else 0.1
            features.append(confidence)
            
            nearby_counts = []
            for cat in ['intersection', 'curve', 'endpoint']:
                count = sum(1 for px, py in heuristic_results.get(cat, []) 
                          if np.sqrt((x - px)**2 + (y - py)**2) <= 50)
                nearby_counts.append(float(count))
            features.extend(nearby_counts)
            
            min_dist = 1000.0
            for cat in ['intersection', 'curve', 'endpoint']:
                for px, py in heuristic_results.get(cat, []):
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    min_dist = min(min_dist, dist)
            features.append(min_dist)
            
            total_count = sum(nearby_counts)
            density = total_count / (np.pi * 50 * 50) * 1000 if total_count > 0 else 0
            features.append(density)
        else:
            features.extend([0] * 12)
        
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def _get_heuristic_class(self, x, y, heuristic_results):
        threshold = 5.0
        for i, cat in enumerate(['intersection', 'curve', 'endpoint']):
            for px, py in heuristic_results.get(cat, []):
                if np.sqrt((x - px)**2 + (y - py)**2) < threshold:
                    return i + 1
        return 0
    
    def run(self):
        try:
            self.progress.emit(10, "수정된 세션 로드 중...")
            sessions = []
            
            for session_path in self.session_paths:
                session = load_session(session_path)
                if session:
                    sessions.append(session)
                    self.log_message.emit(f"세션 로드: {Path(session_path).name}")
            
            self.progress.emit(30, "학습 데이터 준비 중...")
            features, labels = self._prepare_user_actions_data(sessions)
            self.log_message.emit(f"총 {len(features)}개의 학습 데이터 준비 완료")
            
            self.progress.emit(40, "기존 모델 로드 중...")
            agent = create_agent()
            agent.load(self.base_model_path)
            self.log_message.emit("기존 모델 로드 완료")
            
            self.progress.emit(50, "재학습 시작...")
            
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = 0.0001
            
            split_idx = int(len(features) * 0.8)
            train_features = features[:split_idx]
            train_labels = labels[:split_idx]
            val_features = features[split_idx:]
            val_labels = labels[split_idx:]
            
            start_time = time.time()
            best_accuracy = 0
            
            for epoch in range(self.epochs):
                agent.q_network.train()
                
                for feat, label in zip(train_features, train_labels):
                    reward = 1.0 if label > 0 else 0.1
                    agent.remember(feat, label, reward, feat, False)
                
                if len(agent.memory) >= agent.batch_size:
                    for _ in range(10):
                        agent.replay()
                
                agent.q_network.eval()
                val_predictions = agent.predict(val_features)
                val_accuracy = np.mean(val_predictions == val_labels)
                
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                
                progress = 50 + int((epoch + 1) / self.epochs * 40)
                self.progress.emit(progress, f"Epoch {epoch + 1}/{self.epochs} - Accuracy: {val_accuracy:.3f}")
                self.log_message.emit(f"Epoch {epoch + 1}: Accuracy = {val_accuracy:.3f}")
            
            self.progress.emit(95, "모델 저장 중...")
            agent.save(self.save_path)
            
            training_time = time.time() - start_time
            
            results = {
                'success': True,
                'sessions_used': len(sessions),
                'total_samples': len(features),
                'best_accuracy': best_accuracy,
                'training_time': training_time,
                'model_path': self.save_path
            }
            
            self.progress.emit(100, "재학습 완료!")
            self.training_completed.emit(results)
            
        except Exception as e:
            logger.error(f"재학습 오류: {e}")
            self.error_occurred.emit(str(e))


class RetrainingDialog(QDialog):
    def __init__(self, session_paths, base_model_path, epochs, parent=None):
        super().__init__(parent)
        self.session_paths = session_paths
        self.base_model_path = base_model_path
        self.epochs = epochs
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("모델 재학습")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        info_group = QGroupBox("재학습 정보")
        info_layout = QVBoxLayout()
        
        info_text = f"""기본 모델: {Path(self.base_model_path).name}
수정된 세션: {len(self.session_paths)}개
추가 에폭: {self.epochs}
학습 방식: Fine-tuning (미세 조정)"""
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("QLabel { padding: 10px; }")
        info_layout.addWidget(info_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        progress_group = QGroupBox("진행 상황")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("준비 중...")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        log_group = QGroupBox("학습 로그")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("재학습 시작")
        self.start_btn.clicked.connect(self.start_retraining)
        button_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def start_retraining(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"models/road_detector_retrained_{timestamp}.pth"
        
        self.start_btn.setEnabled(False)
        self.cancel_btn.setText("닫기")
        
        self.log_text.append(f"=== 재학습 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        self.log_text.append(f"저장 경로: {save_path}")
        
        self.worker = RetrainingWorker(
            self.session_paths,
            self.base_model_path,
            self.epochs,
            save_path
        )
        
        self.worker.progress.connect(self.on_progress_update)
        self.worker.log_message.connect(self.log_text.append)
        self.worker.training_completed.connect(self.on_training_completed)
        self.worker.error_occurred.connect(self.on_error)
        
        self.worker.start()
    
    def on_progress_update(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def on_training_completed(self, results):
        self.log_text.append(f"\n=== 재학습 완료 ===")
        self.log_text.append(f"최고 정확도: {results['best_accuracy']:.3f}")
        self.log_text.append(f"학습 시간: {results['training_time']:.1f}초")
        self.log_text.append(f"모델 저장: {results['model_path']}")
        
        QMessageBox.information(
            self, "재학습 완료",
            f"모델 재학습이 완료되었습니다.\n\n"
            f"사용된 세션: {results['sessions_used']}개\n"
            f"최고 정확도: {results['best_accuracy']:.3f}\n"
            f"모델 경로: {Path(results['model_path']).name}"
        )
        
        self.accept()
    
    def on_error(self, error_msg):
        self.log_text.append(f"\n[오류] {error_msg}")
        QMessageBox.critical(self, "재학습 오류", f"재학습 중 오류가 발생했습니다:\n{error_msg}")
        
        self.start_btn.setEnabled(True)
        self.status_label.setText("오류 발생")