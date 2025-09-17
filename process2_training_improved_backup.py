"""
개선된 프로세스 2: Session 데이터 기반 DQN 학습
기존 session JSON을 DQN 학습 데이터로 변환
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QProgressBar, QMessageBox,
    QGroupBox, QSpinBox, QTextEdit, QTabWidget, QTableWidget,
    QTableWidgetItem, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionDataConverter:
    """Session JSON 데이터를 DQN 학습 데이터로 변환"""
    
    def __init__(self):
        self.feature_dim = 10  # 상태 벡터 차원
    
    def convert_sessions_to_dqn_data(self, session_files: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Session 파일들을 DQN 학습 데이터로 변환"""
        all_states = []
        all_labels = []
        
        for session_file in session_files:
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                states, labels = self._extract_features_from_session(session_data)
                all_states.extend(states)
                all_labels.extend(labels)
                
            except Exception as e:
                logger.error(f"Session 파일 처리 실패 {session_file}: {e}")
                continue
        
        if not all_states:
            return np.array([]), np.array([])
        
        return np.array(all_states), np.array(all_labels)
    
    def _extract_features_from_session(self, session_data: Dict) -> Tuple[List, List]:
        """단일 session에서 특징과 라벨 추출"""
        states = []
        labels = []
        
        # 스켈레톤 포인트 가져오기
        skeleton = session_data.get('skeleton', [])
        if not skeleton:
            return [], []
        
        # 라벨 데이터 가져오기
        session_labels = session_data.get('labels', {})
        intersection_points = set(tuple(p) for p in session_labels.get('intersection', []))
        curve_points = set(tuple(p) for p in session_labels.get('curve', []))
        endpoint_points = set(tuple(p) for p in session_labels.get('endpoint', []))
        
        # 각 스켈레톤 포인트에 대해 특징 추출
        for i, point in enumerate(skeleton):
            if len(point) < 2:
                continue
                
            x, y = float(point[0]), float(point[1])
            
            # 상태 벡터 생성
            state_vector = self._create_state_vector(x, y, skeleton, i)
            
            # 라벨 결정 (0: 일반, 1: 교차점, 2: 커브, 3: 끝점)
            point_tuple = (x, y)
            if point_tuple in intersection_points:
                label = 1
            elif point_tuple in curve_points:
                label = 2
            elif point_tuple in endpoint_points:
                label = 3
            else:
                label = 0
            
            states.append(state_vector)
            labels.append(label)
        
        return states, labels
    
    def _create_state_vector(self, x: float, y: float, skeleton: List, idx: int) -> List[float]:
        """포인트의 상태 벡터 생성"""
        features = []
        
        # 1-2. 현재 위치
        features.extend([x, y])
        
        # 3-4. 이전 포인트와의 거리와 각도
        if idx > 0:
            prev_x, prev_y = skeleton[idx-1][0], skeleton[idx-1][1]
            dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            angle = np.arctan2(y - prev_y, x - prev_x)
            features.extend([dist, angle])
        else:
            features.extend([0.0, 0.0])
        
        # 5-6. 다음 포인트와의 거리와 각도
        if idx < len(skeleton) - 1:
            next_x, next_y = skeleton[idx+1][0], skeleton[idx+1][1]
            dist = np.sqrt((next_x - x)**2 + (next_y - y)**2)
            angle = np.arctan2(next_y - y, next_x - x)
            features.extend([dist, angle])
        else:
            features.extend([0.0, 0.0])
        
        # 7. 주변 포인트 밀도
        density = self._calculate_local_density(x, y, skeleton, radius=50)
        features.append(density)
        
        # 8. 곡률 계산
        curvature = self._calculate_curvature(skeleton, idx)
        features.append(curvature)
        
        # 9-10. 위치 정규화 (좌표계 독립적)
        features.extend([x % 100, y % 100])
        
        return features[:self.feature_dim]
    
    def _calculate_local_density(self, x: float, y: float, skeleton: List, radius: float = 50) -> float:
        """주변 포인트 밀도 계산"""
        count = 0
        for point in skeleton:
            if len(point) >= 2:
                px, py = point[0], point[1]
                if np.sqrt((x - px)**2 + (y - py)**2) <= radius:
                    count += 1
        return count / len(skeleton) if skeleton else 0.0
    
    def _calculate_curvature(self, skeleton: List, idx: int) -> float:
        """곡률 계산"""
        if idx == 0 or idx >= len(skeleton) - 1:
            return 0.0
        
        try:
            # 3점을 이용한 곡률 계산
            p1 = np.array(skeleton[idx-1][:2])
            p2 = np.array(skeleton[idx][:2])
            p3 = np.array(skeleton[idx+1][:2])
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 각도 변화량으로 곡률 계산
            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])
            
            curvature = abs(angle2 - angle1)
            if curvature > np.pi:
                curvature = 2 * np.pi - curvature
                
            return curvature
        except:
            return 0.0


class SimpleDQN(nn.Module):
    """간단한 DQN 네트워크"""
    
    def __init__(self, input_size=20, hidden_sizes=[128, 64], output_size=5):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        return self.network(x)


class ImprovedDQNTrainer:
    """개선된 DQN 트레이너"""
    
    def __init__(self, feature_dim=10, num_classes=4):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 초기화
        self.model = SimpleDQN(feature_dim, output_size=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        
        self.converter = SessionDataConverter()
        
        logger.info(f"DQN 트레이너 초기화 완료 - 디바이스: {self.device}")
    
    def train_from_sessions(self, session_dir: str, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        """Session 파일들로부터 학습"""
        session_files = list(Path(session_dir).glob("session_*.json"))
        
        if not session_files:
            return {'success': False, 'error': 'Session 파일을 찾을 수 없습니다'}
        
        logger.info(f"{len(session_files)}개의 session 파일 발견")
        
        # 데이터 변환
        X, y = self.converter.convert_sessions_to_dqn_data(session_files)
        
        if len(X) == 0:
            return {'success': False, 'error': '변환된 학습 데이터가 없습니다'}
        
        logger.info(f"변환된 데이터: {len(X)}개 샘플, {self.feature_dim}차원")
        
        # 클래스 분포 확인
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"클래스 분포: {dict(zip(unique, counts))}")
        
        # 학습/검증 분할
        indices = np.random.permutation(len(X))
        split_idx = int(len(X) * (1 - validation_split))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # 텐서 변환
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        # 학습 실행
        train_losses = []
        val_accuracies = []
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            # 학습
            self.model.train()
            epoch_loss = 0.0
            
            for batch_start in range(0, len(X_train), batch_size):
                batch_end = min(batch_start + batch_size, len(X_train))
                X_batch = X_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # 검증
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_pred = torch.argmax(val_outputs, dim=1)
                val_accuracy = (val_pred == y_val).float().mean().item()
            
            train_losses.append(epoch_loss / (len(X_train) // batch_size))
            val_accuracies.append(val_accuracy)
            
            # 스케줄러 업데이트
            self.scheduler.step(epoch_loss)
            
            # 최고 정확도 업데이트
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss={train_losses[-1]:.4f}, Val_Acc={val_accuracy:.4f}")
        
        # 최종 평가
        final_accuracy = val_accuracies[-1]
        final_loss = train_losses[-1]
        
        return {
            'success': True,
            'epochs': epochs,
            'final_accuracy': final_accuracy,
            'best_accuracy': best_accuracy,
            'final_loss': final_loss,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'num_samples': len(X),
            'class_distribution': dict(zip(unique.tolist(), counts.tolist()))
        }
    
    def save_model(self, model_path: str = "src/learning/models/session_dqn_model.pth"):
        """모델 저장"""
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes
        }, model_path)
        
        logger.info(f"모델 저장 완료: {model_path}")
        return model_path


class ImprovedTrainingWorker(QThread):
    """개선된 학습 워커"""
    progress = pyqtSignal(int, str)
    training_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, session_dir, config):
        super().__init__()
        self.session_dir = session_dir
        self.config = config
        self.trainer = ImprovedDQNTrainer()
        self.is_running = True
    
    def run(self):
        try:
            self.progress.emit(10, "Session 파일 스캔 중...")
            self.log_message.emit(f"📁 Session 디렉토리: {self.session_dir}")
            
            # Session 파일 확인
            session_files = list(Path(self.session_dir).glob("session_*.json"))
            if not session_files:
                self.error_occurred.emit(f"Session 파일을 찾을 수 없습니다: {self.session_dir}")
                return
            
            self.log_message.emit(f"✅ {len(session_files)}개 Session 파일 발견")
            
            self.progress.emit(30, "DQN 학습 시작...")
            self.log_message.emit("🚀 Session → DQN 데이터 변환 및 학습 시작")
            
            # 학습 실행
            results = self.trainer.train_from_sessions(
                self.session_dir,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size']
            )
            
            if not results['success']:
                self.error_occurred.emit(results.get('error', '알 수 없는 오류'))
                return
            
            self.progress.emit(80, "모델 저장 중...")
            
            # 모델 저장
            model_path = self.trainer.save_model()
            results['model_path'] = model_path
            
            self.progress.emit(100, "학습 완료!")
            self.log_message.emit("✨ DQN 학습 성공적 완료!")
            self.training_completed.emit(results)
            
        except Exception as e:
            logger.error(f"학습 오류: {e}")
            self.error_occurred.emit(str(e))


class ImprovedTrainingTool(QMainWindow):
    """개선된 Session 기반 DQN 학습 도구"""
    
    def __init__(self):
        super().__init__()
        self.session_dir = "sessions"  # 기본 세션 디렉토리
        self.training_worker = None
        self.init_ui()
        self.check_session_data()
    
    def init_ui(self):
        self.setWindowTitle("Session 기반 DQN 학습 도구 v2.0")
        self.setGeometry(100, 100, 900, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 왼쪽 컨트롤 패널
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 오른쪽 모니터링 패널
        right_panel = self.create_monitor_panel()
        main_layout.addWidget(right_panel, 2)
        
        self.statusBar().showMessage("✨ Session 기반 DQN 학습 도구 준비")
    
    def create_control_panel(self):
        """컨트롤 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Session 데이터 상태
        data_group = QGroupBox("📊 Session 데이터")
        data_layout = QVBoxLayout()
        
        # Session 디렉토리 선택
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel(f"디렉토리: {self.session_dir}")
        dir_layout.addWidget(self.dir_label)
        
        browse_btn = QPushButton("찾아보기")
        browse_btn.clicked.connect(self.browse_session_dir)
        dir_layout.addWidget(browse_btn)
        data_layout.addLayout(dir_layout)
        
        self.data_status_label = QLabel("데이터 확인 중...")
        data_layout.addWidget(self.data_status_label)
        
        refresh_btn = QPushButton("새로고침")
        refresh_btn.clicked.connect(self.check_session_data)
        data_layout.addWidget(refresh_btn)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # 학습 설정
        config_group = QGroupBox("⚙️ 학습 설정")
        config_layout = QVBoxLayout()
        
        # 에포크 수
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("에포크:"))
        self.epoch_spinbox = QSpinBox()
        self.epoch_spinbox.setRange(10, 200)
        self.epoch_spinbox.setValue(50)
        epoch_layout.addWidget(self.epoch_spinbox)
        config_layout.addLayout(epoch_layout)
        
        # 배치 크기
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("배치 크기:"))
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(16, 128)
        self.batch_spinbox.setValue(32)
        batch_layout.addWidget(self.batch_spinbox)
        config_layout.addLayout(batch_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # 학습 제어
        control_group = QGroupBox("🚀 학습 제어")
        control_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("Session 기반 학습 시작")
        self.start_btn.clicked.connect(self.start_training)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 12px;
                font-size: 14px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("중지")
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
        panel.setLayout(layout)
        return panel
    
    def create_monitor_panel(self):
        """모니터링 패널 생성"""
        tab_widget = QTabWidget()
        
        # 로그 탭
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #263238;
                color: #00E676;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                line-height: 1.4;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        log_tab.setLayout(log_layout)
        tab_widget.addTab(log_tab, "📝 학습 로그")
        
        # 결과 탭
        result_tab = QWidget()
        result_layout = QVBoxLayout()
        
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["항목", "값"])
        result_layout.addWidget(self.result_table)
        
        result_tab.setLayout(result_layout)
        tab_widget.addTab(result_tab, "📊 학습 결과")
        
        return tab_widget
    
    def browse_session_dir(self):
        """Session 디렉토리 선택"""
        directory = QFileDialog.getExistingDirectory(
            self, "Session 디렉토리 선택", self.session_dir
        )
        
        if directory:
            self.session_dir = directory
            self.dir_label.setText(f"디렉토리: {directory}")
            self.check_session_data()
    
    def check_session_data(self):
        """Session 데이터 확인"""
        try:
            session_files = list(Path(self.session_dir).glob("session_*.json"))
            
            if not session_files:
                self.data_status_label.setText(f"""❌ Session 파일 없음

{self.session_dir} 디렉토리에서
session_*.json 파일을 찾을 수 없습니다.

라벨링 도구에서 먼저 작업을 저장하세요.""")
                self.start_btn.setEnabled(False)
                return
            
            # Session 파일 분석
            total_samples = 0
            class_counts = {'intersection': 0, 'curve': 0, 'endpoint': 0, 'normal': 0}
            
            for session_file in session_files[:5]:  # 처음 5개만 분석
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    labels = session_data.get('labels', {})
                    skeleton = session_data.get('skeleton', [])
                    
                    total_samples += len(skeleton)
                    class_counts['intersection'] += len(labels.get('intersection', []))
                    class_counts['curve'] += len(labels.get('curve', []))
                    class_counts['endpoint'] += len(labels.get('endpoint', []))
                    
                except Exception as e:
                    continue
            
            status_text = f"""✅ Session 데이터 준비됨

📁 Session 파일: {len(session_files)}개
📊 예상 학습 샘플: ~{total_samples:,}개

클래스별 라벨 (샘플링):
🔴 교차점: {class_counts['intersection']:,}개
🔵 커브: {class_counts['curve']:,}개  
🟢 끝점: {class_counts['endpoint']:,}개

DQN 학습이 가능합니다!"""
            
            self.data_status_label.setText(status_text)
            self.start_btn.setEnabled(True)
            
        except Exception as e:
            self.data_status_label.setText(f"❌ 데이터 확인 실패:\n{str(e)}")
            self.start_btn.setEnabled(False)
    
    def start_training(self):
        """학습 시작"""
        self.log_text.append(f"=== Session 기반 DQN 학습 시작 ===")
        self.log_text.append(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_text.append(f"📁 Session 디렉토리: {self.session_dir}")
        
        config = {
            'epochs': self.epoch_spinbox.value(),
            'batch_size': self.batch_spinbox.value()
        }
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # 학습 워커 시작
        self.training_worker = ImprovedTrainingWorker(self.session_dir, config)
        self.training_worker.progress.connect(self.on_progress_update)
        self.training_worker.training_completed.connect(self.on_training_completed)
        self.training_worker.error_occurred.connect(self.on_error)
        self.training_worker.log_message.connect(self.log_text.append)
        self.training_worker.start()
    
    def stop_training(self):
        """학습 중지"""
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.is_running = False
            self.training_worker.wait()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("학습 중지됨")
        self.log_text.append("⏹️ 사용자에 의해 학습이 중지되었습니다.")
    
    def on_progress_update(self, progress, message):
        """진행률 업데이트"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
    
    def on_training_completed(self, results):
        """학습 완료"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.log_text.append("\n🎉 === 학습 완료 ===")
        self.log_text.append(f"📈 최종 정확도: {results['final_accuracy']:.2%}")
        self.log_text.append(f"🏆 최고 정확도: {results['best_accuracy']:.2%}")
        self.log_text.append(f"📉 최종 손실: {results['final_loss']:.4f}")
        self.log_text.append(f"📊 학습 샘플: {results['num_samples']:,}개")
        
        if 'model_path' in results:
            self.log_text.append(f"💾 모델 저장: {results['model_path']}")
        
        # 결과 테이블 업데이트
        self.update_result_table(results)
        
        QMessageBox.information(
            self, "🎉 학습 완료",
            f"Session 기반 DQN 모델 학습이 완료되었습니다!\n\n"
            f"📈 최종 정확도: {results['final_accuracy']:.2%}\n"
            f"🏆 최고 정확도: {results['best_accuracy']:.2%}\n"
            f"📊 학습 샘플: {results['num_samples']:,}개\n\n"
            f"이제 라벨링 도구에서 AI 예측을 사용할 수 있습니다!"
        )
    
    def update_result_table(self, results):
        """결과 테이블 업데이트"""
        self.result_table.setRowCount(0)
        
        items = [
            ("학습 상태", "✅ 성공"),
            ("에포크 수", str(results['epochs'])),
            ("최종 정확도", f"{results['final_accuracy']:.2%}"),
            ("최고 정확도", f"{results['best_accuracy']:.2%}"),
            ("최종 손실", f"{results['final_loss']:.4f}"),
            ("학습 샘플 수", f"{results['num_samples']:,}개")
        ]
        
        if 'model_path' in results:
            items.append(("모델 경로", results['model_path']))
        
        if 'class_distribution' in results:
            class_dist = results['class_distribution']
            items.append(("클래스 분포", str(class_dist)))
        
        for i, (key, value) in enumerate(items):
            self.result_table.insertRow(i)
            self.result_table.setItem(i, 0, QTableWidgetItem(key))
            self.result_table.setItem(i, 1, QTableWidgetItem(value))
        
        self.result_table.resizeColumnsToContents()
    
    def on_error(self, error_msg):
        """오류 처리"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append(f"❌ [오류] {error_msg}")
        QMessageBox.critical(self, "학습 오류", f"학습 중 오류가 발생했습니다:\n{error_msg}")


def main():
    """메인 실행 함수"""
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 어두운 테마 적용
    palette = app.palette()
    app.setPalette(palette)
    
    window = ImprovedTrainingTool()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()