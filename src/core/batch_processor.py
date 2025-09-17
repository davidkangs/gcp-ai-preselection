import os
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import logging

logger = logging.getLogger(__name__)

class BatchProcessor(QObject):
    """배치 처리 관리자"""
    
    # 시그널 정의
    progress_updated = pyqtSignal(int, str)  # 진행률, 메시지
    file_processed = pyqtSignal(str, dict)   # 파일경로, 결과
    batch_completed = pyqtSignal(dict)       # 전체 결과
    error_occurred = pyqtSignal(str)         # 에러 메시지
    
    def __init__(self, skeleton_extractor, agent=None):
        super().__init__()
        self.skeleton_extractor = skeleton_extractor
        self.agent = agent
        self.results = []
        self.is_running = False
        
    def process_batch(self, file_paths, use_ai=True, save_sessions=True):
        """여러 파일 배치 처리
        
        Args:
            file_paths: 처리할 shapefile 경로 리스트
            use_ai: AI 자동 검출 사용 여부
            save_sessions: 세션 자동 저장 여부
        """
        self.is_running = True
        self.results = []
        total_files = len(file_paths)
        
        batch_start_time = time.time()
        
        for idx, file_path in enumerate(file_paths):
            if not self.is_running:
                break
            
            try:
                # 진행 상황 업데이트
                progress = int((idx / total_files) * 100)
                self.progress_updated.emit(progress, f"처리 중: {Path(file_path).name}")
                
                # 파일 처리
                result = self.process_single_file(file_path, use_ai)
                
                # 세션 저장
                if save_sessions and result['success']:
                    from ..utils import save_session
                    save_session(
                        file_path,
                        result['labels'],
                        result['skeleton'],
                        result['metadata']
                    )
                
                self.results.append(result)
                self.file_processed.emit(file_path, result)
                
            except Exception as e:
                logger.error(f"파일 처리 실패 {file_path}: {e}")
                error_result = {
                    'file_path': file_path,
                    'success': False,
                    'error': str(e),
                    'processing_time': 0
                }
                self.results.append(error_result)
                self.error_occurred.emit(f"파일 처리 실패: {Path(file_path).name}")
        
        # 배치 처리 완료
        batch_end_time = time.time()
        batch_summary = {
            'total_files': total_files,
            'successful': len([r for r in self.results if r['success']]),
            'failed': len([r for r in self.results if not r['success']]),
            'total_time': batch_end_time - batch_start_time,
            'results': self.results
        }
        
        self.progress_updated.emit(100, "배치 처리 완료")
        self.batch_completed.emit(batch_summary)
        self.is_running = False
    
    def process_single_file(self, file_path, use_ai=True):
        """단일 파일 처리
        
        Returns:
            dict: 처리 결과
        """
        start_time = time.time()
        
        try:
            # 스켈레톤 추출
            skeleton, intersections = self.skeleton_extractor.process_shapefile(file_path)
            
            labels = {
                'intersection': [(float(x), float(y)) for x, y in intersections],
                'curve': [],
                'endpoint': []
            }
            
            # AI 검출
            if use_ai and self.agent is not None:
                ai_labels = self.detect_with_ai(skeleton)
                # AI 검출 결과 병합 (교차점은 이미 휴리스틱으로 검출됨)
                labels['curve'] = ai_labels.get('curve', [])
                labels['endpoint'] = ai_labels.get('endpoint', [])
            
            # 메타데이터
            metadata = {
                'processing_time': time.time() - start_time,
                'skeleton_points': len(skeleton),
                'detected_intersections': len(labels['intersection']),
                'detected_curves': len(labels['curve']),
                'detected_endpoints': len(labels['endpoint']),
                'used_ai': use_ai
            }
            
            return {
                'file_path': file_path,
                'success': True,
                'skeleton': skeleton,
                'labels': labels,
                'metadata': metadata,
                'processing_time': metadata['processing_time']
            }
            
        except Exception as e:
            logger.error(f"파일 처리 오류: {e}")
            return {
                'file_path': file_path,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def detect_with_ai(self, skeleton):
        """AI를 사용한 특징점 검출"""
        from ..utils import extract_point_features
        
        all_features = []
        
        # 특징 추출
        for i, point in enumerate(skeleton):
            window_points = []
            for j in range(max(0, i-50), min(len(skeleton), i+50)):
                window_points.append(skeleton[j])
            
            features = extract_point_features(point, window_points, skeleton)
            all_features.append(features)
        
        # AI 예측
        predictions = self.agent.predict(np.array(all_features))
        
        # 결과 정리
        ai_labels = {
            'intersection': [],
            'curve': [],
            'endpoint': []
        }
        
        for i, pred in enumerate(predictions):
            if pred == 1:  # 교차점 (휴리스틱과 중복 가능)
                ai_labels['intersection'].append(tuple(skeleton[i]))
            elif pred == 2:  # 커브
                ai_labels['curve'].append(tuple(skeleton[i]))
            elif pred == 3:  # 끝점
                ai_labels['endpoint'].append(tuple(skeleton[i]))
        
        return ai_labels
    
    def stop(self):
        """배치 처리 중단"""
        self.is_running = False

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.metrics = {
            'training_history': [],
            'inference_times': [],
            'accuracy_history': [],
            'file_processing_times': {}
        }
        self.log_file = Path("logs/performance.json")
        self.log_file.parent.mkdir(exist_ok=True)
    
    def log_training(self, epoch, loss, accuracy=None, learning_rate=None):
        """학습 메트릭 기록"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'loss': float(loss),
            'accuracy': float(accuracy) if accuracy else None,
            'learning_rate': float(learning_rate) if learning_rate else None
        }
        self.metrics['training_history'].append(entry)
        self.save_metrics()
    
    def log_inference(self, num_points, inference_time):
        """추론 시간 기록"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'num_points': num_points,
            'inference_time': float(inference_time),
            'points_per_second': num_points / inference_time if inference_time > 0 else 0
        }
        self.metrics['inference_times'].append(entry)
        self.save_metrics()
    
    def log_file_processing(self, file_path, processing_time, num_detections):
        """파일 처리 성능 기록"""
        file_name = Path(file_path).name
        self.metrics['file_processing_times'][file_name] = {
            'timestamp': datetime.now().isoformat(),
            'processing_time': float(processing_time),
            'num_detections': num_detections
        }
        self.save_metrics()
    
    def calculate_accuracy(self, predictions, ground_truth):
        """정확도 계산"""
        if len(predictions) != len(ground_truth):
            return 0.0
        
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        accuracy = correct / len(predictions) if len(predictions) > 0 else 0.0
        
        self.metrics['accuracy_history'].append({
            'timestamp': datetime.now().isoformat(),
            'accuracy': float(accuracy),
            'total_samples': len(predictions),
            'correct_predictions': correct
        })
        self.save_metrics()
        
        return accuracy
    
    def save_metrics(self):
        """메트릭을 파일로 저장"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"메트릭 저장 실패: {e}")
    
    def load_metrics(self):
        """저장된 메트릭 로드"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    self.metrics = json.load(f)
            except Exception as e:
                logger.error(f"메트릭 로드 실패: {e}")
    
    def get_summary(self):
        """성능 요약 통계"""
        summary = {}
        
        # 학습 통계
        if self.metrics['training_history']:
            losses = [m['loss'] for m in self.metrics['training_history']]
            summary['training'] = {
                'total_epochs': len(self.metrics['training_history']),
                'final_loss': losses[-1],
                'min_loss': min(losses),
                'avg_loss': np.mean(losses)
            }
        
        # 추론 통계
        if self.metrics['inference_times']:
            times = [m['inference_time'] for m in self.metrics['inference_times']]
            pps = [m['points_per_second'] for m in self.metrics['inference_times']]
            summary['inference'] = {
                'total_inferences': len(self.metrics['inference_times']),
                'avg_time': np.mean(times),
                'avg_points_per_second': np.mean(pps)
            }
        
        # 정확도 통계
        if self.metrics['accuracy_history']:
            accuracies = [m['accuracy'] for m in self.metrics['accuracy_history']]
            summary['accuracy'] = {
                'current_accuracy': accuracies[-1],
                'best_accuracy': max(accuracies),
                'avg_accuracy': np.mean(accuracies)
            }
        
        # 파일 처리 통계
        if self.metrics['file_processing_times']:
            times = [v['processing_time'] for v in self.metrics['file_processing_times'].values()]
            summary['file_processing'] = {
                'total_files': len(self.metrics['file_processing_times']),
                'avg_processing_time': np.mean(times),
                'total_processing_time': sum(times)
            }
        
        return summary
    
    def export_to_csv(self, output_dir):
        """메트릭을 CSV로 내보내기"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 학습 히스토리
        if self.metrics['training_history']:
            df = pd.DataFrame(self.metrics['training_history'])
            df.to_csv(output_dir / 'training_history.csv', index=False)
        
        # 추론 시간
        if self.metrics['inference_times']:
            df = pd.DataFrame(self.metrics['inference_times'])
            df.to_csv(output_dir / 'inference_times.csv', index=False)
        
        # 정확도 히스토리
        if self.metrics['accuracy_history']:
            df = pd.DataFrame(self.metrics['accuracy_history'])
            df.to_csv(output_dir / 'accuracy_history.csv', index=False)

class BatchProcessorThread(QThread):
    """배치 처리를 위한 별도 스레드"""
    
    def __init__(self, batch_processor, file_paths, use_ai=True, save_sessions=True):
        super().__init__()
        self.batch_processor = batch_processor
        self.file_paths = file_paths
        self.use_ai = use_ai
        self.save_sessions = save_sessions
    
    def run(self):
        """스레드 실행"""
        self.batch_processor.process_batch(
            self.file_paths, 
            self.use_ai, 
            self.save_sessions
        )