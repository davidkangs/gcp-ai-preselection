"""
파일명: process1_labeling_tool.py
경로: /프로젝트루트/process1_labeling_tool.py
설명: 프로세스 1 - 라벨링 도구 (향상된 휴리스틱 통합)
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import geopandas as gpd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox, 
    QTextEdit, QProgressDialog, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

# 프로젝트 모듈 import
from src.core.skeleton_extractor import SkeletonExtractor
from src.ui.canvas_widget import CanvasWidget
from src.utils import save_session
# from enhanced_heuristic_detector_v2 import EnhancedHeuristicDetectorV2  # 원본: 임시 비활성화
# 라벨링 도구 시작할 때
# from src.learning import DQNDataCollector  # 원본: 임시 비활성화
# collector = DQNDataCollector()  # 원본: 임시 비활성화
# collector.connect_to_canvas(self.canvas)  # 원본: 임시 비활성화
# 위성영상 설정
# 위성영상 설정 (패치: 임시 비활성화)
# try:
#     from src.utils.satellite_config import setup_providers
#     setup_providers()
# except:
#     pass
# networkx는 여기서 import하지 않음! enhanced_heuristic_detector_v2에서만 사용


class ProcessingThread(QThread):
    """백그라운드 처리 스레드"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.skeleton_extractor = SkeletonExtractor()
        # self.heuristic_detector = EnhancedHeuristicDetectorV2()  # 원본: 임시 비활성화
    
    def run(self):
        """처리 실행"""
        try:
            # 1단계: Shapefile 읽기
            self.progress.emit(10, "Shapefile 읽는 중...")
            gdf = gpd.read_file(self.file_path)
            if os.path.exists(self.file_path):
                gdf = gpd.read_file(self.file_path)
            else:
                # data 폴더에서 찾기
                data_path = os.path.join('data', os.path.basename(self.file_path))
                if os.path.exists(data_path):
                    gdf = gpd.read_file(data_path)
                else:
                    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.file_path}")
            # 2단계: 스켈레톤 추출
            self.progress.emit(40, "스켈레톤 추출 중...")
            start_time = time.time()
            skeleton, intersections = self.skeleton_extractor.process_shapefile(self.file_path)
            processing_time = time.time() - start_time
            
            # 3단계: 기본 휴리스틱 적용 (패치: 간소화)
            self.progress.emit(85, "기본 휴리스틱 검출 중...")
            
            # 기본 교차점, 커브, 끝점 검출
            detected = {
                'intersection': intersections,  # 기본 교차점 사용
                'curve': [],  # 임시로 비움
                'endpoint': []  # 임시로 비움
            }
            
            # 4단계: 결과 정리
            self.progress.emit(90, "결과 정리 중...")
            result = {
                'success': True,
                'gdf': gdf,
                'skeleton': skeleton,
                'intersections': detected['intersection'],  # NetworkX 기반 교차점
                'curves': detected['curve'],  # 민감한 커브 검출
                'endpoints': detected['endpoint'],  # 끝점 검출
                'processing_time': processing_time
            }
            
            self.progress.emit(100, "완료!")
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class Process1LabelingTool(QMainWindow):
    """프로세스 1 - 라벨링 도구 메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.processing_thread = None
        
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("도로망 라벨링 도구 - Process 1 (향상된 휴리스틱)")
        self.setGeometry(100, 100, 1400, 900)
        
        # 메인 위젯과 레이아웃
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Splitter로 좌우 분할
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 왼쪽 패널
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # 오른쪽 캔버스
        self.canvas_widget = CanvasWidget()
        splitter.addWidget(self.canvas_widget)
        
        splitter.setSizes([400, 1000])
        
        # 상태바
        self.statusBar().showMessage("준비")
        
    def create_left_panel(self):
        """왼쪽 컨트롤 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 파일 선택 그룹
        file_group = QGroupBox("1. 파일 선택")
        file_layout = QVBoxLayout()
        
        select_btn = QPushButton("Shapefile 선택")
        select_btn.clicked.connect(self.select_file)
        file_layout.addWidget(select_btn)
        
        self.file_label = QLabel("파일: 선택 안됨")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        process_btn = QPushButton("처리 시작")
        process_btn.clicked.connect(self.process_file)
        process_btn.setStyleSheet("QPushButton {background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;}")
        file_layout.addWidget(process_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 통계 그룹
        stats_group = QGroupBox("2. 라벨링 통계")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        stats_layout.addWidget(self.stats_text)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # 수동 편집 그룹
        edit_group = QGroupBox("3. 수동 편집")
        edit_layout = QVBoxLayout()
        
        edit_info = QLabel(
            "• 좌클릭: 커브 수정 (필요시)\n"
            "• 우클릭: 끝점 추가 (도로의 끝)\n"
            "• Shift+클릭: 제거\n"
            "• D: 가장 가까운 점 삭제\n"
            "• Space: 화면 맞춤\n"
            "\n※ 교차점과 커브는 자동 검출됩니다"
        )
        edit_info.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; }")
        edit_layout.addWidget(edit_info)
        
        edit_group.setLayout(edit_layout)
        layout.addWidget(edit_group)
        

        # DQN 예측 그룹 추가
        dqn_group = QGroupBox("5. DQN AI 예측")
        dqn_layout = QVBoxLayout()
        
        dqn_predict_btn = QPushButton("DQN 자동 예측")
        dqn_predict_btn.clicked.connect(self.run_dqn_prediction)
        dqn_predict_btn.setStyleSheet("QPushButton {background-color: #9C27B0; color: white; font-weight: bold; padding: 8px;}")
        dqn_layout.addWidget(dqn_predict_btn)
        
        dqn_info = QLabel("학습된 DQN 모델로 포인트 예측\n※ 먼저 프로세스 2에서 학습 필요")
        dqn_info.setStyleSheet("QLabel {background-color: #f8f8f8; padding: 8px; font-size: 10px;}")
        dqn_layout.addWidget(dqn_info)
        
        dqn_group.setLayout(dqn_layout)
        layout.addWidget(dqn_group)

        # 저장 그룹
        save_group = QGroupBox("4. 저장")
        save_layout = QVBoxLayout()
        
        save_btn = QPushButton("세션 저장")
        save_btn.clicked.connect(self.save_session)
        save_layout.addWidget(save_btn)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)

        layout.addStretch()
        return panel
    



        
        layout.addStretch()
        return panel
    
    def select_file(self):
        """파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Shapefile 선택", "", "Shapefiles (*.shp)"
        )
        
        if file_path:
            self.current_file = file_path
            self.file_label.setText(f"파일: {Path(file_path).name}")
            self.canvas_widget.clear_all()
            self.update_stats()
    
    def process_file(self):
        """파일 처리"""
        if not self.current_file:
            QMessageBox.warning(self, "경고", "파일을 먼저 선택하세요.")
            return
        
        # 진행 다이얼로그
        self.progress_dialog = QProgressDialog("처리 중...", "취소", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # 처리 스레드 시작
        self.processing_thread = ProcessingThread(self.current_file)
        self.processing_thread.progress.connect(self.on_progress)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()
    
    def on_progress(self, value, message):
        """진행 상황 업데이트"""
        self.progress_dialog.setValue(value)
        self.progress_dialog.setLabelText(message)
        self.statusBar().showMessage(message)
    
    def on_processing_finished(self, result):
        """처리 완료"""
        self.progress_dialog.close()
        
        if result['success']:
            # 캔버스에 데이터 설정
            self.canvas_widget.current_file = self.current_file
            
            # 도로 데이터는 원본 좌표계 유지
            self.canvas_widget.set_road_data(result['gdf'])
            self.canvas_widget.skeleton = result['skeleton']
            self.canvas_widget.processing_time = result['processing_time']
            
            # 교차점 설정 (NetworkX 기반)
            self.canvas_widget.canvas.points['intersection'] = [
                (float(x), float(y)) for x, y in result['intersections']
            ]
            
            # 커브 설정 (민감한 커브 검출)
            self.canvas_widget.canvas.points['curve'] = [
                (float(x), float(y)) for x, y in result.get('curves', [])
            ]
            
            # 끝점 설정 (휴리스틱)
            self.canvas_widget.canvas.points['endpoint'] = [
                (float(x), float(y)) for x, y in result.get('endpoints', [])
            ]
            
            # 디스플레이 업데이트
            self.canvas_widget.update_display()
            
            # 통계 업데이트
            self.update_stats()
            
            self.statusBar().showMessage(
                f"처리 완료 - 처리시간: {result['processing_time']:.2f}초"
            )
    
    def on_processing_error(self, error_msg):
        """처리 오류"""
        self.progress_dialog.close()
        QMessageBox.critical(self, "오류", f"처리 중 오류 발생:\n{error_msg}")
        self.statusBar().showMessage("오류 발생")
    
    def update_stats(self):
        """통계 업데이트"""
        if not hasattr(self.canvas_widget, 'canvas'):
            self.stats_text.clear()
            return
        
        points = self.canvas_widget.canvas.points
        
        stats_text = f"""=== 라벨링 통계 ===
교차점: {len(points.get('intersection', []))}개 (NetworkX 그래프 기반)
커브: {len(points.get('curve', []))}개 (경계선 곡률 분석)
끝점: {len(points.get('endpoint', []))}개 (스켈레톤 끝점)
━━━━━━━━━━━━━━━━━━━━━
전체: {sum(len(v) for v in points.values())}개

※ 향상된 휴리스틱 적용:
  - 면적 편차가 크면 하위 5% 제거
  - 교차점 10m, 커브 10m 클러스터링
  - 교차점 주변 20m 내 커브 제거"""
        
        self.stats_text.setText(stats_text)
    

    def run_dqn_prediction(self):
        """DQN 예측 실행 (임시 비활성화)"""
        QMessageBox.information(self, "알림", "DQN 예측 기능은 프로세스 2 완료 후 활성화됩니다.")
        # if not hasattr(self.canvas_widget, 'canvas') or not self.canvas_widget.skeleton:
        #     QMessageBox.warning(self, "경고", "먼저 파일을 처리하세요.")
        #     return
        # 
        # try:
        #     self.canvas_widget.canvas.run_dqn_prediction()
        #     self.update_stats()
        #     QMessageBox.information(self, "완료", "DQN 예측이 완료되었습니다!")
        # except Exception as e:
        #     QMessageBox.critical(self, "오류", f"DQN 예측 실패:\n{str(e)}")

    def save_session(self):
        """세션 저장"""
        if not self.current_file:
            QMessageBox.warning(self, "경고", "저장할 데이터가 없습니다.")
            return
        
        try:
            labels = self.canvas_widget.canvas.points
            skeleton = self.canvas_widget.skeleton
            
            metadata = {
                'processing_time': getattr(self.canvas_widget, 'processing_time', 0),
                'heuristic_version': 'v2_enhanced',
                'features': {
                    'area_filtering': True,
                    'networkx_intersections': True,
                    'sensitive_curves': True,
                    'clustering': True
                }
            }
            
            session_path = save_session(self.current_file, labels, skeleton, metadata)
            
            if session_path:
                QMessageBox.information(self, "성공", f"세션이 저장되었습니다.\n{session_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "오류", f"저장 실패:\n{str(e)}")


def main():
    """메인 실행"""
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = Process1LabelingTool()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()