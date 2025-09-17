"""
세션 관리 모듈 - 세션 저장/로드 및 관리
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

from ..utils import save_session, load_session, get_polygon_session_name

logger = logging.getLogger(__name__)


class SessionManager:
    """세션 관리 클래스"""
    
    def __init__(self, session_dir: str = "sessions"):
        """
        Args:
            session_dir: 세션 저장 디렉토리
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        
        # 현재 세션 정보
        self.current_session_path = None
        self.current_file_path = None
        self.modified_sessions = []
        
        logger.info(f"SessionManager 초기화: {self.session_dir}")
    
    def save_session(self, file_path: str, 
                    points: Dict[str, List[Tuple[float, float]]], 
                    skeleton: List[List[float]], 
                    metadata: Optional[Dict[str, Any]] = None,
                    user_actions: Optional[List[Dict]] = None,
                    polygon_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        세션 저장
        
        Args:
            file_path: 원본 파일 경로
            points: 점 데이터 {'intersection': [...], 'curve': [...], 'endpoint': [...]}
            skeleton: 스켈레톤 데이터
            metadata: 메타데이터
            user_actions: 사용자 액션 리스트
            polygon_info: 폴리곤 정보 (멀티폴리곤의 경우)
        
        Returns:
            저장된 세션 파일 경로
        """
        try:
            # 기본 메타데이터 설정
            if metadata is None:
                metadata = {}
            
            default_metadata = {
                'process': 'ai_correction',
                'total_points': len(skeleton) if skeleton else 0,
                'point_counts': {
                    category: len(point_list) for category, point_list in points.items()
                },
                'file_mode': metadata.get('file_mode', 'road'),
                'timestamp': metadata.get('timestamp'),
                'version': '1.0'
            }
            
            # 메타데이터 병합
            final_metadata = {**default_metadata, **metadata}
            
            # 세션 저장
            session_path = save_session(
                file_path=file_path,
                labels=points,
                skeleton=skeleton,
                metadata=final_metadata,
                user_actions=user_actions or [],
                polygon_info=polygon_info
            )
            
            if session_path:
                self.current_session_path = session_path
                self.current_file_path = file_path
                
                # 수정된 세션 목록에 추가
                if session_path not in self.modified_sessions:
                    self.modified_sessions.append(session_path)
                
                logger.info(f"세션 저장 성공: {session_path}")
                return session_path
            else:
                logger.error("세션 저장 실패")
                return None
                
        except Exception as e:
            logger.error(f"세션 저장 오류: {e}")
            return None
    
    def load_session(self, session_path: str) -> Optional[Dict[str, Any]]:
        """
        세션 로드
        
        Args:
            session_path: 세션 파일 경로
        
        Returns:
            세션 데이터
        """
        try:
            session_data = load_session(session_path)
            
            if session_data:
                self.current_session_path = session_path
                logger.info(f"세션 로드 성공: {session_path}")
                return session_data
            else:
                logger.error(f"세션 로드 실패: {session_path}")
                return None
                
        except Exception as e:
            logger.error(f"세션 로드 오류: {e}")
            return None
    
    def save_polygon_session(self, file_path: str, 
                           points: Dict[str, List[Tuple[float, float]]], 
                           skeleton: List[List[float]],
                           polygon_index: int,
                           total_polygons: int,
                           target_crs: str = 'EPSG:5186',
                           metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        폴리곤 세션 저장 (멀티폴리곤용)
        
        Args:
            file_path: 원본 파일 경로
            points: 점 데이터
            skeleton: 스켈레톤 데이터
            polygon_index: 폴리곤 인덱스 (1부터 시작)
            total_polygons: 총 폴리곤 수
            target_crs: 좌표계
            metadata: 추가 메타데이터
        
        Returns:
            저장된 세션 파일 경로
        """
        try:
            # 폴리곤 정보
            polygon_info = {
                'index': polygon_index,
                'total': total_polygons
            }
            
            # 메타데이터 설정
            polygon_metadata = {
                'file_mode': 'district',
                'polygon_index': polygon_index,
                'total_polygons': total_polygons,
                'target_crs': target_crs,
                **(metadata or {})
            }
            
            # 세션 저장
            session_path = self.save_session(
                file_path=file_path,
                points=points,
                skeleton=skeleton,
                metadata=polygon_metadata,
                polygon_info=polygon_info
            )
            
            if session_path:
                logger.info(f"폴리곤 세션 저장: {polygon_index}/{total_polygons}")
                return session_path
            else:
                logger.error(f"폴리곤 세션 저장 실패: {polygon_index}/{total_polygons}")
                return None
                
        except Exception as e:
            logger.error(f"폴리곤 세션 저장 오류: {e}")
            return None
    
    def list_sessions(self, file_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        세션 목록 조회
        
        Args:
            file_pattern: 파일명 패턴 (선택사항)
        
        Returns:
            세션 정보 리스트
        """
        try:
            sessions = []
            
            for session_file in self.session_dir.glob("*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    # 파일 패턴 필터링
                    if file_pattern and file_pattern not in session_file.name:
                        continue
                    
                    # 세션 정보 추출
                    session_info = {
                        'file_path': str(session_file),
                        'file_name': session_file.name,
                        'original_file': session_data.get('original_file', ''),
                        'metadata': session_data.get('metadata', {}),
                        'point_counts': session_data.get('metadata', {}).get('point_counts', {}),
                        'timestamp': session_data.get('metadata', {}).get('timestamp', ''),
                        'file_size': session_file.stat().st_size,
                        'modified_time': session_file.stat().st_mtime
                    }
                    
                    sessions.append(session_info)
                    
                except Exception as e:
                    logger.warning(f"세션 파일 읽기 오류: {session_file} - {e}")
                    continue
            
            # 수정 시간순 정렬 (최신 순)
            sessions.sort(key=lambda x: x['modified_time'], reverse=True)
            
            logger.info(f"세션 목록 조회: {len(sessions)}개")
            return sessions
            
        except Exception as e:
            logger.error(f"세션 목록 조회 오류: {e}")
            return []
    
    def delete_session(self, session_path: str) -> bool:
        """
        세션 삭제
        
        Args:
            session_path: 삭제할 세션 파일 경로
        
        Returns:
            삭제 성공 여부
        """
        try:
            session_file = Path(session_path)
            
            if session_file.exists():
                session_file.unlink()
                
                # 수정된 세션 목록에서 제거
                if session_path in self.modified_sessions:
                    self.modified_sessions.remove(session_path)
                
                logger.info(f"세션 삭제 성공: {session_path}")
                return True
            else:
                logger.warning(f"세션 파일이 존재하지 않습니다: {session_path}")
                return False
                
        except Exception as e:
            logger.error(f"세션 삭제 오류: {e}")
            return False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        세션 통계 정보
        
        Returns:
            통계 정보
        """
        try:
            sessions = self.list_sessions()
            
            if not sessions:
                return {
                    'total_sessions': 0,
                    'modified_sessions': len(self.modified_sessions),
                    'total_size': 0,
                    'avg_size': 0
                }
            
            # 통계 계산
            total_sessions = len(sessions)
            total_size = sum(session['file_size'] for session in sessions)
            avg_size = total_size / total_sessions if total_sessions > 0 else 0
            
            # 파일 모드별 분포
            file_modes = {}
            for session in sessions:
                mode = session['metadata'].get('file_mode', 'unknown')
                file_modes[mode] = file_modes.get(mode, 0) + 1
            
            return {
                'total_sessions': total_sessions,
                'modified_sessions': len(self.modified_sessions),
                'total_size': total_size,
                'avg_size': avg_size,
                'file_modes': file_modes
            }
            
        except Exception as e:
            logger.error(f"세션 통계 계산 오류: {e}")
            return {
                'total_sessions': 0,
                'modified_sessions': 0,
                'total_size': 0,
                'avg_size': 0,
                'file_modes': {}
            }
    
    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """
        오래된 세션 정리
        
        Args:
            max_age_days: 최대 보관 기간 (일)
        
        Returns:
            삭제된 세션 수
        """
        try:
            import time
            
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            deleted_count = 0
            
            for session_file in self.session_dir.glob("*.json"):
                try:
                    file_age = current_time - session_file.stat().st_mtime
                    
                    if file_age > max_age_seconds:
                        session_file.unlink()
                        deleted_count += 1
                        logger.info(f"오래된 세션 삭제: {session_file.name}")
                        
                except Exception as e:
                    logger.warning(f"세션 삭제 오류: {session_file} - {e}")
                    continue
            
            logger.info(f"오래된 세션 정리 완료: {deleted_count}개 삭제")
            return deleted_count
            
        except Exception as e:
            logger.error(f"오래된 세션 정리 오류: {e}")
            return 0
    
    def export_sessions(self, export_path: str, file_pattern: Optional[str] = None) -> bool:
        """
        세션 내보내기
        
        Args:
            export_path: 내보낼 경로
            file_pattern: 파일명 패턴 (선택사항)
        
        Returns:
            내보내기 성공 여부
        """
        try:
            import shutil
            
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            sessions = self.list_sessions(file_pattern)
            exported_count = 0
            
            for session in sessions:
                try:
                    source_path = Path(session['file_path'])
                    dest_path = export_dir / source_path.name
                    
                    shutil.copy2(source_path, dest_path)
                    exported_count += 1
                    
                except Exception as e:
                    logger.warning(f"세션 내보내기 오류: {session['file_name']} - {e}")
                    continue
            
            logger.info(f"세션 내보내기 완료: {exported_count}개")
            return exported_count > 0
            
        except Exception as e:
            logger.error(f"세션 내보내기 오류: {e}")
            return False
    
    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """현재 세션 정보 반환"""
        if self.current_session_path:
            return {
                'session_path': self.current_session_path,
                'file_path': self.current_file_path,
                'modified_sessions_count': len(self.modified_sessions)
            }
        return None 