"""
투명한 캐싱 시스템 - 기존 코드 동작은 그대로 유지하면서 성능 향상
"""

import os
import pickle
import hashlib
import time
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Callable, Dict

logger = logging.getLogger(__name__)

def get_executable_dir():
    """실행 파일이 위치한 디렉토리 반환 (PyInstaller 호환)"""
    if getattr(sys, 'frozen', False):
        # PyInstaller로 빌드된 실행파일인 경우
        return Path(sys.executable).parent
    else:
        # 개발 환경에서 실행하는 경우
        return Path(__file__).parent.parent.parent

class TransparentCache:
    """기존 코드에 지장 없이 투명하게 작동하는 캐싱 시스템"""
    
    def __init__(self, cache_dir: str = "cache"):
        # 실행파일 기준으로 캐시 디렉토리 설정
        executable_dir = get_executable_dir()
        self.cache_dir = executable_dir / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 캐시 타입별 디렉토리 생성
        (self.cache_dir / "skeleton").mkdir(exist_ok=True)
        (self.cache_dir / "ai_prediction").mkdir(exist_ok=True)
        (self.cache_dir / "clipping").mkdir(exist_ok=True)
        (self.cache_dir / "processing").mkdir(exist_ok=True)
        # 🆕 폴리곤별 사용자 편집 상태 캐시
        (self.cache_dir / "polygon_edit").mkdir(exist_ok=True)
        
        logger.info(f"💾 캐시 시스템 초기화: {self.cache_dir}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """파일 경로와 수정 시간을 기반으로 해시 생성"""
        try:
            stat = os.stat(file_path)
            # 파일 경로 + 수정 시간 + 크기로 고유 식별
            content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(content.encode()).hexdigest()
        except:
            return hashlib.md5(file_path.encode()).hexdigest()
    
    def _get_cache_key(self, file_path: str, operation: str, **kwargs) -> str:
        """캐시 키 생성"""
        file_hash = self._get_file_hash(file_path)
        # 추가 파라미터들도 키에 포함
        params_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{operation}_{file_hash}_{hashlib.md5(params_str.encode()).hexdigest()[:8]}"
    
    def _get_cache_path(self, cache_type: str, cache_key: str) -> Path:
        """캐시 파일 경로 생성"""
        return self.cache_dir / cache_type / f"{cache_key}.pkl"
    
    def get(self, file_path: str, operation: str, **kwargs) -> Optional[Any]:
        """캐시에서 결과 가져오기"""
        try:
            cache_key = self._get_cache_key(file_path, operation, **kwargs)
            cache_path = self._get_cache_path(operation, cache_key)
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    logger.info(f"✅ 캐시 히트: {operation} - {Path(file_path).name}")
                    return cached_data
            
        except Exception as e:
            logger.warning(f"캐시 읽기 실패: {e}")
        
        return None
    
    def set(self, file_path: str, operation: str, data: Any, **kwargs) -> None:
        """결과를 캐시에 저장"""
        try:
            cache_key = self._get_cache_key(file_path, operation, **kwargs)
            cache_path = self._get_cache_path(operation, cache_key)
            
            # 디렉토리 생성
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                logger.info(f"💾 캐시 저장: {operation} - {Path(file_path).name}")
                
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {e}")
    
    def clear_old_cache(self, max_age_days: int = 30) -> None:
        """오래된 캐시 파일 정리"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 3600
            
            for cache_file in self.cache_dir.rglob("*.pkl"):
                if (current_time - cache_file.stat().st_mtime) > max_age_seconds:
                    cache_file.unlink(missing_ok=True)
                    
        except Exception as e:
            logger.warning(f"캐시 정리 실패: {e}")
    
    # 🆕 폴리곤별 사용자 편집 상태 캐싱 메서드들
    def get_polygon_cache_key(self, file_path: str, polygon_index: int = 0) -> str:
        """폴리곤별 고유 캐시 키 생성"""
        file_name = Path(file_path).stem
        file_hash = self._get_file_hash(file_path)[:12]  # 짧은 해시
        return f"{file_name}_p{polygon_index}_{file_hash}"
    
    def save_polygon_edit_state(self, file_path: str, polygon_index: int, edit_state: Dict) -> None:
        """폴리곤별 사용자 편집 상태 저장"""
        try:
            cache_key = self.get_polygon_cache_key(file_path, polygon_index)
            cache_path = self.cache_dir / "polygon_edit" / f"{cache_key}.pkl"
            
            # 편집 상태에 메타데이터 추가
            full_state = {
                'file_path': file_path,
                'polygon_index': polygon_index,
                'timestamp': time.time(),
                'edit_data': edit_state
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(full_state, f)
                logger.info(f"💾 폴리곤 편집 상태 저장: {Path(file_path).name} (폴리곤 {polygon_index + 1})")
                
        except Exception as e:
            logger.error(f"폴리곤 편집 상태 저장 실패: {e}")
    
    def load_polygon_edit_state(self, file_path: str, polygon_index: int = 0) -> Optional[Dict]:
        """폴리곤별 사용자 편집 상태 로드"""
        try:
            cache_key = self.get_polygon_cache_key(file_path, polygon_index)
            cache_path = self.cache_dir / "polygon_edit" / f"{cache_key}.pkl"
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    full_state = pickle.load(f)
                    
                    # 파일 변경 확인 (파일 수정 시간 체크)
                    cached_file_hash = self._get_file_hash(file_path)
                    cache_file_hash = self.get_polygon_cache_key(file_path, polygon_index).split('_')[-1]
                    
                    if cached_file_hash[:12] == cache_file_hash:
                        logger.info(f"✅ 폴리곤 편집 상태 로드: {Path(file_path).name} (폴리곤 {polygon_index + 1})")
                        return full_state['edit_data']
                    else:
                        logger.info(f"🔄 파일 변경됨 - 캐시 무효화: {Path(file_path).name}")
                        cache_path.unlink(missing_ok=True)
            
        except Exception as e:
            logger.warning(f"폴리곤 편집 상태 로드 실패: {e}")
        
        return None
    
    def clear_polygon_cache(self, file_path: str = None) -> None:
        """폴리곤 캐시 삭제 (특정 파일 또는 전체)"""
        try:
            polygon_cache_dir = self.cache_dir / "polygon_edit"
            
            if file_path:
                # 특정 파일의 폴리곤 캐시만 삭제
                file_name = Path(file_path).stem
                for cache_file in polygon_cache_dir.glob(f"{file_name}_*.pkl"):
                    cache_file.unlink(missing_ok=True)
                logger.info(f"🗑️ 폴리곤 캐시 삭제: {Path(file_path).name}")
            else:
                # 모든 폴리곤 캐시 삭제
                for cache_file in polygon_cache_dir.glob("*.pkl"):
                    cache_file.unlink(missing_ok=True)
                logger.info("🗑️ 모든 폴리곤 캐시 삭제")
                
        except Exception as e:
            logger.error(f"폴리곤 캐시 삭제 실패: {e}")

# 전역 캐시 인스턴스
_global_cache = None

def get_cache() -> TransparentCache:
    """전역 캐시 인스턴스 가져오기"""
    global _global_cache
    if _global_cache is None:
        _global_cache = TransparentCache()
    return _global_cache

def cached_operation(operation_name: str, file_path: str = "", **cache_params):
    """데코레이터: 함수 결과를 투명하게 캐싱"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # 캐시 확인
            if file_path:
                cached_result = cache.get(file_path, operation_name, **cache_params)
                if cached_result is not None:
                    return cached_result
            
            # 캐시가 없으면 원본 함수 실행
            result = func(*args, **kwargs)
            
            # 결과 캐싱
            if file_path and result is not None:
                cache.set(file_path, operation_name, result, **cache_params)
            
            return result
        return wrapper
    return decorator

def cache_skeleton_extraction(file_path: str, extractor_func: Callable, *args, **kwargs):
    """스켈레톤 추출 결과 캐싱 (오류 방지 강화)"""
    try:
        cache = get_cache()
        
        # 🔍 캐시 확인 로깅
        logger.info(f"📂 캐시 확인 중: {Path(file_path).name}")
        cached_result = cache.get(file_path, "skeleton", **kwargs)
        if cached_result is not None:
            logger.info(f"✅ 캐시 히트: {Path(file_path).name} - 스켈레톤 로드됨")
            return cached_result
        
        # 캐시가 없으면 원본 함수 실행
        logger.info(f"🔄 스켈레톤 추출 중: {Path(file_path).name}")
        result = extractor_func(*args, **kwargs)
        
        # 🔍 결과 캐싱 로깅
        if result is not None:
            cache.set(file_path, "skeleton", result, **kwargs)
            logger.info(f"💾 캐시 저장됨: {Path(file_path).name}")
        else:
            logger.warning(f"⚠️ 빈 결과로 캐시 저장 안됨: {Path(file_path).name}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 캐시 오류 - 직접 실행: {e}")
        # 캐시 오류 시 원본 함수 직접 실행
        return extractor_func(*args, **kwargs)

def cache_ai_prediction(file_path: str, model_path: str, predictor_func: Callable, skeleton, *args, **kwargs):
    """AI 예측 결과 캐싱 (오류 방지 강화)"""
    try:
        cache = get_cache()
        
        # 스켈레톤 해시를 포함한 캐시 키 생성
        skeleton_hash = hashlib.md5(str(skeleton).encode()).hexdigest()[:16]
        model_hash = get_cache()._get_file_hash(model_path) if os.path.exists(model_path) else "no_model"
        
        cache_params = {
            "skeleton_hash": skeleton_hash,
            "model_hash": model_hash,
            **kwargs
        }
        
        # 🔍 캐시 확인 로깅
        logger.info(f"🤖 AI 예측 캐시 확인 중: {Path(file_path).name}")
        cached_result = cache.get(file_path, "ai_prediction", **cache_params)
        if cached_result is not None:
            logger.info(f"✅ AI 예측 캐시 히트: {Path(file_path).name}")
            return cached_result
        
        # 캐시가 없으면 원본 함수 실행
        logger.info(f"🤖 AI 예측 중: {Path(file_path).name}")
        result = predictor_func(skeleton, *args, **kwargs)
        
        # 🔍 결과 캐싱 로깅
        if result is not None:
            cache.set(file_path, "ai_prediction", result, **cache_params)
            logger.info(f"💾 AI 예측 캐시 저장됨: {Path(file_path).name}")
        else:
            logger.warning(f"⚠️ AI 예측 빈 결과로 캐시 저장 안됨: {Path(file_path).name}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ AI 예측 캐시 오류 - 직접 실행: {e}")
        # 캐시 오류 시 원본 함수 직접 실행
        return predictor_func(skeleton, *args, **kwargs)

def cache_road_clipping(district_file: str, road_folder: str, clipper_func: Callable, *args, **kwargs):
    """도로망 클리핑 결과 캐싱"""
    cache = get_cache()
    
    # 도로 폴더의 수정 시간도 고려
    try:
        road_folder_stat = os.stat(road_folder)
        road_folder_hash = hashlib.md5(f"{road_folder}_{road_folder_stat.st_mtime}".encode()).hexdigest()[:16]
    except:
        road_folder_hash = hashlib.md5(road_folder.encode()).hexdigest()[:16]
    
    cache_params = {
        "road_folder_hash": road_folder_hash,
        **kwargs
    }
    
    # 캐시 확인
    cached_result = cache.get(district_file, "clipping", **cache_params)
    if cached_result is not None:
        return cached_result
    
    # 캐시가 없으면 원본 함수 실행
    logger.info(f"✂️ 도로망 클리핑 중: {Path(district_file).name}")
    result = clipper_func(*args, **kwargs)
    
    # 결과 캐싱
    cache.set(district_file, "clipping", result, **cache_params)
    
    return result 