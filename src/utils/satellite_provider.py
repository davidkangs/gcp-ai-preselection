#!/usr/bin/env python3
"""위성영상 제공자 모듈"""
import requests
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SatelliteProvider(ABC):
    """위성영상 제공자 추상 클래스"""
    
    @abstractmethod
    def get_image(self, lon: float, lat: float, width: int, height: int, zoom: int = 16) -> Optional[bytes]:
        """위성영상 이미지 데이터 반환"""
        pass


class MapboxProvider(SatelliteProvider):
    """Mapbox 위성영상 제공자"""
    
    def __init__(self, access_token: str):
        self.access_token = access_token
    
    def get_image(self, lon: float, lat: float, width: int, height: int, zoom: int = 16) -> Optional[bytes]:
        url = (
            f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
            f"{lon},{lat},{zoom}/{width}x{height}@2x"
            f"?access_token={self.access_token}"
        )
        
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.content
        except Exception as e:
            logger.error(f"Mapbox 오류: {e}")
        return None


class GoogleMapsProvider(SatelliteProvider):
    """Google Maps 위성영상 제공자"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def get_image(self, lon: float, lat: float, width: int, height: int, zoom: int = 16) -> Optional[bytes]:
        url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={lat},{lon}&zoom={zoom}&size={width}x{height}"
            f"&maptype=satellite&style=feature:all|element:labels|visibility:off"
            f"&key={self.api_key}"
        )
        
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.content
        except Exception as e:
            logger.error(f"Google Maps 오류: {e}")
        return None


class VWorldProvider(SatelliteProvider):
    """VWorld (국토지리정보원) 위성영상 제공자"""
    
    def __init__(self, api_key: str = "BA5EDB0B-73BE-31C5-B7FB-2310EB2BDC85"):
        self.api_key = api_key
    
    def get_image(self, lon: float, lat: float, width: int, height: int, zoom: int = 17) -> Optional[bytes]:
        # 타일 좌표 계산
        n = 2.0 ** zoom
        x_tile = int((lon + 180.0) / 360.0 * n)
        y_tile = int((1.0 - np.log(np.tan(lat * np.pi / 180.0) + 
                     (1.0 / np.cos(lat * np.pi / 180.0))) / np.pi) / 2.0 * n)
        
        url = (
            f"http://api.vworld.kr/req/wmts/1.0.0/{self.api_key}/"
            f"Satellite/GoogleMapsCompatible/{zoom}/{x_tile}/{y_tile}.jpeg"
        )
        
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.content
        except Exception as e:
            logger.error(f"VWorld 오류: {e}")
        return None


class SatelliteManager:
    """위성영상 관리자"""
    
    def __init__(self):
        self.providers = {}
        self.current_provider = None
        
        # 기본 제공자 등록
        self.register_provider('vworld', VWorldProvider())
    
    def register_provider(self, name: str, provider: SatelliteProvider):
        """제공자 등록"""
        self.providers[name] = provider
        if self.current_provider is None:
            self.current_provider = name
    
    def set_provider(self, name: str):
        """현재 제공자 설정"""
        if name in self.providers:
            self.current_provider = name
            return True
        return False
    
    def get_image(self, lon: float, lat: float, width: int, height: int, zoom: int = 16) -> Optional[bytes]:
        """현재 제공자로부터 이미지 가져오기"""
        if self.current_provider and self.current_provider in self.providers:
            provider = self.providers[self.current_provider]
            return provider.get_image(lon, lat, width, height, zoom)
        return None
    
    def convert_coordinates(self, x: float, y: float, from_crs: str = 'EPSG:5186') -> Tuple[float, float]:
        """좌표 변환 (TM → WGS84)"""
        try:
            import pyproj
            transformer = pyproj.Transformer.from_crs(from_crs, 'EPSG:4326', always_xy=True)
            return transformer.transform(x, y)
        except:
            # pyproj 없으면 근사 변환 (대전 기준)
            if abs(x) > 180:  # TM 좌표
                lon = 127.385 + (x - 200000) * 0.00001
                lat = 36.351 + (y - 500000) * 0.00001
                return lon, lat
            return x, y


# 싱글톤 인스턴스
satellite_manager = SatelliteManager()