#!/usr/bin/env python3
"""위성영상 API 설정"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.satellite_provider import satellite_manager, MapboxProvider, GoogleMapsProvider

# API 키 설정
MAPBOX_TOKEN = "pk.eyJ1Ijoia2FuZ2RhZXJpIiwiYSI6ImNtY2FtbTQyODA1Y2Iybm9ybmlhbTZrbDUifQ.dwjb3fq0FqvXDx6-OuLYHw"  # Mapbox Access Token
GOOGLE_API_KEY = ""  # Google Maps API Key

def setup_providers():
    """위성영상 제공자 설정"""
    
    # Mapbox 설정
    if MAPBOX_TOKEN:
        satellite_manager.register_provider('mapbox', MapboxProvider(MAPBOX_TOKEN))
        satellite_manager.set_provider('mapbox')
        print("✓ Mapbox 제공자 등록됨")
    
    # Google Maps 설정
    if GOOGLE_API_KEY:
        satellite_manager.register_provider('google', GoogleMapsProvider(GOOGLE_API_KEY))
        if not MAPBOX_TOKEN:  # Mapbox가 없으면 Google을 기본으로
            satellite_manager.set_provider('google')
        print("✓ Google Maps 제공자 등록됨")
    
    # 기본값은 VWorld (API 키 필요 없음)
    if not MAPBOX_TOKEN and not GOOGLE_API_KEY:
        print("✓ VWorld (기본) 제공자 사용 중")
    
    print(f"현재 제공자: {satellite_manager.current_provider}")

if __name__ == "__main__":
    print("=== 위성영상 API 설정 ===")
    print("\n1. API 키 받기:")
    print("   - Mapbox: https://www.mapbox.com/signup")
    print("   - Google: https://console.cloud.google.com")
    print("\n2. 이 파일의 MAPBOX_TOKEN 또는 GOOGLE_API_KEY 변수에 키 입력")
    print("\n3. setup_providers() 함수가 자동으로 실행됨")
    
    setup_providers()