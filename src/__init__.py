"""
AI Survey Control Point Pre-Selection System
GCP 강화학습 기반 지적측량기준점 AI 선점 시스템

메인 패키지 모듈
"""

__version__ = "1.0.0"
__author__ = "강상우 (ksw3037@lx.or.kr)"
__description__ = "GCP 강화학습 기반 지적측량기준점 AI 선점 시스템"

# 패키지 정보
PACKAGE_INFO = {
    "name": "AI Survey Control Point Pre-Selection",
    "version": __version__,
    "author": __author__,
    "description": __description__,
    "python_requires": ">=3.11.4",
    "license": "MIT"
}

def get_version():
    """패키지 버전 반환"""
    return __version__

def get_package_info():
    """패키지 정보 반환"""
    return PACKAGE_INFO.copy()