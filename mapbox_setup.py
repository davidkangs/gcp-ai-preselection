#!/usr/bin/env python3
"""위성영상 디버깅 패치"""
from pathlib import Path
import re

canvas_file = Path('src/ui/canvas_widget.py')
content = canvas_file.read_text(encoding='utf-8')

# load_satellite_background 메서드 디버깅 버전
debug_method = '''    def load_satellite_background(self):
        """위성영상 배경 로드 - 디버깅 버전"""
        try:
            if not self.scene.items():
                print("⚠️ Scene이 비어있음")
                return
                
            # 뷰포트 크기
            w = min(512, self.viewport().width())
            h = min(512, self.viewport().height())
            print(f"📐 뷰포트 크기: {w}x{h}")
            
            # 현재 보이는 영역의 중심
            view_rect = self.mapToScene(self.viewport().rect()).boundingRect()
            center_x = view_rect.center().x()
            center_y = view_rect.center().y()
            print(f"📍 Scene 중심 좌표: x={center_x:.2f}, y={center_y:.2f}")
            
            # CRS 확인
            print(f"🗺️ 원본 CRS: {self.original_crs}")
            
            # 좌표 변환
            if abs(center_x) > 180:  # TM 좌표
                print("🔄 TM 좌표 → WGS84 변환 필요")
                try:
                    import pyproj
                    transformer = pyproj.Transformer.from_crs('EPSG:5186', 'EPSG:4326', always_xy=True)
                    lon, lat = transformer.transform(center_x, center_y)
                    print(f"✅ pyproj 변환 성공: {lon:.4f}, {lat:.4f}")
                except Exception as e:
                    print(f"❌ pyproj 실패: {e}")
                    # 근사 변환
                    lon = 127.385 + (center_x - 200000) * 0.00001
                    lat = 36.351 + (center_y - 500000) * 0.00001
                    print(f"🔧 근사 변환: {lon:.4f}, {lat:.4f}")
            else:
                lon, lat = center_x, center_y
                print(f"✅ WGS84 좌표 그대로 사용: {lon:.4f}, {lat:.4f}")
            
            # 범위 확인
            if not (120 < lon < 135 and 30 < lat < 45):
                print(f"⚠️ 한국 범위 벗어남: {lon:.4f}, {lat:.4f}")
                # 강제로 대전 좌표 사용
                lon, lat = 127.385, 36.351
                print(f"🔧 대전 좌표로 대체: {lon:.4f}, {lat:.4f}")
            
            # Mapbox API
            zoom = 15
            token = "pk.eyJ1Ijoia2FuZ2RhZXJpIiwiYSI6ImNtY2FtbTQyODA1Y2Iybm9ybmlhbTZrbDUifQ.dwjb3fq0FqvXDx6-OuLYHw"
            
            url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom},0,0/{w}x{h}?access_token={token}"
            print(f"🌐 Mapbox URL: {url[:100]}...")
            
            resp = requests.get(url, timeout=10)
            print(f"📡 응답 상태: {resp.status_code}")
            
            if resp.status_code == 200:
                print(f"📦 이미지 크기: {len(resp.content)} bytes")
                img = QImage()
                if img.loadFromData(resp.content):
                    print(f"✅ QImage 로드 성공: {img.width()}x{img.height()}")
                    
                    # 스케일링 생략하고 바로 사용
                    pix = QPixmap.fromImage(img)
                    
                    # 투명도
                    if self.satellite_opacity < 1.0:
                        painter = QPainter(pix)
                        painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
                        painter.fillRect(pix.rect(), QColor(0, 0, 0, int(255 * self.satellite_opacity)))
                        painter.end()
                    
                    self.setBackgroundBrush(QBrush(pix))
                    print("✅ 배경 설정 완료!")
                else:
                    print("❌ QImage 로드 실패")
            else:
                print(f"❌ Mapbox 오류: {resp.status_code}")
                print(f"응답: {resp.text[:200]}")
                
        except Exception as e:
            print(f"❌ 예외 발생: {e}")
            import traceback
            traceback.print_exc()'''

# 메서드 교체
pattern = r'def load_satellite_background\(self\):.*?(?=\n    def|\nclass|\Z)'
content = re.sub(pattern, debug_method, content, flags=re.DOTALL)

canvas_file.write_text(content, encoding='utf-8')
print("✓ 디버깅 패치 적용 완료")
print("\n이제 프로그램 실행하고 위성영상 체크박스 체크하면")
print("콘솔에 자세한 디버깅 정보가 출력됩니다!")