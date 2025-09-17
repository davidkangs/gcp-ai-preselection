# apply_patch.py
from pathlib import Path
import re

# 1) 패치 대상 파일 경로
file_path = Path(__file__).parent / "src" / "ui" / "canvas_widget.py"
text = file_path.read_text(encoding="utf-8")

# 2) QTransform 임포트 추가
if "QTransform" not in text:
    text = re.sub(
        r"(from PyQt5\.QtGui import \([\s\S]*?)(QPainter, QPixmap, QImage)",
        r"\1QTransform, \2",
        text,
        flags=re.MULTILINE
    )

# 3) InteractiveCanvas.__init__ 안에 색상 팔레트 정의 삽입
text = re.sub(
    r"(self\.show_points_layer\s*=\s*True\s*)",
    r"""\1
        # Color palette
        self.colors = {
            'road':           QColor(200,200,200),
            'road_stroke':    QColor(150,150,150),
            'skeleton':       QColor(50,50,200),
            'intersection':   QColor(255,0,0),
            'curve':          QColor(0,0,255),
            'endpoint':       QColor(0,255,0),
            'ai_intersection':QColor(128,0,128),
            'ai_curve':       QColor(255,165,0),
            'ai_endpoint':    QColor(139,69,19),
        }
""",
    text
)

# 4) 위성영상 관련 메서드들 삽입
satellite_methods = """
    def set_satellite_opacity(self, opacity):
        \"\"\"위성영상 투명도 설정\"\"\"
        self.satellite_opacity = opacity / 100.0
        for item in self.tile_items.values():
            item.setOpacity(self.satellite_opacity)

    def toggle_satellite_layer(self, visible):
        \"\"\"위성영상 레이어 표시/숨김\"\"\"
        self.show_satellite_layer = visible
        for item in self.tile_items.values():
            item.setVisible(visible)
        if visible:
            self.load_satellite_tiles()

    def load_satellite_tiles(self):
        \"\"\"현재 뷰포트에 필요한 위성 타일 로드\"\"\"
        if not self.show_satellite_layer:
            return
        for i in range(5):
            for j in range(5):
                key = f"15_{i}_{j}"
                if key not in self.tile_items:
                    loader = TileLoader(i, j, 15)
                    loader.tile_loaded.connect(self.on_tile_loaded)
                    loader.start()
                    self.tile_loaders.append(loader)

    def on_tile_loaded(self, x, y, z, pixmap, lon_min, lat_min, lon_max, lat_max):
        \"\"\"타일 로드 완료 시\"\"\"
        key = f"{z}_{x}_{y}"
        item = self.scene.addPixmap(pixmap)
        item.setZValue(-10)
        item.setOpacity(self.satellite_opacity)
        # 위성 타일을 경위도로 매핑
        w, h = pixmap.width(), pixmap.height()
        sx = (lon_max - lon_min) / w
        sy = (lat_min - lat_max) / h
        transform = QTransform().translate(lon_min, lat_max).scale(sx, sy)
        item.setTransform(transform)
        item.setVisible(self.show_satellite_layer)
        self.tile_items[key] = item
"""

# "__init__" 끝난 직후에 삽입 (표식 주석 찾아서)
text = text.replace(
    "        # InteractiveCanvas 클래스 안에 추가",
    "        # InteractiveCanvas 클래스 안에 추가" + satellite_methods
)

# 5) 파일 덮어쓰기
file_path.write_text(text, encoding="utf-8")
print("canvas_widget.py 패치 완료!")
