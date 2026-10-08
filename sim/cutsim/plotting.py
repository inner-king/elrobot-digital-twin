"""matplotlib 공통 설정: 한글 폰트가 있으면 쓴다."""
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt  # noqa: E402

for _name in ("NanumSquare", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK TC"):
    if any(f.name == _name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _name
        break
plt.rcParams["axes.unicode_minus"] = False
