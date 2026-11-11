"""Minimal Praat TextGrid parser.

shuushi/04_Ph-Ipa/iiyama/TextGrid.py を元にした最小実装。
intervals: size=0 (空のtier)も扱えるよう修正済み。
"""
import codecs


class Interval:
    __slots__ = ("xmin", "xmax", "text")

    def __init__(self, xmin=0.0, xmax=0.0, text="?"):
        self.xmin = xmin
        self.xmax = xmax
        self.text = text

    def __repr__(self):
        return f"Interval({self.text!r}, {self.xmin:.4f}, {self.xmax:.4f})"


class Tier:
    def __init__(self):
        self.name = None
        self.xmin = 0.0
        self.xmax = 0.0
        self.intervals = []

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, idx):
        return self.intervals[idx]


def _read_text(filename):
    data = open(filename, "rb").read()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    return data.decode("utf-8", errors="ignore")


def load_textgrid(filename):
    """TextGridファイルを読み込み、Tierのリストを返す。"""
    text = _read_text(filename)
    lines = [s.strip() for s in text.splitlines()]

    tiers = []
    num_items = 0
    item_idx = 0
    tier = None
    interval = None
    interval_idx = 0
    num_intervals = None

    for s in lines:
        if s.startswith("size = "):
            num_items = int(s[len("size = "):])
            tiers = [Tier() for _ in range(num_items)]
        elif s == f"item [{item_idx + 1}]:":
            tier = tiers[item_idx]
            item_idx += 1
            interval_idx = 0
            interval = None
            num_intervals = None
        elif num_intervals is not None and s == f"intervals [{interval_idx + 1}]:":
            interval = Interval()
            tier.intervals.append(interval)
            interval_idx += 1
        elif tier is not None and interval is None:
            if s.startswith("intervals: size = "):
                num_intervals = int(s[len("intervals: size = "):])
            elif s.startswith("xmin = "):
                tier.xmin = float(s[len("xmin = "):])
            elif s.startswith("xmax = "):
                tier.xmax = float(s[len("xmax = "):])
            elif s.startswith("name = "):
                tier.name = s[s.index('"') + 1: s.rindex('"')]
        elif tier is not None and interval is not None:
            if s.startswith("xmin = "):
                interval.xmin = float(s[len("xmin = "):])
            elif s.startswith("xmax = "):
                interval.xmax = float(s[len("xmax = "):])
            elif s.startswith("text = "):
                interval.text = s[s.index('"') + 1: s.rindex('"')]
    return tiers


def write_lab(path, intervals):
    """(xmin, xmax, text) の並びを `start end text` 形式で書き出す。"""
    with codecs.open(path, "w", "utf-8") as f:
        for iv in intervals:
            f.write(f"{iv.xmin:.6f} {iv.xmax:.6f} {iv.text}\n")
