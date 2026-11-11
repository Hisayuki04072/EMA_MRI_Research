"""s47のTextGrid(item[2]="phonetic")の記号表記を、F1側のipa_dict(06/08が使う辞書)に
存在する表記へ正規化し、Dataset_s47/text/ に実ファイルとして書き出す。

変換内容:
  - IPA長音記号(U+02D0 'ː')付きの記号 → 長音記号を外し、区間を2分割して同じ記号を2回連続させる
    (F1側のtier2は長音・促音を単一の長音記号ではなく同一記号の連続区間で表現しているため)
  - 結束線(U+0361)付きの破擦音 → 合字1文字に正規化 (d͡ʑ→ʥ, t͡s→ʦ, t͡ɕ→ʨ)
  - その他の未登録記号の個別正規化 (ɡ→g, y→j, çʲ→ç, ɰ̃→ɰ, i̥→i, ɯ̥→ɯ)
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(__file__))
from textgrid import load_textgrid, write_textgrid, Interval

S47_ROOT = "/workspace/s47/by_sentence"
OUT_TEXT_DIR = "/workspace/EMA_MRI_Research/shuushi/Dataset_s47/text"

LENGTH_MARK = "ː"  # ː
TIE_BAR = "͡"      # combining double inverted breve

SIMPLE_SUB = {
    "ɡ": "g",       # ɡ (U+0261) -> g (ASCII)
    "y": "j",
    "çʲ": "ç",  # çʲ -> ç
    "ɰ̃": "ɰ",  # ɰ̃ -> ɰ
    "i̥": "i",       # i̥ -> i
    "ɯ̥": "ɯ",  # ɯ̥ -> ɯ
}

TIE_SUB = {
    "d" + TIE_BAR + "ʑ": "ʥ",   # d͡ʑ -> ʥ
    "t" + TIE_BAR + "s": "ʦ",        # t͡s -> ʦ
    "t" + TIE_BAR + "ɕ": "ʨ",   # t͡ɕ -> ʨ
}


def normalize_interval(iv):
    """1区間を正規化する。長音の場合は2つのIntervalに分割して返す。"""
    text = iv.text

    for src, dst in TIE_SUB.items():
        if text == src:
            text = dst

    for src, dst in SIMPLE_SUB.items():
        if text == src:
            text = dst

    if LENGTH_MARK in text:
        base = text.replace(LENGTH_MARK, "")
        mid = (iv.xmin + iv.xmax) / 2
        return [Interval(iv.xmin, mid, base), Interval(mid, iv.xmax, base)]

    return [Interval(iv.xmin, iv.xmax, text)]


def process_sentence(sent_id):
    tg_path = os.path.join(S47_ROOT, sent_id, f"{sent_id}.TextGrid")
    tiers = load_textgrid(tg_path)

    new_intervals = []
    for iv in tiers[1].intervals:
        new_intervals.extend(normalize_interval(iv))
    tiers[1].intervals = new_intervals

    out_path = os.path.join(OUT_TEXT_DIR, f"{sent_id}.TextGrid")
    write_textgrid(out_path, tiers)
    return out_path


def main():
    sent_ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(S47_ROOT, "*")) if os.path.isdir(p))
    for sent_id in sent_ids:
        try:
            process_sentence(sent_id)
        except Exception as e:
            print(f"ERROR {sent_id}: {e}", file=sys.stderr)
    print(f"done: {len(sent_ids)} sentences")


if __name__ == "__main__":
    main()
