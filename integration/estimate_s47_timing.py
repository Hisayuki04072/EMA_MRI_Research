"""F1音声とs47音声をDTWで対応付け、F1のTextGrid(phoneticタイヤ=IPA)の
音素境界をs47のタイミングに変換する。

出力: s47/by_sentence/<id>/<id>_s47_timing.lab
      (start_sec end_sec symbol の3列、tab区切り)

対象タイヤ: item[2] "phonetic"(=06_Ipa-EMAのdata/_dict.pyと一致するIPA, 97記号)
"""
import os
import sys
import glob
import argparse

import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

sys.path.insert(0, os.path.dirname(__file__))
from textgrid import load_textgrid, Interval, write_lab

S47_ROOT = "/workspace/s47/by_sentence"
NORMALIZED_TEXT_DIR = "/workspace/EMA_MRI_Research/shuushi/Dataset_s47/text"
PHONETIC_TIER_IDX = 1  # item[2] (0-indexed) = "phonetic"

DTW_SR = 16000
DTW_HOP = 160  # 100fps相当。DTW計算用の特徴量レート(最終タイミングは秒単位なので解像度に依存しない)
N_MFCC = 20


def extract_mfcc(wav_path, sr=DTW_SR, hop=DTW_HOP, n_mfcc=N_MFCC):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=n_mfcc)
    return mfcc.T  # (T, n_mfcc)


def build_f1_to_s47_mapper(f1_wav, s47_wav):
    """F1の時刻(秒)をs47の時刻(秒)へ変換する関数を返す。"""
    mfcc_f1 = extract_mfcc(f1_wav)
    mfcc_s47 = extract_mfcc(s47_wav)

    _, path = fastdtw(mfcc_f1, mfcc_s47, dist=euclidean)
    path = np.array(path)  # (N, 2): (f1_frame_idx, s47_frame_idx)

    # f1フレームごとに対応するs47フレーム(複数ある場合は平均)を取る
    f1_idx_unique, inv = np.unique(path[:, 0], return_inverse=True)
    s47_idx_mean = np.array([path[inv == i, 1].mean() for i in range(len(f1_idx_unique))])

    f1_times = f1_idx_unique * DTW_HOP / DTW_SR
    s47_times = s47_idx_mean * DTW_HOP / DTW_SR

    def mapper(t_f1):
        return float(np.interp(t_f1, f1_times, s47_times))

    return mapper


def process_sentence(sent_id, verbose=False):
    sent_dir = os.path.join(S47_ROOT, sent_id)
    tg_path = os.path.join(NORMALIZED_TEXT_DIR, f"{sent_id}.TextGrid")  # 正規化済み(shuushiのipa_dictと一致)
    f1_wav = os.path.join(sent_dir, f"{sent_id}_F1.WAV")
    s47_wav = os.path.join(sent_dir, f"{sent_id}_s47.WAV")
    out_path = os.path.join(sent_dir, f"{sent_id}_s47_timing.lab")

    tiers = load_textgrid(tg_path)
    phonetic = tiers[PHONETIC_TIER_IDX]

    mapper = build_f1_to_s47_mapper(f1_wav, s47_wav)

    out_intervals = []
    for iv in phonetic.intervals:
        new_xmin = mapper(iv.xmin)
        new_xmax = mapper(iv.xmax)
        out_intervals.append(Interval(new_xmin, new_xmax, iv.text))

    # 単調増加を保証(DTWの丸め誤差で逆転した場合に補正)
    for i in range(1, len(out_intervals)):
        if out_intervals[i].xmin < out_intervals[i - 1].xmax:
            out_intervals[i].xmin = out_intervals[i - 1].xmax
        if out_intervals[i].xmax < out_intervals[i].xmin:
            out_intervals[i].xmax = out_intervals[i].xmin

    write_lab(out_path, out_intervals)

    if verbose:
        s47_dur = librosa.get_duration(path=s47_wav)
        print(f"{sent_id}: tier intervals={len(out_intervals)}  "
              f"mapped_end={out_intervals[-1].xmax:.3f}s  s47_total={s47_dur:.3f}s")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", default=None, help="処理する文IDを指定(省略時は全503文)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.ids:
        sent_ids = args.ids
    else:
        sent_ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(S47_ROOT, "*")) if os.path.isdir(p))

    for sent_id in sent_ids:
        try:
            process_sentence(sent_id, verbose=args.verbose)
        except Exception as e:
            print(f"ERROR {sent_id}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
