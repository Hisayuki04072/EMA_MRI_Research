"""s47(Dataset_s47)のEMAデータから、06_Ipa-EMAが使う正規化パラメータ
(mean_param, std_param, normalize_param, max_param, min_param, ema_max_duration)
を再計算し、data/_dict_s47.py として書き出す。

06既存コード(IPA_AND_EMA_DATASET.cut_silence / get_params)をそのまま再利用し、
計算方法自体はF1用のdata/_dict.pyと完全に同じにする(データが違うだけ)。
"""
import os
import sys

sys.path.insert(0, "/workspace/EMA_MRI_Research/shuushi/06_Ipa-EMA/maruyama_and_iiyama")

import torch
from IPA_AND_EMA_DATASET import IPA_EMA_DATASET
from train import read_txtfile, make_filepath

DATASET_S47 = "/workspace/EMA_MRI_Research/shuushi/Dataset_s47"
OUT_PATH = "/workspace/EMA_MRI_Research/shuushi/06_Ipa-EMA/maruyama_and_iiyama/data/_dict_s47.py"


def main():
    text_dataset = os.path.join(DATASET_S47, "text")
    ema_dataset = os.path.join(DATASET_S47, "ema")

    all_basenames = []
    for split in ("train.txt", "valid.txt", "test.txt"):
        all_basenames += read_txtfile(os.path.join(DATASET_S47, "set_textgrid", split))

    ipa_files = make_filepath(text_dataset, all_basenames)
    ema_files = make_filepath(ema_dataset, [b.replace(".TextGrid", ".ema") for b in all_basenames])
    files = list(zip(ipa_files, ema_files))

    print(f"loading {len(files)} files to compute stats...")
    dataset = IPA_EMA_DATASET(files, pad=False)  # pad=FalseでEMA長をそのまま保持(pad_sequencesの切り詰めを回避)

    mean, std, _max, _min = dataset.get_params(dataset.voiced_ema)
    ema_max_duration = max(dataset.emaT_per_file)
    # 安全マージン(今後fine-tuning等で多少長い発話が出ても切り詰められないように)
    ema_max_duration_padded = int(((ema_max_duration + 50) // 50) * 50)

    # normalize_param: restrict()での クリップ(±1)にほぼかからないよう、
    # 元のdata/_dict.pyと同じ考え方(標準化後の最大絶対値相当)で算出する。
    normalize_param = torch.max(torch.abs((torch.cat(dataset.voiced_ema, dim=0) - mean) / std), dim=0)[0]

    print("mean:\n", mean)
    print("std:\n", std)
    print("max:\n", _max)
    print("min:\n", _min)
    print("normalize_param:\n", normalize_param)
    print("ema_max_duration (raw/padded):", ema_max_duration, ema_max_duration_padded)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("import torch\n")
        f.write("from data._dict import ipa_dict, position_dict\n\n")
        f.write(f"mean_param = torch.Tensor({mean.tolist()})\n\n")
        f.write(f"std_param = torch.Tensor({std.tolist()})\n\n")
        f.write(f"normalize_param = torch.Tensor({normalize_param.tolist()})\n\n")
        f.write(f"max_param = torch.Tensor({_max.tolist()})\n\n")
        f.write(f"min_param = torch.Tensor({_min.tolist()})\n\n")
        f.write(f"ema_max_duration = {ema_max_duration_padded}\n")

    print(f"written: {OUT_PATH}")


if __name__ == "__main__":
    main()
