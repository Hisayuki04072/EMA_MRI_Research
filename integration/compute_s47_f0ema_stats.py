"""08用: Dataset_s47のcut_sil_ema・F0・ENERGYから、
F0EMAIPA_to_MEL_Dataset.pyがハードコードしているMAX_EMA/MIN_EMA/MAX_F0/MIN_F0/
MAX_ENE/MIN_ENEをs47データに合わせて再計算し、
F0EMAIPA_to_MEL_Dataset_s47.py内の該当定数を書き換える。
"""
import glob
import os
import re

import numpy as np

DATASET_S47 = "/workspace/EMA_MRI_Research/shuushi/Dataset_s47"
TARGET_FILE = "/workspace/EMA_MRI_Research/shuushi/08_Generate_MEL/F0EMAIPA_to_MEL_Dataset_s47.py"


def read_ema_text(filename):
    result = []
    with open(filename, "r") as f:
        for s in f.readlines():
            s = s.strip().split(" ")
            s = list(map(float, s))
            data = [s[i:i + 2] for i in range(0, len(s), 2)]
            result.append(data)
    return np.array(result)  # (T, 6, 2)


def main():
    ema_files = sorted(glob.glob(os.path.join(DATASET_S47, "cut_sil_ema", "*.ema")))
    f0_files = sorted(glob.glob(os.path.join(DATASET_S47, "F0", "*.npy")))
    ene_files = sorted(glob.glob(os.path.join(DATASET_S47, "ENERGY", "*.npy")))

    all_ema = np.concatenate([read_ema_text(f) for f in ema_files], axis=0)  # (sum_T, 6, 2)
    all_f0 = np.concatenate([np.load(f).reshape(-1) for f in f0_files])
    all_ene = np.concatenate([np.load(f).reshape(-1) for f in ene_files])

    max_ema = all_ema.max(axis=0)  # (6,2)
    min_ema = all_ema.min(axis=0)
    max_f0, min_f0 = float(all_f0.max()), float(all_f0.min())
    max_ene, min_ene = float(all_ene.max()), float(all_ene.min())

    print("MAX_EMA\n", max_ema)
    print("MIN_EMA\n", min_ema)
    print("MAX_F0/MIN_F0", max_f0, min_f0)
    print("MAX_ENE/MIN_ENE", max_ene, min_ene)

    text = open(TARGET_FILE, encoding="utf-8").read()

    max_ema_str = "torch.Tensor(" + repr(max_ema.tolist()) + ")"
    min_ema_str = "torch.Tensor(" + repr(min_ema.tolist()) + ")"

    text = re.sub(r"MAX_EMA = torch\.Tensor\(\[.*?\]\)", "MAX_EMA = " + max_ema_str, text, flags=re.S, count=1)
    text = re.sub(r"MIN_EMA = torch\.Tensor\(\[.*?\]\)", "MIN_EMA = " + min_ema_str, text, flags=re.S, count=1)
    text = re.sub(r"MAX_F0 = [-\d.]+", f"MAX_F0 = {max_f0}", text, count=1)
    text = re.sub(r"MIN_F0 = [-\d.]+", f"MIN_F0 = {min_f0}", text, count=1)
    text = re.sub(r"MAX_ENE = [-\d.]+", f"MAX_ENE = {max_ene}", text, count=1)
    text = re.sub(r"MIN_ENE = [-\d.]+", f"MIN_ENE = {min_ene}", text, count=1)

    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"updated: {TARGET_FILE}")


if __name__ == "__main__":
    main()
