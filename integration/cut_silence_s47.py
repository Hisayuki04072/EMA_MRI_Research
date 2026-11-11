"""Dataset_s47のema/wav/textから無音区間(文頭silB・文末silE)を除去し、
08_Generate_MELが読む形式(cut_sil_ema: テキスト形式, cut_sil_wav: wav)で保存する。
shuushi/01_PreProcess/PreProcess_cut_silence/cut_silence.pyと同じ処理内容。
"""
import os
import sys
import glob

import torchaudio

sys.path.insert(0, "/workspace/EMA_MRI_Research/shuushi/06_Ipa-EMA/maruyama_and_iiyama")
from common.class_ema import EMA
from common.class_ipa import IPA

DATASET_S47 = "/workspace/EMA_MRI_Research/shuushi/Dataset_s47"


def cut_ema(ema_data, start, end, ema_hz=250):
    return ema_data[int(start * ema_hz):int(end * ema_hz)]


def cut_wav(wav_file, start, end):
    waveform, sample_rate = torchaudio.load(wav_file)
    return waveform[:, int(start * sample_rate):int(end * sample_rate)], sample_rate


def save_ema(ema_data, out_filename):
    with open(out_filename, mode="w") as f:
        for t in range(len(ema_data)):
            for j in range(6):
                for i in range(2):
                    f.write("%f " % (ema_data[t, j, i]))
            f.write("\n")


def main():
    out_ema_dir = os.path.join(DATASET_S47, "cut_sil_ema")
    out_wav_dir = os.path.join(DATASET_S47, "cut_sil_wav")
    os.makedirs(out_ema_dir, exist_ok=True)
    os.makedirs(out_wav_dir, exist_ok=True)

    ema_files = sorted(glob.glob(os.path.join(DATASET_S47, "ema", "*.ema")))
    for ema_file in ema_files:
        basename = os.path.splitext(os.path.basename(ema_file))[0]
        wav_file = os.path.join(DATASET_S47, "wav", basename + ".WAV")
        ipa_file = os.path.join(DATASET_S47, "text", basename + ".TextGrid")

        try:
            ipa_pair = IPA(ipa_file).ipa_pair
            ema_data = EMA(ema_file).ema_data
            start, end = ipa_pair[0][1][1], ipa_pair[-1][1][0]

            ema_cut = cut_ema(ema_data, start, end)
            wav_cut, sr = cut_wav(wav_file, start, end)

            save_ema(ema_cut, os.path.join(out_ema_dir, basename + ".ema"))
            torchaudio.save(os.path.join(out_wav_dir, basename + ".wav"), src=wav_cut, sample_rate=sr)
        except Exception as e:
            print(f"ERROR {basename}: {e}", file=sys.stderr)

    print(f"done: {len(ema_files)} files")


if __name__ == "__main__":
    main()
