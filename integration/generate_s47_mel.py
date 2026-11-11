"""s47音声からメルスペクトログラムを生成する。

shuushi(01_PreProcess/PreProcess_EMA_wav_to_mel/wav_to_mel.py)と全く同じ
mel_spectrogram実装を使い、HiFi-GAN/shuushiの設定(16kHz・hop=64・60mel)に揃える。
(EMAデータへの長さ合わせパディングは行わない。s47には対応するEMAが無いため)

出力: s47/by_sentence/<id>/<id>_s47_mel.npy  (shape: [num_mels, T])
"""
import os
import sys
import glob
import argparse

import numpy as np
import torch
import soxr
from scipy.io.wavfile import read as wavread
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn

S47_ROOT = "/workspace/s47/by_sentence"
MAX_WAV_VALUE = 32768.0

# shuushi/01_PreProcess/PreProcess_EMA_wav_to_mel/set.yaml と同じ設定
NUM_MELS = 60
N_FFT = 512
HOP_SIZE = 64
WIN_SIZE = 512
SAMPLING_RATE = 16000
FMIN = 0
FMAX = 8000

_mel_basis = None
_hann_window = None


def _get_filters():
    global _mel_basis, _hann_window
    if _mel_basis is None:
        mel = librosa_mel_fn(sr=SAMPLING_RATE, n_fft=N_FFT, n_mels=NUM_MELS, fmin=FMIN, fmax=FMAX)
        _mel_basis = torch.from_numpy(mel).float()
        _hann_window = torch.hann_window(WIN_SIZE)
    return _mel_basis, _hann_window


def mel_spectrogram(y):
    """shuushi/wav_to_mel.pyのmel_spectrogramと同一実装(center=False)。"""
    mel_basis, hann_window = _get_filters()
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((N_FFT - HOP_SIZE) / 2), int((N_FFT - HOP_SIZE) / 2)),
                                mode='reflect')
    y = y.squeeze(1)
    spec = torch.stft(y, N_FFT, hop_length=HOP_SIZE, win_length=WIN_SIZE, window=hann_window,
                      center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    return mel_spec


def read_wav_resampled(wav_path, target_sr=SAMPLING_RATE):
    sr, wave = wavread(wav_path)
    wave = wave.astype(np.float64) / MAX_WAV_VALUE
    if sr != target_sr:
        wave = soxr.resample(wave, sr, target_sr, 'VHQ')
    wave = normalize(wave) * 0.95
    return torch.tensor(wave, dtype=torch.float32)


def process_sentence(sent_id, verbose=False):
    sent_dir = os.path.join(S47_ROOT, sent_id)
    wav_path = os.path.join(sent_dir, f"{sent_id}_s47.WAV")
    out_path = os.path.join(sent_dir, f"{sent_id}_s47_mel.npy")

    audio = read_wav_resampled(wav_path).unsqueeze(0)
    mel = mel_spectrogram(audio).squeeze(0).numpy()  # (num_mels, T)
    np.save(out_path, mel)

    if verbose:
        print(f"{sent_id}: mel shape={mel.shape}  duration={mel.shape[1] * HOP_SIZE / SAMPLING_RATE:.3f}s")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", default=None)
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
