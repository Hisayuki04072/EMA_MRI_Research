"""08用: Dataset_s47のcut_sil_wav/cut_sil_emaからMEL・ENERGYを生成する。
shuushi/01_PreProcess/PreProcess_EMA_wav_to_mel/wav_to_mel.pyのmain()を
set_s47.yaml向けに呼び出すだけ(スクリプト自体は無変更)。
"""
import sys

sys.path.insert(0, "/workspace/EMA_MRI_Research/shuushi/01_PreProcess/PreProcess_EMA_wav_to_mel")
from wav_to_mel import main

if __name__ == "__main__":
    main("/workspace/EMA_MRI_Research/shuushi/01_PreProcess/PreProcess_EMA_wav_to_mel/set_s47.yaml")
