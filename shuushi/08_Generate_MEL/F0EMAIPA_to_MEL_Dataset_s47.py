import sys
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as data
import torch
import glob
import os


import librosa
from librosa.filters import mel as librosa_mel_fn
import math
import warnings
import random
import json
from _dict import ipa_dict

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

MAX_EMA = torch.Tensor([[-62.459999, -81.164001], [-17.694, -86.867996], [-18.445, -121.110001], [-19.77, -68.242996], [-12.984, 46.014], [-0.75219, -58.324001]])

MIN_EMA = torch.Tensor([[-73.209999, -90.828003], [-107.449997, -126.300003], [-45.814999, -143.919998], [-47.787998, -104.629997], [-95.711998, -94.705002], [-32.743999, -85.872002]])
MAX_F0 = 426.8289354671351
MIN_F0 = 0.0

MAX_ENE = 88.39842987060547
MIN_ENE = 13.44694709777832


"""
GPUに乗せる
"""
def change_to_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e

random.seed(0)


class F0EMAIPA2Mel_Dataset(Dataset):
    def __init__(self,files,batch_size,mode_lst):
        #[(ema_filename,f0_filename,ene_filename,mel_filename)...]
        self.files = files 
        self.split = False if batch_size==1 else True
        self.mode_lst = mode_lst

        self.max_f0_value = change_to_gpu(torch.tensor(MAX_F0, dtype=torch.float32))
        self.min_f0_value =  change_to_gpu(torch.tensor(MIN_F0, dtype=torch.float32))
        self.max_ema_value = change_to_gpu(MAX_EMA)
        self.min_ema_value =change_to_gpu(MIN_EMA)
        self.max_ene_value =  change_to_gpu(torch.tensor(MAX_ENE, dtype=torch.float32))
        self.min_ene_value = change_to_gpu(torch.tensor(MIN_ENE, dtype=torch.float32))


        self.ema_hz = 250
        #最小値が370
        self.segment_size=500

        self.file_num = len(files)

        #クラスの配列を作成[(basename,ema_class,f0_class,ene_class,mel_class,ipa_class)]
        self.class_ary=self.make_class_ary()
        
        #[(basename,input_dim,f0,ene,mel)]
        self.dataset_ary=self.make_dataset_ary()
       

        
    def __getitem__(self, index):
        basename,input_dim_lst,f0_lst,ene_lst,mel_lst = self.dataset_ary[index]

        if self.split:
            if input_dim_lst.shape[0] >= self.segment_size:
                max_len_start = input_dim_lst.shape[0] - self.segment_size
                len_start = random.randint(0,max_len_start)

                input_dim = input_dim_lst[len_start:len_start+self.segment_size]
                f0 = f0_lst[len_start:len_start+self.segment_size]
                ene = ene_lst[len_start:len_start+self.segment_size]
                mel = mel_lst[len_start:len_start+self.segment_size]
            else:
                f0 = torch.nn.functional.pad(f0_lst,(0,self.segment_size-len(f0_lst)),'constant',0)
                ene = torch.nn.functional.pad(ene_lst,(0,self.segment_size-len(ene_lst)),'constant',0)
                mel = torch.nn.functional.pad(mel_lst,(0,0,0,self.segment_size-mel_lst.shape[0]),'constant',0)
                input_dim = torch.nn.functional.pad(input_dim_lst,(0,0,0,self.segment_size-input_dim_lst.shape[0]),'constant',0)

        else:
            input_dim = input_dim_lst
            f0 = f0_lst
            ene = ene_lst
            mel = mel_lst
        
        return basename,input_dim,f0,ene,mel
    
        

    def __len__(self,):
        return self.file_num
    
    def make_class_ary(self,):
        class_ary = []
        for ema_file,f0_file,ene_file,mel_file,ipa_file in self.files:
            basename = os.path.splitext(os.path.basename(ema_file))[0]

            class_ary.append((basename,EMA(ema_file),F0(f0_file),ENE(ene_file),MEL(mel_file),IPA(ipa_file,self.ema_hz)))
        return class_ary
    
    def make_dataset_ary(self,):
        dataset_ary = []
        for basename,ema_class,f0_class,ene_class,mel_class,ipa_class in self.class_ary:
            ema = ema_class.ema
            f0 = f0_class.f0
            ene = ene_class.ene
            mel = mel_class.mel
            ipa = ipa_class.ipa
            
        

            ###それぞれの長さを合わせる
            min_len = min([ema.shape[0],f0.shape[0],ene.shape[0],mel.shape[1],ipa.shape[0]])

            ema = ema[:min_len]
            f0 = f0[:min_len]
            ene = ene[:min_len]
            mel = mel[:,:min_len]
            ipa = ipa[:min_len,:]


            #torch型に変換
            ema = change_to_gpu(torch.Tensor(ema))
            f0 = change_to_gpu(torch.Tensor(f0))
            ene = change_to_gpu(torch.Tensor(ene))
            mel = change_to_gpu(torch.Tensor(mel)).transpose(1,0)
            ipa = change_to_gpu(torch.Tensor(ipa))
            

            #前処理それぞれを0から1に変換する
            ema = self.normalize(data=ema,
                                 max_value=self.max_ema_value,
                                 min_value=self.min_ema_value).view(ema.shape[0],6*2)
            f0 = self.normalize(data=f0,
                                max_value=self.max_f0_value,
                                min_value=self.min_f0_value,
                                threshold=self.max_f0_value)
            ene = self.normalize(data=ene,
                                max_value=self.max_ene_value,
                                min_value=self.min_ene_value)
            
            #入力
            ema_condition = ("EMA" in self.mode_lst)
            ipa_condition = ("IPA" in self.mode_lst)
            f0_condition = ("F0" in self.mode_lst)
            if ema_condition & ipa_condition & f0_condition:
                input_dim = torch.cat([ema,f0.clone().unsqueeze(1),ipa],1)
            elif ema_condition & ipa_condition & (not(f0_condition)):
                input_dim = torch.cat([ema,ipa],1)
            elif ema_condition & (not(ipa_condition)) &  (not(f0_condition)):
                input_dim = ema
            elif ema_condition & (not(ipa_condition)) & f0_condition:
                input_dim = torch.cat([ema,f0.clone().unsqueeze(1)],1)
            elif (not(ema_condition)) & ipa_condition & f0_condition:
                input_dim = torch.cat([f0.clone().unsqueeze(1),ipa],1)
            elif (not(ema_condition)) & (not(ipa_condition)) & f0_condition:
                input_dim = f0.clone().unsqueeze(1)
            elif (not(ema_condition)) & ipa_condition & (not(f0_condition)):
                input_dim = ipa
            else:
                raise ValueError("condition is False")
            
            

            

            dataset_ary.append((basename,input_dim,f0,ene,mel))
        return dataset_ary
            


            

    def normalize(self,data,max_value,min_value,threshold = None):
        #最大値を超える値を丸める
        if not(threshold is None):
            data = torch.where(data > threshold,threshold,data)
        return (data- min_value) / (max_value-min_value)






#EMAクラス
class EMA():
    def __init__(self,filename):
        #ema配列
        self.filename = filename
        #emaデータ
        self.ema = self.get_ema()


    def get_ema(self,):
        result = []
        with open(self.filename,"r") as f:
            for s in f.readlines():
                s = s.strip().split(" ")
                s = list(map(float,s))
                data = []

                for i in range(0,len(s),2):
                    data.append(s[i:i+2])
                result.append(data)
        return np.array(result)
    
class F0():
    def __init__(self,filename):
        self.filename = filename
        self.f0 = np.load(self.filename)

class MEL():
    def __init__(self,filename):
        self.filename = filename
        self.mel = np.load(self.filename)


class ENE():
    def __init__(self,filename):
        self.filename = filename
        self.ene = np.load(self.filename)

class IPA():
    def __init__(self,filename,ema_hz):
        self.ema_hz = ema_hz
        self.ipa = self.get_ipa(filename)

    def get_ipa(self,filename):
        ipa_pair = self.read_ipa(filename)
        ipas = self.ajust_ipa(ipa_pair)

        def ipa_to_onehot(ipa_str):
                #単位行列
                unit = np.eye(len(ipa_dict))

                def to_onehot(x):
                    return unit[ipa_dict[x]]
                
                return change_to_gpu(torch.Tensor(list(map(to_onehot,ipa_str))))
            
        #ipaをonehotに変換
        ipa_onehot = ipa_to_onehot(ipa_str=ipas)
        return ipa_onehot



    def ajust_ipa(self,ipa_pair):
        ajust_ipa_str = []
        
        mod_index = 0
        for ipa_text,ipa_len in ipa_pair:
            
            length_s = ipa_len

            length_index = (self.ema_hz*length_s)+mod_index

            ipas = [ipa_text] * int(length_index)

            mod_index = length_index - int(length_index)

            ajust_ipa_str.extend(ipas)
       

        ajust_ipa_str.extend([ipa_text]*(1 if mod_index>0 else 0))
        return ajust_ipa_str

    #ipaファイルの読み込み
    def read_ipa(self,filename):
        _min = []
        _max = []
        text =[]
        with open(filename,encoding="utf-16") as f:
            i = 0
   
            for s in f.readlines():
                s=s.strip()
                if s=="item [2]:":
                    i=1

                if i==1 and s.startswith("text = "):
                    text.append(s[s.index("\"") + 1: s.rindex("\"")])

                elif i == 1 and s.startswith("xmin = "):
                    _min.append(float(s[len("xmin = "):]))
                
                elif i == 1 and s.startswith("xmax = "):
                    _max.append(float(s[len("xmax = "):]))
            

        #先頭を削除 #無音区間を削除する
        _min = _min[2:-1]
        _max = _max[2:-1]
        text = text[1:-1]
        
    
        #音素の継続長
        ipa_len = [end-start for start,end in zip(_min,_max)]

        ipa_pair = [(te,le) for te,le in zip(text,ipa_len)]

        return ipa_pair



    


