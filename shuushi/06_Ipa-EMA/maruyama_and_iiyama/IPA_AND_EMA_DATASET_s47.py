import sys
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as data
import torch
import glob

import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from common.class_ema import EMA
from common.class_ipa import IPA
from data._dict_s47 import ipa_dict,position_dict,mean_param,std_param,normalize_param,max_param,min_param,ema_max_duration
import warnings
warnings.filterwarnings('ignore')



class IPA_EMA_DATASET(Dataset):

    def __init__(self,files,ema_hz=250,pre_process="min-max",pad=True):
        self.ema_hz = ema_hz
        #ファイルの配列
        #[(textgrid,ema),.....]
        self.files = files
        #ファイルの数
        self.file_num = len(files)
        #class_ary : [(IPA_class,EMA_class),....]
        self.class_ary = self.get_class_ary(self.files)

        #辞書をロード
        self.ipa_dict = ipa_dict
        self.position_dict = position_dict
        

        #無音区間の削除
        #voiced_ipa_str : (file_num,T_ipa)  #voiced_ipa_len: (file_num,T_ipa,2) #voiced_ema:(file_num,T_ema,6,2)
        self.voiced_ipa_str,self.voiced_ipa_len,self.voiced_ema=self.cut_silence(self.class_ary)

        


        
        
        #emaデータの継続長
        self.emaT_per_file = [temp_ema.shape[0] for temp_ema in self.voiced_ema]
       
        #パラメータ: self.get_params(self.voiced_ema)<-学習データのみから取得
        self.mean_param = mean_param
        self.std_param = std_param
        self.normalize_param = normalize_param
        self.max_param = max_param
        self.min_param = min_param
        
        self.ema_max_duration = ema_max_duration

        

        if pre_process=="min-max":
            self.nomalized_ema = self.nomalize_minmax(self.voiced_ema)
        elif pre_process == "stand":
            self.nomalized_ema = self.normalize_stand(self.voiced_ema)
        else:
            print("PreProcess_Error in IPA_AND_EMA_DATASET.py")
            sys.exit()

        self.nomalized_ema = self.restrict(self.nomalized_ema)


        #ipaの長さをema継続に合わせる ajust_ipa_str:(file_num,T_ema)
        self.ajust_ipa_str,self.estimate_ajust_ipa_len =self.ajust_ipa_str_size(self.voiced_ipa_str,self.voiced_ipa_len)

        #ipaをonehotに変換 ipa_onehot_ary: (file_num,T_ema,len(self.ipa_dict))
        self.ipa_onehot_list= self.ipa_onehot(self.ajust_ipa_str)

    
        #0埋めをする
        if pad==True:
            self.ipa_onehot_list = self.pad_0(self.ipa_onehot_list,type=None,value=0)
            self.raw_ema_list = self.pad_0(self.voiced_ema,type=float,value=0)
            self.nomalized_ema_list= self.pad_0(self.nomalized_ema,type=float,value=0)
            self.estimate_ipa_len_list = self.pad_0(self.estimate_ajust_ipa_len,type=None,value=-1)
        else:
            self.ipa_onehot_list = self.ipa_onehot_list
            self.raw_ema_list = self.voiced_ema
            self.nomalized_ema_list= self.nomalized_ema
            self.estimate_ipa_len_list = self.estimate_ajust_ipa_len
            

    def __len__(self,):
        return self.file_num
    
    def __getitem__(self, index):
        ema_basename = os.path.basename(self.files[index][1])
        ipa = torch.Tensor(self.ipa_onehot_list[index])
        raw_ = torch.Tensor(self.raw_ema_list[index])
        normalized_= torch.Tensor(self.nomalized_ema_list[index])
        estimate = torch.Tensor(self.estimate_ipa_len_list[index])

        return ipa,raw_,normalized_,self.emaT_per_file[index],estimate,ema_basename




    
    def pad_0(self,_in,type,value):
        result = []
        for i in _in:
            if type is not None:
                result.append(pad_sequences([i],maxlen=ema_max_duration,dtype=type,padding="post",truncating="post",value=value)[0])
            else:
                result.append(pad_sequences([i],maxlen=ema_max_duration,padding="post",truncating="post",value=value)[0])
        return result


    

    """
    ipaをonehotベクトルに変換
    """
    def ipa_onehot(self,all_ipa):
        #単位行列
        unit = np.eye(len(self.ipa_dict))
        result = []

        def to_onehot(x):
            return unit[self.ipa_dict[x]]

        for ipa_per_file in all_ipa:     
            result.append(torch.Tensor(list(map(to_onehot,ipa_per_file))))
            

        return result
    

    """
    ipaの長さをema継続に合わせる
    """
    def ajust_ipa_str_size(self,voiced_ipa_str,voiced_ipa_len):
        ajust_ipa_str = []
        ajust_ipa_len = []

        for ipa_str_per_file,ipa_len_per_file in zip(voiced_ipa_str,voiced_ipa_len):
            target_len = int(ipa_len_per_file[-1][1]*self.ema_hz) - int(ipa_len_per_file[0][0]*self.ema_hz)

            ajust_ipa_per_file = []
            ipa_len_per_file_ary = []
            
            mod_index = 0

            for index,(ipa_str,ipa_len) in enumerate(zip(ipa_str_per_file,ipa_len_per_file)):
                start_s,end_s = ipa_len
                length_s = -start_s+end_s 

                

                length_index = ((self.ema_hz*length_s)+mod_index)

                temp_ipa =[ipa_str for _ in range(int(length_index))] 

                ajust_ipa_per_file.extend(temp_ipa)

                mod_index = length_index - int(length_index)

                
                ipa_len_per_file_ary.append(int(length_index))
                
            
            
            if len(ajust_ipa_per_file) < target_len:
                diff = target_len - len(ajust_ipa_per_file)

                for _ in range(diff):
                    ajust_ipa_per_file.append(ajust_ipa_per_file[-1])
                    ipa_len_per_file_ary[-1] += 1

            elif len(ajust_ipa_per_file) > target_len:
                diff =  len(ajust_ipa_per_file) - target_len

                for _ in range(diff):
                    del ajust_ipa_per_file[-1]
                    ipa_len_per_file_ary[-1] -= 1

            
            ipa_len_per_file_ary = [sum(ipa_len_per_file_ary[0:i+1])for i in range(len(ipa_len_per_file_ary))]
            ajust_ipa_str.append(ajust_ipa_per_file)
            ajust_ipa_len.append(ipa_len_per_file_ary)

        return ajust_ipa_str,ajust_ipa_len

        


    """
    -1より下の値、1より上の値を-1もしくは1にする<=外れ値による影響軽減させるため
    """
    def restrict(self,normalized_ema):
        
        out_ema = []
        for ema_per_file in normalized_ema:
            ema_per_file = ema_per_file.double()
            temp_ema =  torch.where(ema_per_file>1.0,1.0,ema_per_file)
            temp_ema = torch.where(temp_ema< -1.0,-1.0,temp_ema)
            out_ema.append(temp_ema)
        return out_ema
    
    """
    標準化の逆変換
    """
    def inverse_stand(self,normalized_ema):
        if torch.cuda.is_available():
            return (normalized_ema*self.normalize_param.cuda()*self.std_param.cuda()) + self.mean_param.cuda()
        else:    
            return (normalized_ema*self.normalize_param*self.std_param) + self.mean_param

        
    """
    標準化(-1から1)
    """    
    def normalize_stand(self,raw_ema):
        normalized_ema = []
        for raw_ema_per_file in raw_ema:
            normalized_ema.append((raw_ema_per_file-self.mean_param) / (self.normalize_param*self.std_param))
        return normalized_ema
    
    """
    正規化の逆変換
    """
    def inverse_minmax(self,normalized_ema):
        if torch.cuda.is_available():
            return (normalized_ema+1)*((self.max_param.cuda()-self.min_param.cuda())/2) + self.min_param.cuda()
        else:
            return (normalized_ema+1)*((self.max_param-self.min_param)/2) + self.min_param
    """
    正規化(-1から1)
    """
    def nomalize_minmax(self,raw_ema):
        normalized_ema = []
        for raw_ema_per_file in raw_ema:
            normalized_ema.append(((raw_ema_per_file-self.min_param) / (self.max_param-self.min_param))*2-1)
        return normalized_ema

    """
    パラメータ探索(普通は使わない)
    """
    def get_params(self,ema_data):
        all_ema = []
        for ema in ema_data:
            all_ema.append(ema)
        all_ema = torch.cat(all_ema,dim=0)
        
        mean = torch.mean(all_ema,0)
        

        std = torch.std(all_ema,0)
        _max = torch.max(all_ema,0)[0]
        _min = torch.min(all_ema,0)[0]
        
        
        return mean,std,_max,_min

    """
    IPAとEMAデータの前後無音区間を削除する
    """
    def cut_silence(self,class_ary):
        ipa_text = []
        ipa_len = []
        ema = []

        for ipa_class,ema_class in class_ary:
            voiced_start = ipa_class.ipa_pair[0][1][1]
            voiced_end = ipa_class.ipa_pair[-1][1][0]
            
            
            #無音記号0(最初と最後のみ)を削除する
            voiced_ipa_string = [ipa_str for (ipa_str,(_,_) ) in ipa_class.ipa_pair][1:-1]
            voiced_ipa_len = [ duration for (_,duration ) in ipa_class.ipa_pair][1:-1]

            #emaデータの無音区間を除去
            voiced_ema = ema_class.ema_data[int(voiced_start*self.ema_hz):int(voiced_end*self.ema_hz)]

            ipa_text.append(voiced_ipa_string)
            ipa_len.append(voiced_ipa_len)
            ema.append(voiced_ema)
        
        return ipa_text,ipa_len,ema
            


    """
    クラスの配列
    """
    def get_class_ary(self,files):
        class_ary = []
        #class_ary : [(IPA_class,EMA_class),....]
        for (ipa_file,ema_file) in files:
            class_ary.append([IPA(ipa_file),EMA(ema_file)])

        return class_ary
            


def read_txtfile(filename):
    basename_ary = []
    with open(filename,mode="r") as f:
        for s in f.readlines():
            s = s.strip()
            basename_ary.append(s)
    return basename_ary


if __name__ == "__main__":
    import os
    ipa_path = "/Users/maruyamakoki/Desktop/研究/修士研究/Dataset/text"
    ema_path =  "/Users/maruyamakoki/Desktop/研究/修士研究/Dataset/ema"



    train_file = "/Users/maruyamakoki/Desktop/研究/修士研究/Dataset/set_textgrid/test.txt"
    basenames = read_txtfile(train_file)
    

    ipa_files = [os.path.join(ipa_path,b) for b in basenames]
    ema_files = [os.path.join(ema_path,b.replace("AD20180911","201809110").replace("_0_labeled.TextGrid",".ema")) for b in basenames]
    

    files = [(i,j) for i,j in zip(ipa_files,ema_files)]
    """

    files = [("/Users/maruyamakoki/Desktop/研究/修士研究/Dataset/text/AD201809110001_0_labeled.TextGrid","/Users/maruyamakoki/Desktop/研究/修士研究/Dataset/ema/2018091100001.ema")
             ,("/Users/maruyamakoki/Desktop/研究/修士研究/Dataset/text/AD201809110002_0_labeled.TextGrid","/Users/maruyamakoki/Desktop/研究/修士研究/Dataset/ema/2018091100002.ema")
             ,("/Users/maruyamakoki/Desktop/研究/修士研究/Dataset/text/AD201809110003_0_labeled.TextGrid","/Users/maruyamakoki/Desktop/研究/修士研究/Dataset/ema/2018091100003.ema")
             ,("/Users/maruyamakoki/Desktop/研究/修士研究/Dataset/text/AD201809110503_0_labeled.TextGrid","/Users/maruyamakoki/Desktop/研究/修士研究/Dataset/ema/2018091100503.ema")]
    """
    dataset = IPA_EMA_DATASET(files[:5],pre_process="stand",pad=True)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False)

    """
    ipa: [batch, 2420, len(IPA_DICT)]
    raw_ema : [batch,2440,6,2]
    normalized_ema : [batch,2420,6,2]
    ema_T : [batch] 
    estimate_ipa_len : [batch,107]
    """
    for index,(ipa,raw_ema,normalized_ema,ema_T,estimate_ipa_len,filename) in enumerate(loader):
        print("-----")
        print(ipa.shape)
        print(raw_ema.shape)
        print(normalized_ema.shape)
        print(ema_T)
        print(estimate_ipa_len.shape)
        print(filename)
        
        
    

