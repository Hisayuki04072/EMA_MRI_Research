import torch
import torch.utils.data as data
import os
import torch.optim as optim
import torch.nn as nn
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import time
import sys
import random
import glob
import datetime
import yaml
import argparse 
import copy
import csv
import librosa.display
import io
import cv2
import matplotlib.colors as mcolors
from _dict import ipa_dict


import shutil
from utils import save_checkpoint,load_checkpoint
import json
from torch.utils.tensorboard import SummaryWriter

from intergration_block_nn import integration
from mel_gen import Conformer
from predictor_nn import predictor
from style_encoder_nn import style_encoder
from F0EMAIPA_to_MEL_Dataset_s47 import F0EMAIPA2Mel_Dataset


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# numpy の seed を固定
np.random.seed(0)


MU_F0 = 0.1
MU_ENE = 0.1
MU_MEL = 0.8



"""
GPUに乗せる
"""
def change_to_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e

def read_yamlfile(filename):
    with open(filename,mode="r") as f:
        config = yaml.safe_load(f)
    return config


def read_txtfile(filename):
    basename_ary = []
    with open(filename,mode="r") as f:
        for s in f.readlines():
            s = s.strip().replace(".TextGrid","")
            basename_ary.append(s)
    return basename_ary

def concat_files(ema_path,f0_path,ene_path,mel_path,ipa_path,target_basenames):
    filenames_ary = []
    for basename in target_basenames:
        ema_filename = os.path.join(ema_path,basename+".ema")
        f0_filename = os.path.join(f0_path,basename+".npy")
        ene_filename = os.path.join(ene_path,basename+".npy")
        mel_filename = os.path.join(mel_path,basename+".npy")
        ipa_filename = os.path.join(ipa_path,basename+".TextGrid")

        condition = (os.path.exists(ema_filename)) and (os.path.exists(f0_filename)) and (os.path.exists(ene_filename)) and (os.path.exists(mel_filename)) and os.path.exists(ipa_filename)
 
        if condition:
            filenames_ary.append((ema_filename,f0_filename,ene_filename,mel_filename,ipa_filename))
       
    return filenames_ary

def make_dataset(ema_path,f0_path,ene_path,mel_path,ipa_path,target_basenames,batch_size,model_lst):

    files = concat_files(ema_path=ema_path,
                         f0_path=f0_path,
                         ene_path=ene_path,
                         mel_path=mel_path,
                         ipa_path = ipa_path,
                         target_basenames=target_basenames)
    
    dataset =F0EMAIPA2Mel_Dataset(files=files,
                                batch_size=batch_size,
                                mode_lst=model_lst)
    
    
    loader = data.DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True)
    return loader



def read_atr503_txt(_file):
    atr_503_dict = {}
    with open(_file,mode="r") as f:
        for s in f.readlines():
            s = s.strip().split(":")
            atr_503_dict[s[1]] = s[0]
    return atr_503_dict 


def get_train_valid(valid_name,atr_503_dict,test_basenams):
    valid_keys = set([key for key in list(atr_503_dict.keys()) if valid_name in key])
    train_keys = set(list(atr_503_dict.keys())) - valid_keys

    train_basenames = list(set([atr_503_dict[train_key] for train_key in train_keys]) - set(test_basenams))
    valid_basenames = [atr_503_dict[valid_key] for valid_key in valid_keys]

    return train_basenames,valid_basenames


def make_cross_valid_dataset(atr_503_dict,valid_names,e_train_txt,e_valid_txt,test_txt):
    basenames_dict = {}
    test_basenames = read_txtfile(test_txt)
    for valid_name in valid_names:
        tmp_dict = {}
        tmp_dict["test"] = test_basenames

        if valid_name=="j":
            tmp_dict["train"] = read_txtfile(e_train_txt)
            tmp_dict["valid"] = read_txtfile(e_valid_txt)
        else:
            tmp_dict["train"],tmp_dict["valid"]= get_train_valid(valid_name=valid_name,atr_503_dict=atr_503_dict,test_basenams=test_basenames)
        
        basenames_dict[valid_name] = tmp_dict
    return basenames_dict
            
            



            


def train(yaml_file):
    yaml_dict = read_yamlfile(yaml_file)

    ema_path = yaml_dict["dataset"]["ema_dataset"]
    f0_path = yaml_dict["dataset"]["f0_dataset"]
    mel_path = yaml_dict["dataset"]["mel_dataset"]
    ene_path = yaml_dict["dataset"]["energy_dataset"]
    ipa_path = yaml_dict["dataset"]["ipa_dataset"]

    train_txt = yaml_dict["dataset"]["train_txt"]
    valid_txt = yaml_dict["dataset"]["valid_txt"]
    test_txt = yaml_dict["dataset"]["test_txt"]
    atr_503_txt = yaml_dict["dataset"]["atr_503_txt"]

    valid_interval  = yaml_dict["validation_interval"]
    checkpoint_interval = yaml_dict["checkpoint_interval"]
    epoch_num = yaml_dict["epoch_num"]
    lr = yaml_dict["lr"]
    lr_decay = yaml_dict["lr_decay"]
    num_mels = yaml_dict["num_mel"]
    valid_names = yaml_dict["valid_name"]
    batch_size = yaml_dict["batch_size"]

    atr_503_dict = read_atr503_txt(atr_503_txt)

    basename_dict = make_cross_valid_dataset(atr_503_dict=atr_503_dict,
                                            valid_names=valid_names,
                                            e_train_txt=train_txt,
                                            e_valid_txt=valid_txt,
                                            test_txt=test_txt)
    mode_lst = yaml_dict["mode"]

    ori_out_dir = yaml_dict["out_dir"] + "_input_"+ "_".join(mode_lst)
   

    
    for valid_name in basename_dict:
        train_basenames = basename_dict[valid_name]["train"]
        valid_basenames = basename_dict[valid_name]["valid"]
        test_basenames = basename_dict[valid_name]["test"]

        "出力について"
        out_dir = os.path.join(ori_out_dir,"valid_"+valid_name)
        os.makedirs(out_dir,exist_ok=True)
        state_txt = open(os.path.join(out_dir,"state.txt"),mode="a")

        """set.yamlについて"""
        if not(os.path.exists(os.path.join(out_dir,os.path.basename(yaml_file)))):
            os.makedirs(out_dir,exist_ok=True)
            shutil.copy(yaml_file,os.path.join(out_dir,os.path.basename(yaml_file)))

        """モデルについて"""
        middle_size = 256
        in_size = 0
        if "F0" in mode_lst:
            in_size+=1
        if "IPA" in mode_lst:
            in_size+=len(ipa_dict)
        if "EMA" in mode_lst:
            in_size+=12
        integration_model = change_to_gpu(integration(in_size=in_size,out_size=middle_size))
        style_encoder_model = change_to_gpu(style_encoder(in_size=middle_size))
        f0_predictor_model = change_to_gpu(predictor(in_size=middle_size))
        energy_predictor_model = change_to_gpu(predictor(in_size=middle_size))
        mel_gen_model = change_to_gpu(Conformer(input_dim=middle_size*2,num_heads=4,num_layers=8,ffn_dim=middle_size*4,output_dim=num_mels,depthwise_conv_kernel_size=31))

        integration_optim = optim.AdamW(integration_model.parameters(), lr, betas=[0.8, 0.99])
        style_encoder_optim = optim.AdamW(style_encoder_model.parameters(),lr, betas=[0.8, 0.99])
        f0_predictor_optim =  optim.AdamW(f0_predictor_model.parameters(), lr, betas=[0.8, 0.99])
        energy_predictor_optim = optim.AdamW(energy_predictor_model.parameters(), lr, betas=[0.8, 0.99])
        mel_gen_optim = optim.AdamW(mel_gen_model.parameters(), lr, betas=[0.8, 0.99])

        loss_function = change_to_gpu(nn.L1Loss())

        last_epoch = -1

        if yaml_dict["model_read"]:
            model_path = os.path.join(out_dir,"model",yaml_dict["model_basename"])
            if not os.path.exists(model_path):
                raise ValueError("Model_path_mistake")
            
            state_dict = load_checkpoint(model_path,torch.device('cpu'))

            integration_model.load_state_dict(state_dict["integration_model"])
            style_encoder_model.load_state_dict(state_dict["style_encoder_model"])
            f0_predictor_model.load_state_dict(state_dict["f0_predictor_model"])
            energy_predictor_model.load_state_dict(state_dict["energy_predictor_model"])
            mel_gen_model.load_state_dict(state_dict["mel_gen_model"])

            integration_optim.load_state_dict(state_dict["integration_optim"])
            style_encoder_optim.load_state_dict(state_dict["style_encoder_optim"])
            f0_predictor_optim.load_state_dict(state_dict["f0_predictor_optim"])
            energy_predictor_optim.load_state_dict(state_dict["energy_predictor_optim"])
            mel_gen_optim.load_state_dict(state_dict["mel_gen_optim"])
            last_epoch = state_dict["epoch"]

        if (yaml_dict["model_read"]) and (yaml_dict["test_only"]):
            state_txt.write("Test NOW EPOCH "+str(last_epoch)+"\n")
            model_ary = [integration_model,style_encoder_model,f0_predictor_model,energy_predictor_model,mel_gen_model]

            test_loader = make_dataset(ema_path=ema_path,
                            f0_path=f0_path,
                            mel_path=mel_path,
                            ene_path=ene_path,
                            ipa_path=ipa_path,
                            target_basenames=test_basenames,
                            batch_size=1,
                            model_lst=mode_lst)
            valid_loader = make_dataset(ema_path=ema_path,
                            f0_path=f0_path,
                            mel_path=mel_path,
                            ene_path=ene_path,
                            ipa_path=ipa_path,
                           target_basenames=valid_basenames,
                            batch_size=1,
                            model_lst=mode_lst)
            
            train_loader = make_dataset(ema_path=ema_path,
                            f0_path=f0_path,
                            mel_path=mel_path,
                            ene_path=ene_path,
                            ipa_path=ipa_path,
                           target_basenames=train_basenames,
                            batch_size=1,
                            model_lst=mode_lst)
            
            test(loader=test_loader,
                model_ary=model_ary,
                loss_func=loss_function,
                out_dir=os.path.join(out_dir,"result","test"),
                value_dir=os.path.join(out_dir,"result","mel_output"))
            
            
            
            test(loader=valid_loader,
                model_ary=model_ary,
                loss_func=loss_function,
                out_dir=os.path.join(out_dir,"result","valid"),
                value_dir=os.path.join(out_dir,"result","mel_output"))
            
            test(loader=train_loader,
                model_ary=model_ary,
                loss_func=loss_function,
                out_dir=os.path.join(out_dir,"result","train"),
                value_dir=os.path.join(out_dir,"result","mel_output"))
            
            
            continue
            #test

        
        state_txt.write("TRAIN from "+str(last_epoch)+" EPOCH to "+str(epoch_num)+" EPOCH \n")

        
        valid_loader = make_dataset(ema_path=ema_path,
                            f0_path=f0_path,
                            mel_path=mel_path,
                            ene_path=ene_path,
                            ipa_path=ipa_path,
                           target_basenames=valid_basenames,
                            batch_size=1,
                            model_lst=mode_lst)
        train_loader = make_dataset(ema_path=ema_path,
                        f0_path=f0_path,
                        mel_path=mel_path,
                        ene_path=ene_path,
                        ipa_path=ipa_path,
                        target_basenames=train_basenames,
                        batch_size=batch_size,
                        model_lst=mode_lst)
        
        """出力について"""
        out_model_dir = os.path.join(out_dir,"model")
        out_log_dir = os.path.join(out_dir,"logs")
        os.makedirs(out_model_dir,exist_ok=True)
        os.makedirs(out_log_dir,exist_ok=True)

        with open(os.path.join(out_model_dir,"model.txt"),mode="w") as f:
            f.write("integration_model:"+str(integration_model)+"\n\n")
            f.write("style_encoder_model:"+str(style_encoder_model)+"\n\n")
            f.write("f0_predictor_model:"+str(f0_predictor_model)+"\n\n")
            f.write("energy_predictor_model:"+str(energy_predictor_model)+"\n\n")
            f.write("mel_gen_model:"+str(mel_gen_model)+"\n\n")

        if torch.cuda.is_available():
            print("use GPU in train")
            
        else:
            print("use CPU in train")

        sw = SummaryWriter(out_log_dir)
        with open(os.path.join(out_dir,"web_log.txt"),mode="w") as f:
            f.write("tensorboard --logdir="+str(out_log_dir))
        model_ary = [integration_model,style_encoder_model,f0_predictor_model,energy_predictor_model,mel_gen_model]
        optim_ary=[integration_optim,style_encoder_optim,f0_predictor_optim,energy_predictor_optim,mel_gen_optim]

        for epoch in range(max(0,last_epoch+1),epoch_num):
            current_epoch = str(epoch)+"/"+str(epoch_num)

            """train用"""
            model_ary,optim_ary,train_loss_dict,train_time = learn(loader=train_loader,
                                                                model_ary=model_ary,
                                                                optim_ary=optim_ary,
                                                                loss_func=loss_function,
                                                                current_epoch=current_epoch)
            """出力"""
            print("VALID NAME : "+valid_name +" TRAIN EPOCH:{} TIME:{}".format(epoch,round(train_time,5)))
            print("         LOSS ALL:{} MEL:{} F0:{} ENE:{}".format(round(train_loss_dict["ALL"],5),round(train_loss_dict["MEL"],5),round(train_loss_dict["F0"],5),round(train_loss_dict["ENE"],5)))

            sw.add_scalar("train_LOSS/ALL_LOSS",train_loss_dict["ALL"],epoch)
            sw.add_scalar("train_LOSS/ALL_LOSS",train_loss_dict["ALL"],epoch)
            sw.add_scalar("train_LOSS/MEL_LOSS",train_loss_dict["MEL"],epoch)
            sw.add_scalar("train_LOSS/F0_LOSS",train_loss_dict["F0"],epoch)
            sw.add_scalar("train_LOSS/ENE_LOSS",train_loss_dict["ENE"],epoch)

            if epoch % valid_interval == 0 and epoch!=0:
                valid_loss_dict,valid_time = valid(loader=valid_loader,
                                                model_ary=model_ary,
                                                loss_func=loss_function,
                                                current_epoch=current_epoch,
                                                epoch = epoch,
                                                sw=sw)
                """出力"""
                print("VALID EPOCH:{} TIME:{}".format(epoch,round(valid_time,5)))
                print("         LOSS ALL:{} MEL:{} F0:{} ENE:{}".format(round(valid_loss_dict["ALL"],5),round(valid_loss_dict["MEL"],5),round(valid_loss_dict["F0"],5),round(valid_loss_dict["ENE"],5)))

                sw.add_scalar("valid_LOSS/ALL_LOSS",valid_loss_dict["ALL"],epoch)
                sw.add_scalar("valid_LOSS/MEL_LOSS",valid_loss_dict["MEL"],epoch)
                sw.add_scalar("valid_LOSS/F0_LOSS",valid_loss_dict["F0"],epoch)
                sw.add_scalar("valid_LOSS/ENE_LOSS",valid_loss_dict["ENE"],epoch)
               

            """モデルの保存"""
            if epoch % checkpoint_interval == 0 and epoch!=0:
                save_model(model_ary=model_ary,
                        optim_ary=optim_ary,
                        out_model_dir=out_model_dir,
                        epoch=epoch
                        )



        
        


    

    

    



def save_model(model_ary,optim_ary,out_model_dir,epoch):
    integration_model,style_encoder_model,f0_predictor_model,energy_predictor_model,mel_gen_model = model_ary

    integration_optim,style_encoder_optim,f0_predictor_optim,energy_predictor_optim,mel_gen_optim = optim_ary

    model_path = os.path.join(out_model_dir,"model_ep"+str(epoch)+".pth")

    save_checkpoint(model_path,
                            {"integration_model":integration_model.state_dict(),
                             "style_encoder_model":style_encoder_model.state_dict(),
                             "f0_predictor_model":f0_predictor_model.state_dict(),
                             "energy_predictor_model":energy_predictor_model.state_dict(),
                             "mel_gen_model":mel_gen_model.state_dict(),
                             "integration_optim":integration_optim.state_dict(),
                             "style_encoder_optim":style_encoder_optim.state_dict(),
                             "f0_predictor_optim":f0_predictor_optim.state_dict(),
                             "energy_predictor_optim":energy_predictor_optim.state_dict(),
                             "mel_gen_optim":mel_gen_optim.state_dict(),
                             "epoch":epoch
                             })





"""テスト用"""
def test(loader,model_ary,loss_func,out_dir,value_dir):
    out_dir01 = os.path.join(out_dir,"mel_image")
    out_dir02 = value_dir
    os.makedirs(out_dir01,exist_ok=True)
    os.makedirs(out_dir02,exist_ok=True)
    out_filename_all = os.path.join(out_dir,"loss_all.txt")
    out_filename_mel = os.path.join(out_dir,"loss_mel.txt")

    integration_model,style_encoder_model,f0_predictor_model,energy_predictor_model,mel_gen_model = model_ary


    bar = tqdm(total= len(loader),leave=False)
    bar.set_description("TEST")

    mel_loss_dict = {}
    all_loss_dict = {}

    with torch.no_grad():
        integration_model.eval()
        style_encoder_model.eval()
        mel_gen_model.eval()
        f0_predictor_model.eval()
        energy_predictor_model.eval()

        for basename,input_dim,f0,ene,mel in loader:
            # integration_embedding : (batch,T,256)
            integration_embedding_01 = integration_model(input_dim)
            integration_embedding_02 = integration_embedding_01.clone()

            # style_embedding : (batch,T,256)
            style_embedding_01 = style_encoder_model(integration_embedding_02)
            style_embedding_02 = style_embedding_01.clone()
            style_embedding_03 = style_embedding_01.clone()

            # f0_predict : (batch,T,1)
            f0_predict = f0_predictor_model(style_embedding_02)

            #energy_predict : (batch,T,1)
            energy_predict = energy_predictor_model(style_embedding_03)

            mel_input = torch.cat([integration_embedding_01,style_embedding_01],2)
            length = change_to_gpu(torch.Tensor([mel_input.shape[1]]*mel_input.shape[0]))

            # mel_predict : (batch,T,60)
            mel_predict,_ = mel_gen_model(mel_input,length)

            #loss
            l_f0 = loss_func(f0,f0_predict.squeeze(-1))
            l_ene = loss_func(ene,energy_predict.squeeze(-1))
            l_mel = loss_func(mel,mel_predict)
            l_all = MU_MEL*l_mel + MU_F0*l_f0 + MU_ENE*l_ene


            basename = basename[0]


            mel_loss_dict[basename] = l_mel.item()
            all_loss_dict[basename] = l_all.item()

            """出力"""
            save_mel_number(in_mel=mel_predict.squeeze(0).transpose(1,0),
                            out_filename=os.path.join(out_dir02,basename+".npy"))
            #メルスペクトログラムを保存
            
            save_mel_img(target_mel=mel.squeeze(0),
                         predict_mel=mel_predict.squeeze(0),
                         out_filename=os.path.join(out_dir01,basename+".png"))
            
            bar.update(1)
        bar.close()

    mel_loss_dict = get_avg(sort_dict(mel_loss_dict))
    all_loss_dict = get_avg(sort_dict(all_loss_dict))
    write_dict(out_filename_mel,mel_loss_dict)
    write_dict(out_filename_all,all_loss_dict)


def write_dict(output_filename,_dict):
    with open(output_filename,mode="w") as f:
        for key in _dict:
            f.write(key+" : "+str(_dict[key])+"\n")


def sort_dict(_dict):
    return dict(sorted(_dict.items(), key = lambda __dict : __dict[1]))

def get_avg(_dict):
    _dict["AVG"] = np.mean(np.array([_dict[key] for key in _dict]))
    return _dict



def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def save_mel_img(target_mel,predict_mel,out_filename):
    target_mel_np = target_mel.transpose(1,0).to('cpu').detach().numpy().copy()
    predict_mel_np = predict_mel.transpose(1,0).to('cpu').detach().numpy().copy()

    
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    img1 = librosa.display.specshow(predict_mel_np, y_axis='mel', x_axis='time',sr=16000,n_fft=512,hop_length=64,ax=ax[0],cmap="viridis")
    
    ax[0].set(title='predict')
    ax[0].label_outer()

    img2 = librosa.display.specshow(target_mel_np, y_axis='mel', x_axis='time',
                                sr=16000,n_fft=512,hop_length=64,ax=ax[1],cmap="viridis")
    img1.set_clim(vmin=-12, vmax=0)
    img2.set_clim(vmin=-12, vmax=0)
    ax[1].set(title='target')
    ax[1].label_outer()
    fig.colorbar(img1, ax=ax, format="%+2.f dB")
    if out_filename is None:
        fig.canvas.draw()
        plt.close()
        return fig
    else:
        plt.savefig(out_filename)
    plt.close()
    


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def save_mel_number(in_mel,out_filename):
    in_mel_np = in_mel.to('cpu').detach().numpy().copy()
    np.save(out_filename,in_mel_np)




"""評価用"""
def valid(loader,model_ary,loss_func,current_epoch,epoch,sw):
    integration_model,style_encoder_model,f0_predictor_model,energy_predictor_model,mel_gen_model = model_ary

    bar = tqdm(total= len(loader),leave=False)
    bar.set_description("VALID_"+current_epoch)

    loss_dict = {"ALL":[],"MEL":[],"F0":[],"ENE":[]}

    start = time.time()

    with torch.no_grad():
        integration_model.eval()
        style_encoder_model.eval()
        mel_gen_model.eval()
        f0_predictor_model.eval()
        energy_predictor_model.eval()

        for basename,input_dim,f0,ene,mel in loader:
            # integration_embedding : (batch,T,256)
            integration_embedding_01 = integration_model(input_dim)
            integration_embedding_02 = integration_embedding_01.clone()

            # style_embedding : (batch,T,256)
            style_embedding_01 = style_encoder_model(integration_embedding_02)
            style_embedding_02 = style_embedding_01.clone()
            style_embedding_03 = style_embedding_01.clone()

            # f0_predict : (batch,T,1)
            f0_predict = f0_predictor_model(style_embedding_02)

            #energy_predict : (batch,T,1)
            energy_predict = energy_predictor_model(style_embedding_03)

            mel_input = torch.cat([integration_embedding_01,style_embedding_01],2)
            length = change_to_gpu(torch.Tensor([mel_input.shape[1]]*mel_input.shape[0]))

            # mel_predict : (batch,T,60)
            mel_predict,_ = mel_gen_model(mel_input,length)

            #loss
            l_f0 = loss_func(f0,f0_predict.squeeze(-1))
            l_ene = loss_func(ene,energy_predict.squeeze(-1))
            l_mel = loss_func(mel,mel_predict)
            l_all = MU_MEL*l_mel + MU_F0*l_f0 + MU_ENE*l_ene

            loss_dict["ALL"].append(l_all.item())
            loss_dict["MEL"].append(l_mel.item())
            loss_dict["F0"].append(l_f0.item())
            loss_dict["ENE"].append(l_ene.item())
            
            basename = basename[0]
           
            """
            sw.add_figure("MEL/{}".format(basename),
                          save_mel_img(target_mel=mel.squeeze(0),
                         predict_mel=mel_predict.squeeze(0),
                         out_filename=None),
                         epoch
                         )
            """

            bar.update(1)
        bar.close()
    
    avg_loss_dict = {key : sum(loss_dict[key])/len(loss_dict[key]) for key in loss_dict}
    
    return avg_loss_dict,time.time()-start


"""学習用データ"""
def learn(loader,model_ary,optim_ary,loss_func,current_epoch):
    integration_model,style_encoder_model,f0_predictor_model,energy_predictor_model,mel_gen_model = model_ary

    integration_optim,style_encoder_optim,f0_predictor_optim,energy_predictor_optim,mel_gen_optim = optim_ary

    bar = tqdm(total= len(loader),leave=False)
    bar.set_description("TRAIN_"+current_epoch)

    loss_dict = {"ALL":[],"MEL":[],"F0":[],"ENE":[]}

    start = time.time()

    integration_model.train()
    style_encoder_model.train()
    f0_predictor_model.train()
    energy_predictor_model.train()
    mel_gen_model.train()

    for _,input_dim,f0,ene,mel in loader:
        integration_optim.zero_grad()
        style_encoder_optim.zero_grad()
        f0_predictor_optim.zero_grad()
        energy_predictor_optim.zero_grad()
        mel_gen_optim.zero_grad()


        #input_dim : (batch,T,in_channel)
        #f0 : (batch,T)
        #ene : (batch,T)
        #mel : (batch,T,num_mels)

        # integration_embedding : (batch,T,256)
        integration_embedding_01 = integration_model(input_dim)
        integration_embedding_02 = integration_embedding_01.clone()

        # style_embedding : (batch,T,256)
        style_embedding_01 = style_encoder_model(integration_embedding_02)
        style_embedding_02 = style_embedding_01.clone()
        style_embedding_03 = style_embedding_01.clone()

        # f0_predict : (batch,T,1)
        f0_predict = f0_predictor_model(style_embedding_02)

        #energy_predict : (batch,T,1)
        energy_predict = energy_predictor_model(style_embedding_03)

        mel_input = torch.cat([integration_embedding_01,style_embedding_01],2)
        length = change_to_gpu(torch.Tensor([mel_input.shape[1]]*mel_input.shape[0]))

        # mel_predict : (batch,T,60)
        mel_predict,_ = mel_gen_model(mel_input,length)

        #loss
        l_f0 = loss_func(f0,f0_predict.squeeze(-1))
        l_ene = loss_func(ene,energy_predict.squeeze(-1))
        l_mel = loss_func(mel,mel_predict)
        l_all = MU_MEL*l_mel + MU_F0*l_f0 + MU_ENE*l_ene

        loss_dict["ALL"].append(l_all.item())
        loss_dict["MEL"].append(l_mel.item())
        loss_dict["F0"].append(l_f0.item())
        loss_dict["ENE"].append(l_ene.item())

        l_all.backward()
        integration_optim.step()
        style_encoder_optim.step()
        f0_predictor_optim.step()
        energy_predictor_optim.step()
        mel_gen_optim.step()

        bar.update(1)
    bar.close()

    model_ary = [integration_model,style_encoder_model,f0_predictor_model,energy_predictor_model,mel_gen_model]
    optim_ary=[integration_optim,style_encoder_optim,f0_predictor_optim,energy_predictor_optim,mel_gen_optim]

    avg_loss_dict = {key : sum(loss_dict[key])/len(loss_dict[key]) for key in loss_dict}

    return model_ary,optim_ary,avg_loss_dict,time.time()-start





















    

    
    
    





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-yaml', '--inyaml',default="set.yaml")
    args = parser.parse_args()

    yaml_file = args.inyaml
    train(yaml_file)