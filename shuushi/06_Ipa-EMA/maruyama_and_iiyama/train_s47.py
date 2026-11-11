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

from IPA_AND_EMA_DATASET_s47 import IPA_EMA_DATASET
from nn import SEnet_1D,BLSTM_AND_LSTM
from data._dict_s47 import position_dict,ipa_dict


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# numpy の seed を固定
np.random.seed(0)


"""
GPUに乗せる
"""
def change_to_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e
"""
yamlファイルを読み込む
"""
def read_yamlfile(filename):
    with open(filename,mode="r") as f:
        config = yaml.safe_load(f)
    return config
"""
txtファイルを読み込む
"""
def read_txtfile(filename):
    basename_ary = []
    with open(filename,mode="r") as f:
        for s in f.readlines():
            s = s.strip()
            basename_ary.append(s)
    return basename_ary
"""
パスの結合
"""
def make_filepath(_folder,basenames):
    return [os.path.join(_folder,basename) for basename in basenames]

"""
IPAファイルとEMAファイルを結合
"""
def concat_file(ipa_files,ema_files):
    return [(a,b) for a,b in zip(ipa_files,ema_files)]


def train(dataset_ary,yaml_dict):
    batch_size = yaml_dict["batch_size"]
    pad = yaml_dict["pad"]
    resnet = yaml_dict["ResNet"]
    normalize = yaml_dict["normalize"]
    epoch_num = yaml_dict["epoch_num"]
    if batch_size!=1 and pad==False:
        print("YAML_ERROR")
        sys.exit()
    if resnet:
        out_dir = yaml_dict["out_dir"] +"_SENET"
    else:
        out_dir = yaml_dict["out_dir"] +"_BLSTM"
    out_dir+= "_batch"+str(batch_size)+"_"+normalize


    train_files,valid_files,test_files = dataset_ary
    
    #学習データ
    train_dataset = IPA_EMA_DATASET(train_files,pre_process=normalize,pad=pad)
    train_loader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

    #評価用データ
    valid_dataset = IPA_EMA_DATASET(valid_files,pre_process=normalize,pad=pad)
    valid_loader = data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=False)

    #テスト用データ
    test_dataset = IPA_EMA_DATASET(test_files,pre_process=normalize,pad=pad)
    test_loader = data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    if resnet:
        model = SEnet_1D(len(ipa_dict),num_block=4,kernel_size=39)
    else:
        model = BLSTM_AND_LSTM(len(ipa_dict),64,3)


    model = change_to_gpu(model)
    model_optimizer =  optim.SGD(model.parameters(),lr=0.05,momentum=0.9,weight_decay=1e-4) #変更前はoptim.SGD(model.parameters(),lr=0.5,momentum=0.9)

    loss_function = nn.MSELoss()

    #出力
    out_model_dir = os.path.join(out_dir,"model")
    out_one_dir = os.path.join(out_model_dir,"one")
    out_sepa_dir = os.path.join(out_model_dir,"sepa")
    out_else_dir = os.path.join(out_dir,"else")
    out_else_png_dir = os.path.join(out_else_dir,"png")

    out_result_dir = os.path.join(out_dir,"result")
    out_result_one_dir = os.path.join(out_result_dir,"one")
    out_result_sepa_dir = os.path.join(out_result_dir,"sepa")
    out_result_train_dir = os.path.join(out_result_sepa_dir,"train")
    out_result_valid_dir = os.path.join(out_result_sepa_dir,"valid")
    out_result_test_dir = os.path.join(out_result_sepa_dir,"test")


    os.makedirs(out_one_dir,exist_ok=True)
    os.makedirs(out_sepa_dir,exist_ok=True)
    os.makedirs(out_else_dir,exist_ok=True)
    os.makedirs(out_result_one_dir,exist_ok=True)
    os.makedirs(out_result_train_dir,exist_ok=True)
    os.makedirs(out_result_valid_dir,exist_ok=True)
    os.makedirs(out_result_test_dir,exist_ok=True)
    os.makedirs(out_else_png_dir,exist_ok=True)

    with open(os.path.join(out_model_dir,"model.txt"),mode="w") as f:
        f.write("model:"+str(model)+"\n")
        f.write("optimizer:"+str(model_optimizer)+"\n")

    
    if torch.cuda.is_available():
        print("use GPU in train")
        
    else:
        print("use CPU in train")

    #エポックごとの損失
    train_loss_ary,valid_loss_ary,test_loss_ary = [],[],[]

    train_rmse_ary,valid_rmse_ary,test_rmse_ary = [],[],[]
    train_part_rmse_ary,valid_part_rmse_ary,test_part_rmse_ary = [],[],[]

    model_ary = []

    for epoch in range(epoch_num):
        #エポックごとの開始時間
        current_epoch = str(epoch)+"/"+str(epoch_num)
        
        model,model_optimizer,train_loss,train_all_rmse,train_part_rmse,train_time = learn(train_loader,model,model_optimizer,loss_function,mode="train",normalize=normalize,epoch=current_epoch)

        print("Epoch:{}  Loss: {} Time: {} s".format(epoch,train_loss,train_time))
        #モデルの保存
        model_ary.append(copy.deepcopy(model))

        _,_,valid_loss,valid_all_rmse,valid_part_rmse,_ = learn(valid_loader,model,model_optimizer,loss_function,mode="valid",normalize=normalize,epoch=current_epoch)

        _,_,test_loss,test_all_rmse,test_part_rmse,_ = learn(test_loader,model,model_optimizer,loss_function,mode="test",normalize=normalize,epoch=current_epoch)

        """
        パラメータの記入
        """
        write_param_to_txt(train_time,os.path.join(out_else_dir,"train","train_time.txt"),epoch)
        write_param_to_txt(train_loss,os.path.join(out_else_dir,"train","train_loss.txt"),epoch)
        write_param_to_txt(valid_loss,os.path.join(out_else_dir,"valid","valid_loss.txt"),epoch)
        write_param_to_txt(test_loss,os.path.join(out_else_dir,"test","test_loss.txt"),epoch)

        write_param_to_txt(train_all_rmse,os.path.join(out_else_dir,"train","ALL_rmse.txt"),epoch)
        write_param_to_txt(valid_all_rmse,os.path.join(out_else_dir,"valid","ALL_rmse.txt"),epoch)
        write_param_to_txt(test_all_rmse,os.path.join(out_else_dir,"test","ALL_rmse.txt"),epoch)

        for j in range(6):
            for i in range(2):
                basename = position_dict[(j,i)]+"_rmse.txt"
                write_param_to_txt(train_part_rmse[j,i],os.path.join(out_else_dir,"train",basename),epoch)
                write_param_to_txt(valid_part_rmse[j,i],os.path.join(out_else_dir,"valid",basename),epoch)
                write_param_to_txt(test_part_rmse[j,i],os.path.join(out_else_dir,"test",basename),epoch)


        train_loss_ary.append(train_loss),valid_loss_ary.append(valid_loss),test_loss_ary.append(test_loss)
        train_rmse_ary.append(train_all_rmse),valid_rmse_ary.append(valid_all_rmse),test_rmse_ary.append(test_all_rmse)
        train_part_rmse_ary.append(train_part_rmse),valid_part_rmse_ary.append(valid_part_rmse),test_part_rmse_ary.append(test_part_rmse)

    train_part_rmse_ary,valid_part_rmse_ary,test_part_rmse_ary = list2torch(train_part_rmse_ary),list2torch(valid_part_rmse_ary),list2torch(test_part_rmse_ary)
    train_rmse_ary ,valid_rmse_ary,test_rmse_ary= torch.Tensor(train_rmse_ary),torch.Tensor(valid_rmse_ary),torch.Tensor(test_rmse_ary)
    train_loss_ary ,valid_loss_ary,test_loss_ary= torch.Tensor(train_loss_ary),torch.Tensor(valid_loss_ary),torch.Tensor(test_loss_ary)

    """
    ベストなモデルを作成
    """
    best_model_one,best_epoch = get_best_model(valid_rmse_ary,model_ary)
    save_model(best_model_one,os.path.join(out_one_dir,"ALL_ep"+str(best_epoch)+".pth"))
    best_model_sepa_dict = {}
    for j in range(6):
        for i in range(2):
            best_model_sepa,best_epoch = get_best_model(valid_part_rmse_ary[:,j,i],model_ary)
            best_model_sepa_dict[(j,i)] = copy.deepcopy(best_model_sepa)
            model_filename = os.path.join(out_sepa_dir,position_dict[(j,i)]+"_ep"+str(best_epoch)+".pth")
            save_model(best_model_sepa_dict[(j,i)],model_filename)

    """
    最終テスト
    """
    _,_,_,train_all_rmse,train_part_rmse,_ = learn(train_loader,best_model_one,None,loss_function,mode="FINAL_TEST",normalize=normalize,epoch="###")
    _,_,_,valid_all_rmse,valid_part_rmse,_ = learn(valid_loader,best_model_one,None,loss_function,mode="FINAL_TEST",normalize=normalize,epoch="###")
    _,_,_,test_all_rmse,test_part_rmse,_ = learn(test_loader,best_model_one,None,loss_function,mode="FINAL_TEST",normalize=normalize,epoch="###")

    write_final(train_all_rmse,train_part_rmse,os.path.join(out_result_one_dir,"train_rmse.txt"))
    write_final(valid_all_rmse,valid_part_rmse,os.path.join(out_result_one_dir,"valid_rmse.txt"))
    write_final(test_all_rmse,test_part_rmse,os.path.join(out_result_one_dir,"test_rmse.txt"))

    test_best_model(test_loader,best_model_sepa_dict,out_result_test_dir,normalize,save=True)
    test_best_model(train_loader,best_model_sepa_dict,out_result_train_dir,normalize,save=False)
    test_best_model(valid_loader,best_model_sepa_dict,out_result_valid_dir,normalize,save=True)

    """
    画像を保存する(png)
    """
    epoch_loss_ary = [train_loss_ary,valid_loss_ary,test_loss_ary]
    all_epoch_rmse_ary = [train_rmse_ary,valid_rmse_ary,test_rmse_ary]

    save_graph(epoch_loss_ary,os.path.join(out_else_png_dir,"loss.png"))
    save_graph(all_epoch_rmse_ary,os.path.join(out_else_png_dir,"rmse.png"))

    for j in range(6):
        for i in range(2):
            epoch_rmse_ary = [train_part_rmse_ary[:,j,i],valid_part_rmse_ary[:,j,i],test_part_rmse_ary[:,j,i]]
            save_graph(epoch_rmse_ary,os.path.join(out_else_png_dir,"rmse_"+position_dict[(j,i)]+".png"))



def save_graph(in_list,out_filename):
    train_param,valid_param,test_param = in_list

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    x = np.arange(1,len(train_param)+1)
    ax1.set_xlabel("Epoch")
    ax1.plot(x,train_param.cpu(),label="train",color="orange") 
    ax1.plot(x,valid_param.cpu(),label="valid",color="lime") 
    ax1.plot(x,test_param.cpu(),label="test",color="skyblue")
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend()
    plt.tight_layout()

    fig.savefig(out_filename)
    plt.close()










def test_best_model(loader,model_dict,out_dir:str,normalize:str,save=True):
    dataset = loader.dataset

    #プログレスバーの設定
    bar = tqdm(total = len(loader),leave=False)
    #bar.set_description(mode)
    bar.set_description("SEPA_FINAL_TEST")

    predict_ema_dict = {}
    raw_ema_dict = {}
    
    with torch.no_grad():

        for ipa,raw_ema,normalized_ema,ema_T,_,files in loader:

            
            ipa,raw_ema,normalized_ema = change_to_gpu(ipa),change_to_gpu(raw_ema),change_to_gpu(normalized_ema)
            ema_T= change_to_gpu(ema_T)

            predict_ema_per_file = []
            raw_ema_per_file = []

            current_batch = ipa.shape[0]
            
            for batch in range(current_batch):
                predict_ema_per_file.append(change_to_gpu(torch.zeros(ema_T[batch],6,2)))
                raw_ema_per_file.append(change_to_gpu(torch.zeros(ema_T[batch],6,2)))

            for j in range(6):
                for i in range(2):
                    model_dict[(j,i)].eval()
                    predict_ema_part = model_dict[(j,i)](ipa)[:,:,j,i]
                    for batch in range(current_batch):
                        predict_ema_per_file[batch][:,j,i] = predict_ema_part[batch,:ema_T[batch]]
                        raw_ema_per_file[batch][:,j,i] = raw_ema[batch,:ema_T[batch],j,i]

            for batch in range(current_batch):
                #逆変換
                if normalize=="stand":
                        predict_ema_dict[files[batch]] = dataset.inverse_stand(predict_ema_per_file[batch])
                elif normalize=="min-max":
                        predict_ema_dict[files[batch]]= dataset.inverse_minmax(predict_ema_per_file[batch])
                raw_ema_dict[files[batch]] = raw_ema_per_file[batch]
            bar.update(1)
        bar.close()

        """
        RMSEを計算する
        """
        #{file: part(6,2),all(1),file:...}
        rmse_per_file = {}
        all_predict_ema = []
        all_raw_ema = []
        for key in predict_ema_dict:
            rmse_per_file[key] = rmse(predict_ema_dict[key],raw_ema_dict[key])
            all_predict_ema.append(predict_ema_dict[key])
            all_raw_ema.append(raw_ema_dict[key])
            if save==True:
                #生成されたemaデータを保存
                save_ema(predict_ema_dict[key],out_filename=os.path.join(out_dir,key))

        #ファイルごとのrmseを保存
        write_csvfile(rmse_per_file,os.path.join(out_dir,"rmse_per_file.csv"))

        all_predict_ema = torch.cat(all_predict_ema,0)
        all_raw_ema = torch.cat(all_raw_ema,0)
        
        part_rmse,all_rmse = rmse(all_predict_ema,all_raw_ema)

        write_final(all_rmse,part_rmse,out_filename=os.path.join(out_dir,"rmse.txt"))

                    
def write_csvfile(rmse_per_file,out_filename):
    fieldname = ["file","average"]
    for j in range(6):
        for i in range(2):
            fieldname.append(position_dict[(j,i)])
    data = []
    for file in rmse_per_file:
        _dict = {"file":file}
        part_rmse,all_rmse = rmse_per_file[file]
        _dict["average"] = round(all_rmse.item(),3)

        for j in range(6):
            for i in range(2):
                _dict[position_dict[(j,i)]] =round(part_rmse[j,i].item(),3)
        data.append(_dict)

    with open(out_filename,mode="w",newline='') as f:
        writer = csv.DictWriter(f, fieldnames = fieldname)
        writer.writeheader()
        writer.writerows(data)


                

"""
生成されたemaデータを保存する
"""
def save_ema(ema_data,out_filename):
    with open(out_filename,mode="w") as f:
        for duration in range(len(ema_data)):
            for j in range(6):
                for i in range(2):
                    f.write(f"%f " % (ema_data[duration,j,i]))

            f.write("\n")







def write_final(all_rmse,part_rmse,out_filename):
    with open(out_filename,mode="w") as f:
        for j in range(6):
            for i in range(2):
                f.write(f"%s:%f\n" % (position_dict[(j,i)],part_rmse[j,i]))

        f.write("-----------------\n")
        f.write(f"ALL:%f" % (all_rmse))


"""
モデルを保存
"""
def save_model(model,out_filename:str):
    torch.save(model.state_dict(),out_filename)

"""
validation結果からベストなモデルを選択
"""
def get_best_model(valid_epoch_list,model_ary):
    best_epoch = torch.argmin(valid_epoch_list)
    return model_ary[best_epoch],best_epoch.item()

"""
[torch.Tensor(),torch.Tensor()....] -> torch.Tensor()
"""
def list2torch(arrs):
    return torch.cat(arrs).reshape(len(arrs), *arrs[0].shape)

"""
パラメータをtxtファイルに記入
"""
def write_param_to_txt(param,out_filename:str,epoch:int):
    os.makedirs(os.path.dirname(out_filename),exist_ok=True)
    with open(out_filename,mode="a") as f:
        f.write(f"ep%d:%f\n" % (epoch,param))

"""
loader:データセットをロードしたもの
model : モデル
optimizer:modelのoptimizer
loss_func:ロス関数
mode : "train","valid","test"
normalize : "min-max" or "stand"
epoch : "1/50"
"""
def learn(loader,model,optimizer,loss_func,mode:str,normalize:str,epoch:str):
    dataset = loader.dataset

    #プログレスバーの設定
    bar = tqdm(total = len(loader),leave=False)
    #bar.set_description(mode)
    bar.set_description(mode+"_"+epoch)

    loss_ary = []

    

    time_ary = []

    predict_ema_ary = []
    raw_ema_ary = []

    for ipa,raw_ema,normalized_ema,ema_T,_,_ in loader:
        start_time = time.time()
        ipa,raw_ema,normalized_ema = change_to_gpu(ipa),change_to_gpu(raw_ema),change_to_gpu(normalized_ema)
        ema_T= change_to_gpu(ema_T)

        if mode=="train":
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        #predict_ema : (batch,2420,6,2)
        predict_ema = model(ipa)

        loss = loss_func(predict_ema,normalized_ema)

        if mode=="train":
            loss.backward()
            optimizer.step()

        time_ary.append(time.time()-start_time)
        loss_ary.append(loss.item())

        with torch.no_grad():
            current_batch = ipa.shape[0]
            for batch in range(current_batch):
                if normalize=="stand":
                    denormalized_predict_ema = dataset.inverse_stand(predict_ema[batch,:ema_T[batch]])
                elif normalize=="min-max":
                    denormalized_predict_ema = dataset.inverse_minmax(predict_ema[batch,:ema_T[batch]])
                predict_ema_ary.append(denormalized_predict_ema)
                raw_ema_ary.append(raw_ema[batch,:ema_T[batch]])
                

        bar.update(1)
    bar.close()

    ###RMSE
    predict_ema_ary = change_to_gpu(torch.cat(predict_ema_ary,0))
    raw_ema_ary = change_to_gpu(torch.cat(raw_ema_ary,0))
    #part_rmse : (6,2) all_rmse : (1)
    part_rmse,all_rmse = rmse(predict_ema_ary,raw_ema_ary)

    loss_avg = sum(loss_ary) / len(loss_ary)

    sum_time = sum(time_ary)


    return model,optimizer,loss_avg,all_rmse,part_rmse,sum_time

def rmse(pre,tar):
    s = torch.sum((pre-tar)**2,0)
    part_rmse = torch.sqrt((1/len(pre))*s)
    all_rmse = torch.mean(part_rmse)
    return part_rmse,all_rmse










    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-yaml', '--inyaml',default="set.yaml")
    args = parser.parse_args()

    yaml_file = args.inyaml
    yaml_dict = read_yamlfile(yaml_file)

    #text_dataset
    text_dataset = yaml_dict["dataset"]["text_dataset"]

    #ema_dataset
    ema_dataset = yaml_dict["dataset"]["ema_dataset"]
    

    #train_data
    train_txt = yaml_dict["dataset"]["train_txt"]
    train_basenames = read_txtfile(train_txt)
    train_ipa_files = make_filepath(text_dataset,train_basenames)
    train_ema_files = make_filepath(ema_dataset,[b.replace(".TextGrid",".ema") for b in train_basenames])
    train_files = concat_file(train_ipa_files,train_ema_files)


    #valid_data
    valid_txt = yaml_dict["dataset"]["valid_txt"]
    valid_basenames = read_txtfile(valid_txt)
    valid_ipa_files = make_filepath(text_dataset,valid_basenames)
    valid_ema_files = make_filepath(ema_dataset,[b.replace(".TextGrid",".ema") for b in valid_basenames])
    valid_files = concat_file(valid_ipa_files,valid_ema_files)

    #test_data
    test_txt = yaml_dict["dataset"]["test_txt"]
    test_basenames = read_txtfile(test_txt)
    test_ipa_files = make_filepath(text_dataset,test_basenames)
    test_ema_files = make_filepath(ema_dataset,[b.replace(".TextGrid",".ema") for b in test_basenames])
    test_files = concat_file(test_ipa_files,test_ema_files)

    train([train_files,valid_files,test_files],yaml_dict)


    

