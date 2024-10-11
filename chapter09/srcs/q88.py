
from collections import Counter
import pickle
import ast
import torch.nn.utils.rnn as rnn
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import statistics


word_2_id = {}
words = []

with open('train.txt') as fi:
    for line in fi:
        sent = line.split('\t')[0]
        sent = sent.replace("'","").replace('"','').replace('.','').replace('?','')
        for word in sent.split():
            words.append(word)#  出現する単語をとりあえず入れとく
        



myfile = open('word_count.pickle', 'rb')
rmydata = pickle.load(myfile)

omomifile = open('weights.pickle','rb')
weights = pickle.load(omomifile)

#  idを付与
for i, w_and_c in enumerate(rmydata):
    word = w_and_c[0]
    count = w_and_c[1]
    if count > 1:
        word_2_id[word] = i + 1


#  word_2_idを利用してidを付与する関数
def text_2_id_list(text):
    sentence_id_list = []
    words = text.replace("'","").replace('"','').replace('.','').replace('?','')
    words = words.rstrip().split()
    for word in words:
        if word in word_2_id:
            word_id = word_2_id[word]
            sentence_id_list.append(word_id)
        else:
            sentence_id_list.append(int(0))
            
    return sentence_id_list


#ハイパラたち

dw = 300
#  隠れ状態ベクトルの次元数
dh = 50

input_size = len(word_2_id.values()) + 2 #語彙サイズ＋１尚且つ０のやつもあるので＋１,"-1"の分で＋１
padding_idx = len(word_2_id.values()) +1
emb_size = dw
hidden_size = dh
output_size = 4

out_channels = 100
kernel_size = 3
stride = 1
padding = 1


# 第１引数はインプットのチャネル（今回は1）を指定
# 自然言語処理で畳み込む場合、異なる単語分散表現（word2vecとfasttextみたいな）などを使って、
# 複数チャネルとみなす方法もあるようです。
# 第２引数はアウトプットのチャネル数で、今回は同じフィルターを2枚畳み込みたいので、2を指定
# カーネルサイズは高さ×幅を指定しており、幅は図で説明した通り、単語ベクトルの次元数5を指定
#CNNを用いたモデル定義を行う

class My_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        #  埋め込みの定義
        self.emb = nn.Embedding.from_pretrained(weights, padding_idx)
        #  CNNの定義
        self.conv = nn.Conv1d(dw, dh, kernel_size, stride, padding)# in_channels:dw, out_channels: dh
        #  出力層の定義
        self.linear = nn.Linear(dh, output_size)

        
    #  予測関数の定義    
    def forward(self, x):
        #print(x.size()) torch.Size([50, 15])
        x = self.emb(x)#  埋め込み層 
        #print(x.size()) torch.Size([50, 15, 300])
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        #print(x.size()) #torch.Size([50, 300, 15])
        conv = self.conv(x)
        #print(conv.size()) # torch.Size([50, 50, 15])
        x = F.relu(conv)
        x = F.max_pool1d(x, x.size()[2])
        #print(x.size()) # torch.Size([50, 50, 1])
        x = x.view(x.shape[0], x.shape[1])
        #print(x.size()) #  torch.Size([50, 50])
        #print(x[:, :, -1])
        y = self.linear(x) 
        #print(y.size()) # torch.Size([50, 4])
        return y



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # データの読み出し
        dataframe = []
        with open(data_path) as f:
            for line in f:
                sent = line.split('\t')
                dataframe.append(sent)
        self.dataframe = dataframe
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    # データとラベルの取得
    def __getitem__(self, idx):
        category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
        try:
            label = category_dict[self.dataframe[idx][1].rstrip()]
        except:
            print(self.dataframe[idx][1].rstrip())
        #print(label)
        #print(self.dataframe[idx][0])
        text_id = torch.tensor(text_2_id_list(self.dataframe[idx][0]), dtype=torch.int64)
        # text_id, labelの順でリターン
        return text_id, label


def pad_sec(batch):
    batch_tensor = [item[0] for item in batch]
    labels = torch.LongTensor([item[1] for item in batch])
    pad_tensors = rnn.pad_sequence(batch_tensor,batch_first=True, padding_value=padding_idx)
    return pad_tensors, labels


batch_size = 10

train_dataset = MyDataset('train.txt')
valid_dataset = MyDataset('valid.txt')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_sec)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=pad_sec)

#  学習用の関数を定義



def training_and_valid(num_epochs):
    model = My_cnn()
    train_avg_loss = []
    valid_avg_loss = []
    train_acc = []
    valid_acc = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    # 初期設定
    # GPUが使えるか確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name())
    print("使用デバイス:", device)

    model.to(device)

    for epoch in tqdm(range(num_epochs)):
        total_train_loss = 0
        total_valid_loss = 0 
        valid_loss = 0
        correct_valid = 0 
        train_pred_y_list = []
        valid_pred_y_list = []
        true_y_list = []
        valid_true_y_list = []
        
        for x_train_tensor, y_label in train_dataloader:
            x_train_tensor = x_train_tensor.to(device)
            y_label = y_label.to(device)
            optimizer.zero_grad()
            pred_y = model(x_train_tensor)
            #print(pred_y.size())
            #print(y_label.size())
            loss = criterion(pred_y, y_label)
            loss.backward()
            optimizer.step()
            #予想したラベルを貯めておく
            train_pred_y = torch.argmax(pred_y, dim=1)
            train_pred_y_list.append(train_pred_y)
            true_y_list.append(y_label)
            
            total_train_loss = total_train_loss + loss.item()


        model.eval()
        with torch.no_grad(): 
            for x_valid_tensor, y_valid_label in valid_dataloader:
                x_valid_tensor = x_valid_tensor.to(device)
                y_valid_label = y_valid_label.to(device)
                valid_pred_y = model(x_valid_tensor)
                loss = criterion(valid_pred_y, y_valid_label)
                total_valid_loss += loss.item()
                #予想したラベルを貯めておく
                valid_pred_y = torch.argmax(valid_pred_y, dim=1)
                valid_pred_y_list.append(valid_pred_y)
                valid_true_y_list.append(y_valid_label)

        #  logの表示
        print(f'epoch:{epoch}')
        tra_avg_loss = total_train_loss / len(train_pred_y_list)
        val_avg_loss = total_valid_loss / len(valid_pred_y_list)
        train_avg_loss.append(tra_avg_loss)
        valid_avg_loss.append(val_avg_loss)

        train_y_pre = torch.cat(train_pred_y_list, dim = 0)
        valid_y_pre = torch.cat(valid_pred_y_list, dim = 0)

        true_y_tensor = torch.cat(true_y_list, dim = 0)
        valid_true_y_tensor = torch.cat(valid_true_y_list, dim = 0)

        train_y_pre = train_y_pre.cpu()
        valid_y_pre = valid_y_pre.cpu()

        true_y_tensor = true_y_tensor.cpu()
        valid_true_y_tensor = valid_true_y_tensor.cpu()

        acc_score = accuracy_score(train_y_pre, true_y_tensor)
        valid_acc_score = accuracy_score(valid_y_pre, valid_true_y_tensor)

        train_acc.append(acc_score)
        valid_acc.append(valid_acc_score)

        print(f'train_acc:{acc_score}')
        print(f'valid_acc:{valid_acc_score}')

        print(f'train_avg_loss:{statistics.mean(train_avg_loss)}')
        print(f'valid_avg_loss:{statistics.mean(valid_avg_loss)}')

            
            
            



num_epoch = 5
batch_size = 10

#  main
if __name__ == '__main__':
    print(f'batch_size = {batch_size}')
    print(f'num_epoch = {num_epoch}')

    training_and_valid(num_epoch)