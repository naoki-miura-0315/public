
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

#  単語埋め込み次元数
dw = 300
#  隠れ状態ベクトルの次元数
dh = 50

input_size = len(word_2_id.values()) + 2 #語彙サイズ＋１尚且つ０のやつもあるので＋１,"-1"の分で＋１
padding_idx = len(word_2_id.values()) +1
emb_size = dw
hidden_size = dh
output_size = 4


#RNNを用いたモデル定義を行う

class My_rnn(nn.Module):
    def __init__(self):
        super().__init__()
        #  埋め込みの定義
        self.emb = nn.Embedding.from_pretrained(weights, padding_idx)
        #  RNNの定義
        self.rnn = nn.RNN(emb_size, hidden_size,batch_first=True)#batch_size, seq_len, dim
        #  出力層の定義
        self.linear = nn.Linear(hidden_size, output_size)
        #  softmax層
        #self.softmax = nn.Softmax(dim=1)
        
    #  予測関数の定義    
    def forward(self, x, h0=None):
        x = self.emb(x)#  埋め込み層
        
        y, h = self.rnn(x, h0)#  xと初期の隠れ状態h0でRNNモデルを実行
        #print(y.size())
        y = y[:, -1, :]#  時間経過の中で最後のステップを取得  #系列長、バッチサイズ、次元数
        y = self.linear(y)#  全結合層 →　４次元へ
        #y = self.softmax(y, dim=1)#  ４次元を確率へ
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

batch_size=50

train_dataset = MyDataset('train.txt')
#valid_dataset = MyDataset('data/valid.txt')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_sec)
#valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=pad_sec)

#  学習用の関数を定義



def training(num_epochs):
    model = My_rnn()
    train_avg_loss = []
    test_avg_loss = []
    train_acc = []
    test_acc = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    model.train()

    # 初期設定
    # GPUが使えるか確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name())
    print("使用デバイス:", device)

    model.to(device)

    for epoch in tqdm(range(num_epochs)):
        total_train_loss = 0
        total_test_loss = 0  
        train_pred_y_list = []
        true_y_list = []
        
        for x_train_tensor, y_label in train_dataloader:
            x_train_tensor = x_train_tensor.to(device)
            y_label = y_label.to(device)
            optimizer.zero_grad()
            pred_y = model(x_train_tensor)
            loss = criterion(pred_y, y_label)
            loss.backward()
            optimizer.step()
            #予想したラベルを貯めておく
            train_pred_y = torch.argmax(pred_y, dim=1)
            train_pred_y_list.append(train_pred_y)
            true_y_list.append(y_label)
            
            total_train_loss = total_train_loss + loss.item()
            
        
        #  logの表示
        print(f'epoch:{epoch}')
        avg_loss = total_train_loss / len(train_pred_y_list)
        train_avg_loss.append(avg_loss)
        #print(pred_y)
        #print(train_pred_y)
        #print(train_pred_y_list[0])
        #print(true_y_list[0])
        train_y_pre = torch.cat(train_pred_y_list, dim = 0)
        true_y_tensor = torch.cat(true_y_list, dim = 0)
        train_y_pre = train_y_pre.cpu()
        true_y_tensor = true_y_tensor.cpu()
        acc_score = accuracy_score(train_y_pre, true_y_tensor)
        train_acc.append(acc_score)
        print(f'acc:{acc_score}')
        print(f'loss:{avg_loss}')

num_epoch = 10


#  main
if __name__ == '__main__':
    print('batch_size=50')
    print("num_epoch = 10")

    training(10)