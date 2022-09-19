import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import time
from score import *
from data import *
import os
from torchtext.data.utils import get_tokenizer
import numpy as np
from model import *
import csv
import random

tokenizer = get_tokenizer('basic_english')
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

root_dir = os.path.abspath('.')
stop_words = stopwords.words("english")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clone_runoob(li1):
    li_copy = []
    li_copy.extend(li1)
    return li_copy


def re_process(text, label, Relist, text_d):
    adv_data = []
    sente = clone_runoob(text)
    text_d = text_d[0].cpu().numpy().tolist()
    for i in range(len(Relist)):
        sent = clone_runoob(sente)
        if sente[Relist[i]] != text_d[Relist[i]]:
            sente[Relist[i]] = text_d[Relist[i]]
            labeled = predict_list(sente)
            if label == labeled:
                print("这ge是我们之后得到的:")
                print("attack success label = {}".format(1 if label == 2 else 2))
                sent = [vocab.itos[word] for word in sent]
                sent = " ".join(sent)
                adv_data.append([1 if labeled == 1 else 2, sent])
                with open("Adversarial_data/after_delete_data.csv", "a", newline='', encoding = 'UTF-8') as f:
                    writer = csv.writer(f, delimiter=",")
                    for j in adv_data:
                        writer.writerow(j)
                return ((len(Relist) - i) / len(text_d))
    return


def preprocessing(text, stem=False):
    stemmer = SnowballStemmer('english')
    text = " ".join(text)
    text = text.replace("<br />", "")
    text_cleaning_re = '@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+'
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split(" "):
        if stem:
            tokens.append(stemmer.stem(token))
        else:
            tokens.append(token)
    return tokens


def pearson(W, x, k):
    sumW = torch.sum(W, dim=1) + 1e-9
    sumx = torch.sum(x)
    sumWsq = torch.sum(W * W, dim=1) + 1e-9
    sumxsq = torch.sum(x * x)
    sumWx = torch.matmul(W, x.view((-1,)))
    up = sumWx - sumW * sumx / 40000
    down = (sumWsq - torch.pow(sumWsq, 2) / 40000).sqrt() * (sumxsq - torch.pow(sumxsq, 2) / 40000).sqrt()
    r = up / down
    _, topk = torch.topk(r, k=k)
    topk = topk.cpu().numpy()
    return topk, [r[i].item() for i in topk]


def knn(W, x, k):
    cos = torch.matmul(W, x.view((-1,))) / (
            (torch.sum(W * W, dim=1) + 1e-9).sqrt() * torch.sum(x * x).sqrt())
    _, topk = torch.topk(cos, k=k)
    topk = topk.cpu().numpy()
    return topk, [cos[i].item() for i in topk]


def get_similar_tokens(query_token, k, embed):
    topk, cos = pearson(embed.vectors,
                        embed.vectors[query_token], k + 1)
    # for i, c in zip(topk[1:], cos[1:]):  # 除去输入词
    #     print('cosine sim=%.4f: %s' % (c, (embed.itos[i])))
    #     print(c)
    return topk[1:]


def attack(label, text, k, x):
    Relist = []

    text_d = clone_runoob(text)
    pre, self_att_score = self_attscore(net, text)

    text = text.cpu().numpy().tolist()[0]

    perturbation_rate = 0.0
    # try:
    for i in range(len(self_att_score)):
        word = text[self_att_score[i]]
        Relist.append(self_att_score[i])
        try:
            topic = get_similar_tokens(word, k, vocab)
        except:
            continue
        else:
            word_score = []
            for j in range(k):
                if vocab.itos[topic[j]] in stop_words:
                    continue
                else:
                    text[self_att_score[i]] = topic[j]
                    labeled, word_score = word_scoring(bi_lstm, text, word_score)
                if label != labeled:

                    print("attack success label = %s" % labeled)
                    x = x + 1
                    perturbation_rate = re_process(text, label, Relist, text_d)
                    break
            else:
                if word_score:
                    text[self_att_score[i]] = topic[np.argmax(word_score)]
                continue
            break
    return x, perturbation_rate



def predict(text):
    with torch.no_grad():
        text = text.to(device)
        output = model(text)
        return output.argmax(1).item() + 1

def predict_list(text):
    with torch.no_grad():
        text = torch.tensor(text).clone()
        text = text.view(1, -1)
        text = text.to(device)
        output = bi_lstm(text)
        return output.argmax(1).item() + 1


if __name__ == '__main__':
    perturbation_rate = 0.0
    BATCH_SIZE = 1
    train_csv = "data/train.csv"

    train_iter, test_iter, NEWS = get_adv_data_iter(train_csv, BATCH_SIZE, device)
    vocab = NEWS.vocab

    root_dir = os.path.abspath('.')
    model = torch.load("{0}\\model\\imdb_lstm.pth".format(root_dir))
    net = torch.load("{0}\\model\\imdb_bi_att_lstm.pth".format(root_dir))

    model.eval()
    net.eval()

    x = 0
    d = 0
    w = 0

    for _, batch in enumerate(test_iter):
        text = batch.news
        labo = batch.label
        w = w + 1
        label = predict(language)
        if labo == label:
            d = d + 1
            attack(label, text, 50, x)
        else:
            continue

