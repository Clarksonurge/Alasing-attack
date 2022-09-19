import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class smallRNN(nn.Module):
    def __init__(self, vocab, classes=2, bidirection = True, layernum=1, embedding_size =100, hiddensize = 100):
        super(smallRNN, self).__init__()
        self.embd = nn.Embedding(len(vocab), embedding_size)
        self.lstm = nn.LSTM(embedding_size, hiddensize, layernum, bidirectional=bidirection)
        self.hiddensize = hiddensize
        numdirections = 1 + bidirection
        self.hsize = numdirections * layernum
        self.linear = nn.Linear(hiddensize * numdirections, classes)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        embd = self.embd(x)
        h0 = Variable(torch.zeros(self.hsize, embd.size(0), self.hiddensize)).to("cuda")
        c0 = Variable(torch.zeros(self.hsize, embd.size(0), self.hiddensize)).to("cuda")
        x = embd.transpose(0, 1)
        x,(hn, cn) = self.lstm(x, (h0, c0))
        x = x[-1]
        x = self.log_softmax(self.linear(x))
        return x


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_hiddens, num_layers, output_dim, pad_idx, vocab):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(len(vocab), embed_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.encoder = nn.LSTM(input_size=embed_dim,
                               hidden_size=num_hiddens,
                               num_layers= num_layers,
                               bidirectional=False)
        self.out = nn.Linear(2 * num_hiddens, output_dim)

    def forward(self, inputs):
        """
        :param self:
        :param inputs:
        :return:
        """
        embeddings = self.embed(inputs.permute(1, 0))
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.out(encoding)
        return outs
    
    
class Bi_LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_hiddens, num_layers, output_dim, pad_idx, vocab):
        super(Bi_LSTM, self).__init__()
        self.embed = nn.Embedding(len(vocab), embed_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.encoder = nn.LSTM(input_size=embed_dim,
                               hidden_size=num_hiddens,
                               num_layers= num_layers,
                               bidirectional=True)
        self.out = nn.Linear(4 * num_hiddens, output_dim)

    def forward(self, inputs):
        """
        :param self:
        :param inputs:
        :return:
        """
        embeddings = self.embed(inputs.permute(1, 0))
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.out(encoding)
        return outs


class Bi_att_Lstm(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, attention):
        super(Bi_att_Lstm, self).__init__()
        self.embed = nn.Embedding(len(vocab), embed_size)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True)
        self.attention = attention
        self.num_layers = num_layers
        self.out = nn.Linear(num_hiddens * 2, 2)

    def forward(self, inputs):
        embeddings = self.embed(inputs)
        embeddings = embeddings.permute(1, 0, 2)
        outputs, hidden = self.encoder(embeddings)
        if isinstance(hidden, tuple):
            hidden = hidden[1]

        hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)

        score, linear_combination = self.attention(hidden, outputs, outputs)
        logits = self.out(linear_combination)
        return logits, score


class Attention(nn.Module):
    def __init__(self, query_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        query = query.unsqueeze(1)
        keys = keys.transpose(0, 1).transpose(1, 2)
        score = torch.bmm(query, keys)
        score = F.softmax(score.mul_(self.scale), dim=2)
        values = values.transpose(0, 1)
        linear_combination = torch.bmm(score, values).squeeze(1)
        
        return score, linear_combination


class textCNN(nn.Module):
    def __init__(self, vocab, output_dim, embed_size):
        super(textCNN, self).__init__()
        Cla = output_dim
        Ci = 1
        Knum = 150
        Ks = [3, 4, 5]

        self.embed = nn.Embedding(len(vocab), embed_size)

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (K, embed_size)) for K in Ks])

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(len(Ks) * Knum, Cla)

    def forward(self, x):
        
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc(x)

        return logit


