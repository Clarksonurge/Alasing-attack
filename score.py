import torch
import numpy as np
from attack import *

def self_attscore(net, text):
    pre, self_att_score = net(text.view((1, -1)))
    self_att_score = self_att_score.cpu( )
    self_att_score = (self_att_score[-1].detach().numpy())[0]
    text = text.cpu().numpy().tolist()
    text = text[0]
    score_list = []
    for i in range(len(self_att_score)):
        input_t = text[:i] + text[i + 1:]
        input_t = torch.LongTensor(input_t).to(device='cuda')
        pre, _ = net(input_t.view((1, -1)))
        order = torch.max(pre, -1)[1].view(-1)
        pre = pre.gather(1, order.view(-1, 1)).view(-1)
        pre = pre[-1].cpu().detach().numpy()
        score_combine = np.exp(self_att_score[i])+2*abs(pre)
        score_list.append(score_combine)
    self_att_score = np.argsort(score_list)
    return pre, self_att_score


def word_scoring(model, sentence, word_score):
    """sentence是词语的列表"""
    sentence = torch.tensor(sentence).to(device='cuda')
    pre = model(sentence.view((1, -1)))

    pre = torch.exp(pre)
    label = pre.argmax(1).item() + 1
    order = torch.min(pre, 1)[1].view(-1)
    pre = pre.gather(1, order.view(-1, 1)).view(-1)
    pre = pre[-1].cpu().detach().numpy()

    word_score.append(abs(pre))

    return label, word_score
