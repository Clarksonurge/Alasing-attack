from data import *
from model import LSTM
from test import FastText
from trian import predict_sentiment, device
import os
import torch
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        text = text.to(device)

        text = text.view(-1, 1)

        output = model(text)

        return output.argmax(1).item() + 1


if __name__ == '__main__':
    ag_news_label = {1: "World",
                     2: "Sports",
                     3: "Business",
                     4: "Sci/Tec"}

    sentence_max_size = 50
    BATCH_SIZE = 256
    train_csv = "data/train.csv"
    train_iter, test_iter, NEWS = get_data_iter(train_csv, sentence_max_size, BATCH_SIZE, device)
    vocab = NEWS.vocab
    ex_text_str ="AP - Romario celebrated his retirement from international soccer by scoring two goals to give " \
                 "Brazil's 1994 World Cup team a 2-1 victory Wednesday night over Mexico's World Cup team from that " \
                 "year. "

    root_dir = os.path.abspath('.')
    model = torch.load("{0}\\model\\ag_fasttext_model.pth".format(root_dir))
    predict(ex_text_str, text_pipeline)
    print("This is a %s news" % ag_news_label[ predict( ex_text_str, text_pipeline)])
