import logging
logging.getLogger().setLevel(logging.INFO)
import torch.optim as optim

from model import *
from data import *
from crnn import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, iterator, optimizer, criterion):
    total = 0
    correct = 0
    model.train()
    for epoch in range(N_EPOCHS):
        for i, batch in enumerate(iterator):

            data, label = batch.news, batch.label - 1
            total += label.size(0)
            optimizer.zero_grad()
            outputs, _ = model(data)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == label).sum().item()
            logging.info("train epoch=" + str(epoch) + ",batch_id=" + str(i) + ",loss=" + str(loss.item() / 64)+",acc="+str(100*correct/total))
    return


def train_lstm(model, iterator, optimizer, criterion):
    total = 0
    correct = 0
    model.train()
    for epoch in range(N_EPOCHS):
        for i, batch in enumerate(iterator):

            data, label = batch.news, batch.label - 1

            total += label.size(0)
            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, label)
            _, predicted = torch.max(outputs.data, 1)

            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == label).sum().item()
            logging.info(
                "train epoch=" + str(epoch) + ",batch_id=" + str(i) + ",loss=" + str(loss.item() / 64) + ",acc=" + str(
                    100 * correct / total))
    return


def train_Bi_lstm(model, iterator, optimizer, criterion):
    model.train()
    for epoch in range(N_EPOCHS):
        total = 0
        correct = 0
        for i, batch in enumerate(iterator):

            data, label = batch.news, batch.label - 1

            total += label.size(0)
            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, label)
            _, predicted = torch.max(outputs.data, 1)

            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == label).sum().item()
            logging.info(
                "train epoch=" + str(epoch) + ",batch_id=" + str(i) + ",loss=" + str(loss.item() / 64) + ",acc=" + str(
                    100 * correct / total))
    return


def train_WordCNN(iterator, model, optimizer, criterion):
    model.train()
    print('training...')
    total = 0
    correct = 0
    for epoch in range(N_EPOCHS):
        for i, batch in enumerate(iterator):
            feature, target = batch.news, batch.label - 1
            total += target.size(0)
            optimizer.zero_grad()
            logit = model(feature)
            loss = criterion(logit, target)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(logit.data, 1)
            correct += (predicted == target).sum().item()
            logging.info(
                "train epoch=" + str(epoch) + ",batch_id=" + str(i) + ",loss=" + str(loss.item() / 64) + ",acc=" + str(
                    100 * correct / total))
    return


def train_CRNN(iterator, model, optimizer, criterion):
    model.train()
    print('training...')
    total = 0
    correct = 0
    for epoch in range(N_EPOCHS):
        for i, batch in enumerate(iterator):
            feature, target = batch.news, batch.label - 1
            total += target.size(0)
            optimizer.zero_grad()
            logit = model(feature)
            loss = criterion(logit, target)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(logit.data, 1)
            correct += (predicted == target).sum().item()
            
            logging.info(
                "train epoch=" + str(epoch) + ",batch_id=" + str(i) + ",loss=" + str(loss.item() / 64) + ",acc=" + str(
                    100 * correct / total))
    return


def model_test(net, train_iter):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(train_iter):
            data, label = batch.news, batch.label - 1
            logging.info("test batch_id=" + str(i))
            outputs = net(data)
            _, predicted = torch.max(outputs[0].data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            print('Accuracy of the network on test set: %d %%' % (100 * correct / total))



if __name__ == '__main__':
    sentence_max_size = 100
    BATCH_SIZE = 64
    train_csv = "data/train.csv"
    train_iter, test_iter, NEWS = get_data_iter(train_csv, sentence_max_size, BATCH_SIZE, device)
    N_EPOCHS = 200
    Input_Dim = len(NEWS.vocab)
    Embedding_Dim = 100
    Hidden_Dim = 100
    Out_Dim = 2
    N_Layers = 2
    Bidirectional = False
    Dropout = 0
    net_dir = "model/imdb_bi_att_lstm.pth"
    model_dir = "model/imdb_lstm.pth"
    word_dir = "model/imdb_wordcnn.pth"
    word_adv_dir = "model/imdb_wordcnn_re_adv.pth"
    Bi_dir = "model/imdb_bi_lstm.pth"

    Pad_idx = NEWS.vocab.stoi[NEWS.pad_token]

    vocab = NEWS.vocab

    attention = Attention(Hidden_Dim)

    net = Bi_att_Lstm(vocab, Embedding_Dim, Hidden_Dim, N_Layers, attention)

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    net = net.to(device)

    net.embed.weight.requires_grad = False
    criterion = criterion.to(device)

    train(net, train_iter, optimizer, criterion)

