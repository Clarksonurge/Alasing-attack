from torchtext.data import Iterator, BucketIterator, TabularDataset
from torchtext import data
import torchtext.vocab as Vocab


def get_data_iter(train_csv,fix_length,BATCH_SIZE ,device):
    TEXT = data.Field(sequential=True, lower=True, fix_length=fix_length, batch_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    train_fields = [("label", LABEL), ("news", TEXT)]
    train = TabularDataset(path=train_csv, format="csv", fields=train_fields, skip_header=True)
    train_iter = BucketIterator(train, batch_size=BATCH_SIZE, device=device, sort_key=lambda x: len(x.news),
                                sort_within_batch=False, repeat=False)
    test_fields = [("label", LABEL), ("news", TEXT)]
    test = TabularDataset(path="data/test.csv", format="csv", fields=test_fields, skip_header=True)
    test_iter = Iterator(test, batch_size=BATCH_SIZE, device=device, sort=False, sort_within_batch=False, repeat=False)

    vectors = Vocab.Vectors(name='D:/sentiment-segment/glove/glove.6B.100d.txt')
    TEXT.build_vocab(train, vectors=vectors)

    return train_iter, test_iter, TEXT


def get_adv_data_iter(train_csv, BATCH_SIZE, device):
    TEXT = data.Field(sequential=True, lower=True, batch_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    train_fields = [("label", LABEL), ("news", TEXT)]
    train = TabularDataset(path=train_csv, format="csv", fields=train_fields, skip_header=True)
    train_iter = BucketIterator(train, batch_size=BATCH_SIZE, device=device, sort_key=lambda x: len(x.news),
                                sort_within_batch=False, repeat=False)
    test_fields = [("label", LABEL), ("news", TEXT)]
    test = TabularDataset(path="Adversarial_data/adv_bi_data.csv", format="csv", fields=test_fields, skip_header=True)
    test_iter = Iterator(test, batch_size=BATCH_SIZE, device=device, sort=False, sort_within_batch=False, repeat=False)

    vectors = Vocab.Vectors(name='D:/sentiment-segment/glove/glove.6B.100d.txt')
    TEXT.build_vocab(train, vectors=vectors)

    return train_iter, test_iter, TEXT
