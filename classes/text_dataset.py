import torch
import transformers
from classes.transformer_model import TransformerModel


class TextDataset(torch.utils.data.Dataset):
    _articles: torch.Tensor
    _labels: list
    _size: int

    def __init__(self,
                 articles: list = None,
                 labels: list = None):
        if len(articles) != len(labels):
            raise 'Articles and labels length don\'t match'

        self._articles = articles
        self._labels = labels
        self._size = len(articles)

    def __len__(self):
        return self._size

    def __getitem__(self, i: int):
        return self._articles[i], self._labels[i]
