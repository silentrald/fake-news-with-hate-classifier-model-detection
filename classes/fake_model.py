import torch

from classes.base_model import BaseModel
from classes.article_splitter import ArticleSplitter


class FakeModel(torch.nn.Module):
    _splitter: ArticleSplitter
    _fake_model: BaseModel
    _hate_model: BaseModel
    _device: str
    threshold: float

    def __init__(self,
                 fake_checkpoint: str,
                 hate_checkpoint: str,
                 device: str = 'cpu',
                 threshold: float = 0.75):
        super(FakeModel, self).__init__()

        self.threshold = threshold
        self._device = device

        self._splitter = ArticleSplitter()

        fake_state_dict, fake_config = torch.load(fake_checkpoint)
        self._fake_model = BaseModel(
            model=fake_config['pretrained'],
            msl=fake_config['msl'],
            truncation=fake_config['truncation'],
            device=device
        )
        self._fake_model.load_state_dict(state_dict=fake_state_dict)

        hate_state_dict, hate_config = torch.load(hate_checkpoint)
        self._hate_model = BaseModel(
            model=hate_config['pretrained'],
            msl=hate_config['msl'],
            truncation=hate_config['truncation'],
            device=device
        )
        self._hate_model.load_state_dict(state_dict=hate_state_dict)

    def forward(self, text: str) -> list:
        fake_outputs = self._fake_model([text])
        fake_pred = torch.max(fake_outputs.data, 1)[1][0]

        confidence = fake_outputs.max()
        if confidence > self.threshold:
            return fake_pred

        splits = self._splitter.split([text])[0]
        hate_outputs = self._hate_model(splits)
        hate = torch.max(hate_outputs.data, 1)[1].sum() / len(hate_outputs)

        hate_pred = 1 if hate > 0.5 else 0
        if fake_pred == hate_pred:
            return fake_pred

        if fake_pred == 0:
            confidence = 1 - confidence
        pred = confidence + hate - 0.5
        return 1 if pred > 0.5 else 0
