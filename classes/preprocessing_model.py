import torch


class PreprocessingModel():
    threshold: float

    def __init__(self, threshold=0.75):
        super(PreprocessingModel, self).__init__()

        self.threshold = threshold

    def forward(self, fake: torch.Tensor, hate: torch.Tensor) -> torch.Tensor:
        fake_pred = torch.max(fake.data, 1)[1]
        hate_pred = 1 if hate > 0.5 else 0

        confidence = fake.max()
        if confidence > self.threshold:
            return fake_pred

        if fake_pred == hate_pred:
            return fake_pred

        pred = confidence + hate - 0.5
        return 1 if pred > 0.5 else 0
