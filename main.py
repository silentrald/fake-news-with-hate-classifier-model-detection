from classes.fake_model import FakeModel
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print('Loading Data')
    df = pd.read_csv('./data/dataset-updated-test.csv')

    fake_models = [
        'fake/1c.pt',
        'fake/2c.pt',
        'fake/3c.pt',
        'fake/4c.pt',
        'fake/7c.pt',
        'fake/8c.pt',
        'fake/9c.pt',
        'fake/10c.pt'
    ]
    hate_models = [
        'hate/19c.pt',
        'hate/20c.pt',
        'hate/21c.pt',
        'hate/22c.pt',
        'hate/23c.pt',
        'hate/24c.pt'
    ]
    thresholds = [0.97, 0.975, 0.98, 0.985, 0.99, 0.991,
                  0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]

    print('Processing Text')
    data = df[['article', 'fake']].sample(frac=1)

    total = len(data)
    for fm in fake_models:
        for hm in hate_models:
            print('===============')
            print('Fake Checkpoint:', fm)
            print('Hate Checkpoint:', hm)
            model = FakeModel(
                fake_checkpoint=fm,
                hate_checkpoint=hm,
                device='cuda:0'
            )
            model.eval()

            for threshold in thresholds:
                model.threshold = threshold
                correct = 0

                with torch.no_grad():
                    for _, row in tqdm(data.iterrows(), total=df.shape[0]):
                        fake = model(row.article)
                        correct += 1 if int(fake) == row.fake else 0

                print('>>>')
                print('Threshold:', threshold)
                print('Accuracy:', round(correct / total, 4))


if __name__ == "__main__":
    main()
