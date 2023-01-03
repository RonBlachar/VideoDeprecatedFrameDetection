from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import RocCurveDisplay, roc_curve, precision_recall_curve, classification_report
import pandas as pd
import config as conf
import numpy as np
from IPython.display import display


def evaluate_model(test_target, test_pred, test_target_label):
    metrics = \
        classification_report(test_target, test_pred, labels=[1], target_names=['Noisy Images'], output_dict=True)[
            'Noisy Images']
    print('Noise Detection Metrics:\n')
    print(f'Precision: {metrics["precision"]:.3f}')
    print(f'Recall: {metrics["recall"]:.3f}')
    print(f'F1-score: {metrics["f1-score"]:.3f}')
    print(f'Support: {metrics["support"]}')
    print('------------\n Error Analysis:\n')
    print('\nRecall per Noise Type:\n')
    display(recall_per_target_label(test_target, test_pred, test_target_label))
    print('\nNoise Type Distribution over test:\n')
    plt.hist(test_target_label[test_target_label != 'OriginalData'])


def recall_per_target_label(test_target, test_pred, test_target_label):
    df = pd.DataFrame(columns=['noise_type', 'recall'])
    for noise in conf.NOISE_TYPES:
        y_true = np.extract(test_target_label == noise, test_target)
        y_pred = np.extract(test_target_label == noise, test_pred)
        df.loc[len(df)] = [noise, recall_score(y_true, y_pred)]
    return df
