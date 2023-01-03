import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import roc_curve, auc


def predict(reconstruction_loss, threshold):
    return tf.math.greater(reconstruction_loss, threshold)


def get_reconstruction_loss(model, data, loss=tf.keras.losses.MeanSquaredError(), plot_reconstruction_loss=False):
    reconstructions = model.predict(data)
    reconstruction_loss = []
    for i, frames in enumerate(zip(data, reconstructions)):
        curr_loss = loss(frames[0], frames[1]).numpy()
        reconstruction_loss.append(curr_loss)
    if plot_reconstruction_loss:
        plt.hist(reconstruction_loss, bins=50)
        plt.xlabel("reconstruction loss")
        plt.ylabel("No of examples")
        plt.show()
    return reconstruction_loss


def get_threshold_by_loss_distribution(reconstruction_loss, quantile=0.99, std_coefficient=0):
    """Implements two strategies:
    1. selecting by a given quantile
    2. selecting by mean(reconstruction_loss) + std_coefficient*std(reconstruction_loss)
    If std_coefficient=0 will use quantile, else will use std_coefficient
    """
    threshold = np.mean(reconstruction_loss) + std_coefficient * np.std(reconstruction_loss) \
        if std_coefficient else np.quantile(reconstruction_loss, quantile)
    print(f'\nThreshold={threshold}')
    return threshold


def get_threshold_by_roc_auc(target, reconstruction_loss, plot_roc_auc_curve=True):
    """ Currently uses naive strategy where it TPR and FPR are weight equally"""
    fpr, tpr, thresholds = plot_roc_auc(target, reconstruction_loss) if plot_roc_auc_curve else roc_curve(target,
                                                                                                          reconstruction_loss)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f'\nThreshold={optimal_threshold}')
    return optimal_threshold


def plot_roc_auc(target, reconstruction_loss):
    """
    Plots ROC-AUC with interactive threshold hover
    Returns:
        (fpr, tpr, thresholds)
    """
    fpr, tpr, thresholds = roc_curve(target, reconstruction_loss)
    df = pd.DataFrame({
        'FalsePositiveRate': fpr,
        'TruePositiveRate': tpr,
        'Threshold': thresholds
    })
    fig = px.area(
        data_frame=df,
        x='FalsePositiveRate',
        y='TruePositiveRate',
        title=f'ROC Curve AUC={auc(fpr, tpr):.4f}',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        hover_data=['Threshold'],
        width=800, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()
    return fpr, tpr, thresholds
