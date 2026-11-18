import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, r2_score, mean_squared_error,
    mean_absolute_error, precision_recall_curve, auc
)
from scipy.stats import pearsonr
import numpy as np


def pearson_correlation_coefficient(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    return pearsonr(pred, target)[0]


def compute_regression_metrics(pred, target):
    pred = pred.flatten()
    target = target.flatten()

    metrics = {}
    metrics['MAE'] = mean_absolute_error(target.cpu(), pred.cpu())
    metrics['RMSE'] = mean_squared_error(target.cpu(), pred.cpu(), squared=False)
    metrics['R2'] = r2_score(target.cpu(), pred.cpu())
    metrics['Pearson'] = pearson_correlation_coefficient(pred, target)
    # metrics['MSE'] = mean_squared_error(target.cpu(), pred.cpu())

    return metrics


def compute_binary_classification_metrics(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred).detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    pred_label = (pred >= threshold).astype(int)

    metrics = {}
    metrics['Accuracy'] = accuracy_score(target, pred_label)
    metrics['Precision'] = precision_score(target, pred_label)
    metrics['Recall'] = recall_score(target, pred_label)
    metrics['F1'] = f1_score(target, pred_label)
    metrics['AUROC'] = roc_auc_score(target, pred)
    metrics['AUPRC'] = average_precision_score(target, pred)
    metrics['ConfusionMatrix'] = confusion_matrix(target, pred_label)

    return metrics


def compute_multiclass_classification_metrics(pred, target):
    pred_label = torch.argmax(pred, dim=1).cpu().numpy()
    target = target.cpu().numpy()

    metrics = {}
    metrics['Accuracy'] = accuracy_score(target, pred_label)
    metrics['MacroF1'] = f1_score(target, pred_label, average='macro')
    metrics['MicroF1'] = f1_score(target, pred_label, average='micro')
    metrics['WeightedF1'] = f1_score(target, pred_label, average='weighted')
    # metrics['Precision'] = precision_score(target, pred_label, average='weighted')
    # metrics['Recall'] = recall_score(target, pred_label, average='weighted')
    # metrics['ClassificationReport'] = classification_report(target, pred_label)

    return metrics


if __name__ == "__main__":
    # Simple test data, replace with actual pred and target
    pred_reg = torch.tensor([2.5, 0.0, 2.1, 1.6])
    target_reg = torch.tensor([3.0, -0.5, 2.0, 1.0])
    print("\n[Regression Metrics]")
    print(compute_regression_metrics(pred_reg, target_reg))

    pred_bin = torch.tensor([0.8, 0.2, 0.6, 0.4])
    target_bin = torch.tensor([1, 0, 1, 0])
    print("\n[Binary Classification Metrics]")
    print(compute_binary_classification_metrics(pred_bin, target_bin))

    pred_multi = torch.tensor([[0.1, 0.9], [0.6, 0.4], [0.3, 0.7]])
    target_multi = torch.tensor([1, 0, 1])
    print("\n[Multiclass Classification Metrics]")
    print(compute_multiclass_classification_metrics(pred_multi, target_multi))
