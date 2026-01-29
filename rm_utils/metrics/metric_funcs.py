from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import roc_auc_score
from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred, sample_weight=None):
    """MAPE metric"""
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape, weights=sample_weight, axis=0)

    return output_errors


def shortfall(y_true, y_pred):
    """Shortfall metric"""
    return 1 - np.sum(y_pred) / np.sum(y_true)


def pearson_corr(y_true, y_pred):
    """Pearson correlation coefficient"""
    corr_coef = np.corrcoef(y_true, y_pred)
    return corr_coef[0, 1]


def pearson_nan_corr(y_true, y_pred):
    """Pearson correlation coefficient ignoring nan values"""
    corr_coef = np.ma.corrcoef(np.ma.masked_invalid(y_true), np.ma.masked_invalid(y_pred))

    return corr_coef[0, 1]

def spearman_corr(y_true, y_pred):
    """Pearson correlation coefficient ignoring nan values"""
    corr_coef = np.ma.corrcoef(np.ma.masked_invalid(y_true), np.ma.masked_invalid(y_pred))

    return corr_coef[0, 1]


def root_mse(y_true, y_pred):
    """Root MSE"""
    return mean_squared_error(y_true, y_pred)


def gini_score(y_true, y_pred):
    """Gini score"""
    roc_auc = roc_auc_score(y_true, y_pred)
    return roc_auc * 2 - 1

def gini_score_safe(y_true, y_pred):
    """Gini score for unseasoned data"""
    if len(np.unique(y_true)) != 2:
        return 0
    roc_auc = roc_auc_score(y_true, y_pred)
    return roc_auc * 2 - 1

def roc_auc_score_nan(y_true, y_pred):
    """ROC_AUC score for bad/unseasoned data.
    Calcs only where preds & labels is not None.
    Returns nans instead or raising errors.
    """
    notna_mask = (~np.isnan(y_pred)) & (~np.isnan(y_true))
    y_true, y_pred = y_true[notna_mask], y_pred[notna_mask]
    if (len(y_true) < 3) or (len(np.unique(y_true)) != 2):
        return np.nan
    return roc_auc_score(y_true, y_pred)

def gini_score_nan(y_true, y_pred):
    """Gini score for bad/unseasoned data.
    Calcs only where preds & labels is not None.
    Returns nans instead or raising errors.
    """
    roc_auc = roc_auc_score_nan(y_true, y_pred)
    return roc_auc * 2 - 1
