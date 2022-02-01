import numpy as np
from keras import backend as K


def recall_m(y_true, y_pred):
    """
    Computes the recall
    @param y_true: groundtruth
    @param y_pred: prediction
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """
    Computes the precision
    @param y_true: groundtruth
    @param y_pred: prediction
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """
    Computes the f1_score
    @param y_true: groundtruth
    @param y_pred: prediction
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def normalize_array(np_array):
    """
    Normalizes numpy array
    @param np_array: array to normalize
    """
    arr_min = np_array.min()
    arr_max = np_array.max()

    return (np_array - arr_min) / (arr_max - arr_min)
