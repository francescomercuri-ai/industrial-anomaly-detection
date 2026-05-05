import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_image_level_auroc(y_true, anomaly_scores):
    """
    y_true: lista di 0 (OK) e 1 (Difettoso)
    anomaly_scores: lista del punteggio massimo di anomalia per ogni immagine
    """
    auroc = roc_auc_score(y_true, anomaly_scores)
    return auroc