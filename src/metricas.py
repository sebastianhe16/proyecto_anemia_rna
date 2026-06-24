"""
Cálculo de métricas de evaluación.
Incluye Accuracy, Precision, Recall, F1 y matriz de confusión.
"""

import numpy as np
from typing import Tuple


def matriz_confusion(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de confusión para clasificación binaria.
    
    Args:
        y_real (np.ndarray): Etiquetas verdaderas
        y_pred (np.ndarray): Etiquetas predichas
        
    Returns:
        np.ndarray: Matriz de confusión (2x2)
    """
    TP = np.sum((y_real == 1) & (y_pred == 1))
    TN = np.sum((y_real == 0) & (y_pred == 0))
    FP = np.sum((y_real == 0) & (y_pred == 1))
    FN = np.sum((y_real == 1) & (y_pred == 0))
    
    matriz = np.array([[TN, FP],
                       [FN, TP]])
    return matriz


def accuracy(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula la exactitud (accuracy).
    
    Args:
        y_real (np.ndarray): Etiquetas verdaderas
        y_pred (np.ndarray): Etiquetas predichas
        
    Returns:
        float: Accuracy entre 0 y 1
    """
    return np.mean(y_real == y_pred)


def precision(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula la precisión.
    Precision = TP / (TP + FP)
    
    Args:
        y_real (np.ndarray): Etiquetas verdaderas
        y_pred (np.ndarray): Etiquetas predichas
        
    Returns:
        float: Precisión entre 0 y 1
    """
    TP = np.sum((y_real == 1) & (y_pred == 1))
    FP = np.sum((y_real == 0) & (y_pred == 1))
    
    if TP + FP == 0:
        return 0.0
    
    return TP / (TP + FP)


def recall(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el recall (sensibilidad/tasa de verdaderos positivos).
    Recall = TP / (TP + FN)
    
    Args:
        y_real (np.ndarray): Etiquetas verdaderas
        y_pred (np.ndarray): Etiquetas predichas
        
    Returns:
        float: Recall entre 0 y 1
    """
    TP = np.sum((y_real == 1) & (y_pred == 1))
    FN = np.sum((y_real == 1) & (y_pred == 0))
    
    if TP + FN == 0:
        return 0.0
    
    return TP / (TP + FN)


def f1_score(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el F1-score (media armónica de precisión y recall).
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        y_real (np.ndarray): Etiquetas verdaderas
        y_pred (np.ndarray): Etiquetas predichas
        
    Returns:
        float: F1-score entre 0 y 1
    """
    prec = precision(y_real, y_pred)
    rec = recall(y_real, y_pred)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


def perdida_entropia_cruzada(y_real: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calcula la pérdida de entropía cruzada binaria.
    Loss = -mean(y * log(p) + (1 - y) * log(1 - p))
    
    Args:
        y_real (np.ndarray): Etiquetas verdaderas (0 o 1)
        y_pred_proba (np.ndarray): Probabilidades predichas (entre 0 y 1)
        
    Returns:
        float: Pérdida
    """
    # Clip para evitar log(0)
    y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
    
    loss = -np.mean(y_real * np.log(y_pred_proba) + 
                    (1 - y_real) * np.log(1 - y_pred_proba))
    return loss


def evaluar_modelo(y_real: np.ndarray, y_pred: np.ndarray, 
                   y_pred_proba: np.ndarray = None) -> dict:
    """
    Calcula todas las métricas de evaluación.
    
    Args:
        y_real (np.ndarray): Etiquetas verdaderas
        y_pred (np.ndarray): Etiquetas predichas (0 o 1)
        y_pred_proba (np.ndarray, optional): Probabilidades predichas
        
    Returns:
        dict: Diccionario con todas las métricas
    """
    metricas = {
        'accuracy': accuracy(y_real, y_pred),
        'precision': precision(y_real, y_pred),
        'recall': recall(y_real, y_pred),
        'f1': f1_score(y_real, y_pred),
        'matriz_confusion': matriz_confusion(y_real, y_pred)
    }
    
    if y_pred_proba is not None:
        metricas['perdida'] = perdida_entropia_cruzada(y_real, y_pred_proba)
    
    return metricas
