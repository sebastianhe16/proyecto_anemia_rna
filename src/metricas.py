"""
metricas.py
-------------------------------------------------------------------------------
PROPÓSITO:
    Este archivo implementa desde cero y de forma nativa las funciones
    estadísticas y métricas de evaluación requeridas por la rúbrica del
    proyecto para medir el rendimiento del modelo clasificador de anemia.

    NO utiliza scikit-learn ni ninguna otra librería externa. Todos los
    cálculos se realizan usando tipos de datos primitivos de Python (listas,
    diccionarios, bucles for) para evidenciar la capacidad analítica y de 
    abstracción en ingeniería de sistemas.
"""

import math

def calcular_matriz_confusion(reales, predichos):
    """
    Calcula manualmente los componentes de una matriz de confusión binaria.
    Entradas:
        reales: Lista de etiquetas reales (0 o 1).
        predichos: Lista de etiquetas predichas por el modelo (0 o 1).
    Devuelve:
        Un diccionario con Verdaderos Positivos (TP), Falsos Positivos (FP),
        Verdaderos Negativos (TN) y Falsos Negativos (FN).
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for r, p in zip(reales, predichos):
        if r == 1 and p == 1:
            tp += 1
        elif r == 0 and p == 1:
            fp += 1
        elif r == 0 and p == 0:
            tn += 1
        elif r == 1 and p == 0:
            fn += 1
            
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def evaluar_modelo(reales, predichos, probabilidades=None):
    """
    Calcula todas las métricas solicitadas en la rúbrica a partir de las 
    etiquetas reales y las predicciones del modelo.
    
    Fórmulas aplicadas:
        - Exactitud (Accuracy) = (TP + TN) / Total
        - Precisión (Precision) = TP / (TP + FP)
        - Sensibilidad (Recall) = TP / (TP + FN)
        - F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    """
    matriz = calcular_matriz_confusion(reales, predichos)
    tp = matriz["TP"]
    fp = matriz["FP"]
    tn = matriz["TN"]
    fn = matriz["FN"]
    
    total = len(reales)
    
    # 1. Exactitud (Accuracy)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # 2. Precisión (Precision)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # 3. Sensibilidad o Exhaustividad (Recall)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 4. F1-Score
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
        
    # 5. Cálculo opcional de la Pérdida de Entropía Cruzada Binaria (Log Loss)
    # Útil para el informe y ver la convergencia matemática fina.
    loss_promedio = 0.0
    if probabilidades is not None:
        suma_loss = 0.0
        eps = 1e-15  # Evita errores de logaritmo de cero log(0)
        for r, prob in zip(reales, probabilidades):
            # Forzar límites para estabilidad numérica
            p_ajustada = max(eps, min(1.0 - eps, prob))
            # Fórmula: -[y*log(p) + (1-y)*log(1-p)]
            suma_loss += -(r * math.log(p_ajustada) + (1.0 - r) * math.log(1.0 - p_ajustada))
        loss_promedio = suma_loss / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "matriz_confusion": matriz,
        "loss_promedio": loss_promedio
    }


def imprimir_reporte_metricas(metricas):
    """
    Imprime en consola un reporte formateado y elegante para las defensas del proyecto.
    """
    matriz = metricas["matriz_confusion"]
    print("\n" + "="*50)
    print("      REPORTE DE EVALUACIÓN DEL SISTEMA INTELIGENTE")
    print("="*50)
    print(f"Pérdida promedio (LogLoss): {metricas['loss_promedio']:.6f}")
    print(f"Exactitud (Accuracy):      {metricas['accuracy']*100:.2f}%")
    print(f"Precisión (Precision):     {metricas['precision']*100:.2f}%")
    print(f"Sensibilidad (Recall):     {metricas['recall']*100:.2f}%")
    print(f"F1-Score:                  {metricas['f1']:.4f}")
    print("-"*50)
    print("Matriz de Confusión:")
    print(f"   [ Predicciones -> ]   |  Sano (0)  | Anémico (1) |")
    print(f" Real: Sano (0)          |    {matriz['TN']:^6}  |    {matriz['FP']:^7}  |")
    print(f" Real: Anémico (1)       |    {matriz['FN']:^6}  |    {matriz['TP']:^7}  |")
    print("="*50)


# Prueba unitaria rápida aislada para verificar el comportamiento
if __name__ == "__main__":
    # Simulación de datos: 5 pacientes reales vs predicción del sistema
    reales_test =     [1, 1, 0, 0, 1]
    predichos_test =  [1, 0, 0, 0, 1]
    probabilidad_test=[0.92, 0.12, 0.05, 0.34, 0.88]
    
    res = evaluar_modelo(reales_test, predichos_test, probabilidad_test)
    imprimir_reporte_metricas(res)