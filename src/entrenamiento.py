import numpy as np

from red_neuronal import RedNeuronal
from metricas import evaluar_modelo
from utilidades import guardar_pesos, logging


class Entrenador:
    def __init__(self, arquitectura: list, lr=0.001, epochs=100, batch_size=32):
        self.red = RedNeuronal(arquitectura=arquitectura, tasa_aprendizaje=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.historial = {
            "perdida_entrenamiento": [],
            "perdida_validacion": [],
            "accuracy_entrenamiento": [],
            "accuracy_validacion": [],
        }

    def entrenar_epoch(self, X_train: np.ndarray, y_train: np.ndarray):
        n_muestras = len(X_train)
        n_batches = (n_muestras + self.batch_size - 1) // self.batch_size
        indices = np.random.permutation(n_muestras)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for i in range(n_batches):
            inicio = i * self.batch_size
            fin = min(inicio + self.batch_size, n_muestras)
            X_batch = X_shuffled[inicio:fin]
            y_batch = y_shuffled[inicio:fin]
            for muestra, etiqueta in zip(X_batch, y_batch):
                self.red.entrenar_un_ejemplo(list(muestra), float(etiqueta))

        y_proba_train = np.array(self.red.forward(X_train), dtype=float).reshape(-1)
        metricas_train = evaluar_modelo(y_train, (y_proba_train > 0.5).astype(int), y_proba_train)
        return metricas_train["loss_promedio"], metricas_train["accuracy"]

    def evaluar(self, X_val: np.ndarray, y_val: np.ndarray):
        y_proba_val = np.array(self.red.forward(X_val), dtype=float).reshape(-1)
        y_pred_val = (y_proba_val > 0.5).astype(int)
        metricas_val = evaluar_modelo(y_val, y_pred_val, y_proba_val)
        return metricas_val["loss_promedio"], metricas_val["accuracy"]

    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray, verbose=True) -> dict:
        print(f"Comenzando el entrenamiento del perceptrón multicapa por {self.epochs} épocas...")
        for epoch in range(self.epochs):
            loss_train, acc_train = self.entrenar_epoch(X_train, y_train)
            loss_val, acc_val = self.evaluar(X_val, y_val)
            self.historial["perdida_entrenamiento"].append(loss_train)
            self.historial["perdida_validacion"].append(loss_val)
            self.historial["accuracy_entrenamiento"].append(acc_train)
            self.historial["accuracy_validacion"].append(acc_val)
            if verbose and (epoch + 1) % 10 == 0:
                logging(
                    f"Época {epoch + 1}/{self.epochs} -> "
                    f"Loss Train: {loss_train:.4f} | Loss Val: {loss_val:.4f} || "
                    f"Acc Train: {acc_train * 100:.2f}% | Acc Val: {acc_val * 100:.2f}%"
                )
        return self.historial

    def guardar_modelo(self, ruta: str):
        parametros = self.red.obtener_parametros()
        guardar_pesos(parametros, ruta)

    def cargar_modelo(self, ruta: str):
        from utilidades import cargar_pesos as leer_json
        parametros = leer_json(ruta)
        self.red.establecer_parametros(parametros)
