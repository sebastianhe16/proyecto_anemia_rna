

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd  



class GestorDatos:
    

    def __init__(self,
                 columnas_features: List[str],
                 columna_id: str = "IMAGE_ID",
                 columna_hb: str = "HB_Level") -> None:
        self.columnas_features = columnas_features
        self.columna_id = columna_id
        self.columna_hb = columna_hb

    # --------------------------------------------------------------------
    @staticmethod
    def _clave_orden_natural(id_imagen: str) -> Tuple[str, int]:
        
        coincidencia = re.search(r"(\d+)", str(id_imagen))
        if coincidencia:
            numero = int(coincidencia.group(1))
            prefijo = str(id_imagen)[:coincidencia.start()]
        else:
            numero = 0
            prefijo = str(id_imagen)
        return (prefijo, numero)

    # --------------------------------------------------------------------
    def cargar_y_unir(self, ruta_csv_features: str,
                       ruta_csv_hb: str,
                       sep_features: str = ',', decimal_features: str = '.',
                       sep_hb: str = ',', decimal_hb: str = '.') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        
        df_features = pd.read_csv(
            ruta_csv_features,
            sep=sep_features,
            decimal=decimal_features,
            encoding='utf-8'
        )
        df_hb = pd.read_csv(
            ruta_csv_hb,
            sep=sep_hb,
            decimal=decimal_hb,
            encoding='utf-8'
        )

        df_unido = pd.merge(
            df_features, df_hb, on=self.columna_id, how="inner"
        )

        if df_unido.empty:
            raise ValueError(
                "La unión de los dos CSV produjo un DataFrame vacío. "
                "Verifique que la columna de unión coincida en ambos archivos "
                f"(columna usada: '{self.columna_id}')."
            )

        
        df_unido = df_unido.assign(
            _clave_orden=df_unido[self.columna_id].map(self._clave_orden_natural)
        ).sort_values("_clave_orden").drop(columns="_clave_orden").reset_index(drop=True)

        if self.columna_hb not in df_unido.columns:
            columnas_hb_alternativas = [c for c in df_unido.columns if c.upper() == self.columna_hb.upper()]
            if columnas_hb_alternativas:
                self.columna_hb = columnas_hb_alternativas[0]

        columnas_faltantes = [c for c in self.columnas_features if c not in df_unido.columns]
        if columnas_faltantes:
            raise ValueError(
                f"Faltan columnas de características en el CSV: {columnas_faltantes}"
            )

        if self.columna_hb not in df_unido.columns:
            raise ValueError(
                f"La columna de hemoglobina '{self.columna_hb}' no está presente en el CSV unido."
            )

        X = df_unido[self.columnas_features].to_numpy(dtype=np.float64)
        y = df_unido[self.columna_hb].to_numpy(dtype=np.float64).reshape(-1, 1)
        ids = df_unido[self.columna_id].astype(str).tolist()

        return X, y, ids

    def unir_y_guardar(self, ruta_csv_features: str, ruta_csv_hb: str,
                       ruta_csv_salida: str,
                       sep_features: str = ',', decimal_features: str = '.',
                       sep_hb: str = ';', decimal_hb: str = ',') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        

        df_features = pd.read_csv(
            ruta_csv_features,
            sep=sep_features,
            decimal=decimal_features,
            encoding='utf-8'
        )
        df_hb = pd.read_csv(
            ruta_csv_hb,
            sep=sep_hb,
            decimal=decimal_hb,
            encoding='utf-8'
        )

        df_unido = pd.merge(
            df_features, df_hb, on=self.columna_id, how='inner'
        )

        if df_unido.empty:
            raise ValueError(
                "La unión de los dos CSV produjo un DataFrame vacío. "
                "Verifique que la columna de unión coincida en ambos archivos "
                f"(columna usada: '{self.columna_id}')."
            )

        df_unido = df_unido.assign(
            _clave_orden=df_unido[self.columna_id].map(self._clave_orden_natural)
        ).sort_values("_clave_orden").drop(columns="_clave_orden").reset_index(drop=True)

        df_unido.to_csv(ruta_csv_salida, index=False)

        return self.cargar_desde_csv_unido(ruta_csv_salida)

    def cargar_desde_csv_unido(self, ruta_csv_unido: str,
                               sep: str = ',', decimal: str = '.') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        

        df_unido = pd.read_csv(ruta_csv_unido, sep=sep, decimal=decimal, encoding='utf-8')

        if self.columna_hb not in df_unido.columns:
            columnas_hb_alternativas = [c for c in df_unido.columns if c.upper() == self.columna_hb.upper()]
            if columnas_hb_alternativas:
                self.columna_hb = columnas_hb_alternativas[0]

        if self.columna_hb not in df_unido.columns:
            raise ValueError(
                f"La columna de hemoglobina '{self.columna_hb}' no está presente en el CSV unido."
            )

        columnas_faltantes = [c for c in self.columnas_features if c not in df_unido.columns]
        if columnas_faltantes:
            raise ValueError(
                f"Faltan columnas de características en el CSV unido: {columnas_faltantes}"
            )

        df_unido = df_unido.assign(
            _clave_orden=df_unido[self.columna_id].map(self._clave_orden_natural)
        ).sort_values("_clave_orden").drop(columns="_clave_orden").reset_index(drop=True)

        X = df_unido[self.columnas_features].to_numpy(dtype=np.float64)
        y = df_unido[self.columna_hb].to_numpy(dtype=np.float64).reshape(-1, 1)
        ids = df_unido[self.columna_id].astype(str).tolist()

        return X, y, ids




class EstandarizadorZScore:
    

    def __init__(self) -> None:
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None

    def ajustar(self, X: np.ndarray) -> None:
        
        self.mu = X.mean(axis=0, keepdims=True)
        self.sigma = X.std(axis=0, keepdims=True)

        
        self.sigma[self.sigma == 0.0] = 1e-8

    def transformar(self, X: np.ndarray) -> np.ndarray:
        
        if self.mu is None or self.sigma is None:
            raise RuntimeError(
                "Debe llamar a `ajustar` antes de `transformar` "
                "(no hay mu/sigma calculados)."
            )
        return (X - self.mu) / self.sigma

    def ajustar_transformar(self, X: np.ndarray) -> np.ndarray:
        
        self.ajustar(X)
        return self.transformar(X)




class PerceptronMultiCapa:
    

    def __init__(self, n_entradas: int = 12, n_ocultas: int = 16,
                 semilla: int = 42) -> None:
        rng = np.random.default_rng(semilla)

        
        self.W1 = rng.standard_normal((n_entradas, n_ocultas)) * np.sqrt(2.0 / n_entradas)
        self.b1 = np.zeros((1, n_ocultas))


        self.W2 = rng.standard_normal((n_ocultas, 1)) * np.sqrt(1.0 / n_ocultas)
        self.b2 = np.zeros((1, 1))

       
        self.m_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
 
        self.t = 0

       
        self.historial_perdida_train: List[float] = []
        self.historial_perdida_val: List[float] = []


    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
       
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_derivada(z: np.ndarray) -> np.ndarray:
        
        return (z > 0.0).astype(np.float64)


    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        
        Z1 = X @ self.W1 + self.b1
        A1 = self._relu(Z1)
        Z2 = A1 @ self.W2 + self.b2  

        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2}
        return Z2, cache


    @staticmethod
    def _mse(y_pred: np.ndarray, y_real: np.ndarray) -> float:
        
        return float(np.mean((y_pred - y_real) ** 2))

   
    def backward(self, cache: Dict[str, np.ndarray], y_real: np.ndarray,
                 tasa_aprendizaje: float, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 lambda_l2: float = 0.0) -> None:
        

        X, Z1, A1, Z2 = cache["X"], cache["Z1"], cache["A1"], cache["Z2"]
        n_muestras = X.shape[0]

       
        dZ2 = (2.0 / n_muestras) * (Z2 - y_real)
        dW2 = A1.T @ dZ2 + lambda_l2 * self.W2  # L2 sobre W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self._relu_derivada(Z1)
        dW1 = X.T @ dZ1 + lambda_l2 * self.W1   # L2 sobre W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

      
        self.t += 1

       
        self.m_W1 = beta1 * self.m_W1 + (1 - beta1) * dW1
        self.v_W1 = beta2 * self.v_W1 + (1 - beta2) * (dW1 ** 2)
        m_hat = self.m_W1 / (1 - beta1 ** self.t)
        v_hat = self.v_W1 / (1 - beta2 ** self.t)
        self.W1 -= tasa_aprendizaje * m_hat / (np.sqrt(v_hat) + epsilon)

        
        self.m_b1 = beta1 * self.m_b1 + (1 - beta1) * db1
        self.v_b1 = beta2 * self.v_b1 + (1 - beta2) * (db1 ** 2)
        m_hat = self.m_b1 / (1 - beta1 ** self.t)
        v_hat = self.v_b1 / (1 - beta2 ** self.t)
        self.b1 -= tasa_aprendizaje * m_hat / (np.sqrt(v_hat) + epsilon)

        
        self.m_W2 = beta1 * self.m_W2 + (1 - beta1) * dW2
        self.v_W2 = beta2 * self.v_W2 + (1 - beta2) * (dW2 ** 2)
        m_hat = self.m_W2 / (1 - beta1 ** self.t)
        v_hat = self.v_W2 / (1 - beta2 ** self.t)
        self.W2 -= tasa_aprendizaje * m_hat / (np.sqrt(v_hat) + epsilon)

        
        self.m_b2 = beta1 * self.m_b2 + (1 - beta1) * db2
        self.v_b2 = beta2 * self.v_b2 + (1 - beta2) * (db2 ** 2)
        m_hat = self.m_b2 / (1 - beta1 ** self.t)
        v_hat = self.v_b2 / (1 - beta2 ** self.t)
        self.b2 -= tasa_aprendizaje * m_hat / (np.sqrt(v_hat) + epsilon)

    
    def entrenar(self,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None,
                 epocas: int = 200,
                 tasa_aprendizaje: float = 0.001,
                 tamano_lote: int = 16,
                 semilla: int = 42,
                 verbose: bool = False,
                 imprimir_cada: int = 20) -> None:
        
        rng = np.random.default_rng(semilla)
        n_muestras = X_train.shape[0]

        for epoca in range(1, epocas + 1):
           
            indices_barajados = rng.permutation(n_muestras)
            X_barajado = X_train[indices_barajados]
            y_barajado = y_train[indices_barajados]

            
            for inicio in range(0, n_muestras, tamano_lote):
                fin = inicio + tamano_lote
                X_lote = X_barajado[inicio:fin]
                y_lote = y_barajado[inicio:fin]

                _, cache = self.forward(X_lote)
                self.backward(cache, y_lote, tasa_aprendizaje)

           
            y_pred_train, _ = self.forward(X_train)
            perdida_train = self._mse(y_pred_train, y_train)
            self.historial_perdida_train.append(perdida_train)

            perdida_val = None
            if X_val is not None and y_val is not None:
                y_pred_val, _ = self.forward(X_val)
                perdida_val = self._mse(y_pred_val, y_val)
                self.historial_perdida_val.append(perdida_val)

            if verbose and (epoca == 1 or epoca % imprimir_cada == 0 or epoca == epocas):
                mensaje = f"    Época {epoca:4d}/{epocas} | MSE train: {perdida_train:8.4f}"
                if perdida_val is not None:
                    mensaje += f" | MSE val: {perdida_val:8.4f}"
                print(mensaje)

    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        
        y_pred, _ = self.forward(X)
        return y_pred




class ValidadorCruzado:
    

    def __init__(self, k_pliegues: int = 5, semilla: int = 42) -> None:
        self.k_pliegues = k_pliegues
        self.semilla = semilla

    # --------------------------------------------------------------------
    def generar_indices_pliegues(self, n_muestras: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        
        rng = np.random.default_rng(self.semilla)
        indices_permutados = rng.permutation(n_muestras)
        bloques = np.array_split(indices_permutados, self.k_pliegues)

        combinaciones: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(self.k_pliegues):
            idx_val = bloques[i]
            idx_train = np.concatenate([bloques[j] for j in range(self.k_pliegues) if j != i])
            combinaciones.append((idx_train, idx_val))

        return combinaciones

    # --------------------------------------------------------------------
    def ejecutar(self,
                 X_dev: np.ndarray, y_dev: np.ndarray,
                 n_entradas: int = 12, n_ocultas: int = 16,
                 epocas: int = 150, tasa_aprendizaje: float = 0.001,
                 tamano_lote: int = 16, semilla_modelo: int = 42,
                 verbose: bool = True) -> List[float]:
        
        combinaciones = self.generar_indices_pliegues(X_dev.shape[0])
        mse_por_pliegue: List[float] = []

        for numero_pliegue, (idx_train, idx_val) in enumerate(combinaciones, start=1):
            X_train_local = X_dev[idx_train]
            y_train_local = y_dev[idx_train]
            X_val_local = X_dev[idx_val]
            y_val_local = y_dev[idx_val]

            
            escalador_local = EstandarizadorZScore()
            X_train_local_std = escalador_local.ajustar_transformar(X_train_local)
            X_val_local_std = escalador_local.transformar(X_val_local)

          
            modelo = PerceptronMultiCapa(
                n_entradas=n_entradas, n_ocultas=n_ocultas, semilla=semilla_modelo
            )

            if verbose:
                print(f"\n--- Pliegue {numero_pliegue}/{self.k_pliegues} "
                      f"(train={len(idx_train)}, val={len(idx_val)}) ---")

            modelo.entrenar(
                X_train_local_std, y_train_local,
                X_val_local_std, y_val_local,
                epocas=epocas, tasa_aprendizaje=tasa_aprendizaje,
                tamano_lote=tamano_lote, verbose=verbose,
            )

            y_pred_val_local = modelo.predecir(X_val_local_std)
            mse_val = float(np.mean((y_pred_val_local - y_val_local) ** 2))
            mse_por_pliegue.append(mse_val)

            if verbose:
                print(f"  >> MSE final de validación (pliegue {numero_pliegue}): {mse_val:.4f}")

        return mse_por_pliegue



class EvaluadorClinico:
    

    def __init__(self, umbral_hb: float = 11.0) -> None:
        self.umbral_hb = umbral_hb

    # --------------------------------------------------------------------
    def clasificar(self, valores_hb: np.ndarray) -> np.ndarray:
        
        return (valores_hb < self.umbral_hb).astype(int)

    # --------------------------------------------------------------------
    @staticmethod
    def matriz_confusion(y_real_bin: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, int]:
        
        y_real_bin = y_real_bin.flatten()
        y_pred_bin = y_pred_bin.flatten()

        vp = int(np.sum((y_pred_bin == 1) & (y_real_bin == 1)))
        fp = int(np.sum((y_pred_bin == 1) & (y_real_bin == 0)))
        fn = int(np.sum((y_pred_bin == 0) & (y_real_bin == 1)))
        vn = int(np.sum((y_pred_bin == 0) & (y_real_bin == 0)))

        return {"VP": vp, "FP": fp, "FN": fn, "VN": vn}

    # --------------------------------------------------------------------
    @staticmethod
    def calcular_metricas(matriz: Dict[str, int]) -> Dict[str, float]:
        
        vp, fp, fn, vn = matriz["VP"], matriz["FP"], matriz["FN"], matriz["VN"]
        total = vp + fp + fn + vn

        exactitud = (vp + vn) / total if total > 0 else 0.0
        sensibilidad = vp / (vp + fn) if (vp + fn) > 0 else 0.0
        especificidad = vn / (vn + fp) if (vn + fp) > 0 else 0.0
        precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
        f1 = (2 * precision * sensibilidad / (precision + sensibilidad)
              if (precision + sensibilidad) > 0 else 0.0)

        return {
            "exactitud": exactitud,
            "sensibilidad": sensibilidad,
            "especificidad": especificidad,
            "precision": precision,
            "f1": f1,
        }

    # --------------------------------------------------------------------
    def reporte(self, y_real_continuo: np.ndarray,
                y_pred_continuo: np.ndarray) -> Tuple[Dict[str, int], Dict[str, float]]:
        
        y_real_bin = self.clasificar(y_real_continuo)
        y_pred_bin = self.clasificar(y_pred_continuo)

        matriz = self.matriz_confusion(y_real_bin, y_pred_bin)
        metricas = self.calcular_metricas(matriz)

        print("\n" + "=" * 60)
        print("MATRIZ DE CONFUSIÓN (Test Final - 20% aislado)")
        print("=" * 60)
        print(f"  Verdaderos Positivos (VP) - Anemia detectada correctamente : {matriz['VP']}")
        print(f"  Falsos Positivos     (FP) - Predijo Anemia, paciente sano  : {matriz['FP']}")
        print(f"  Falsos Negativos     (FN) - Anemia NO detectada (riesgo!!) : {matriz['FN']}")
        print(f"  Verdaderos Negativos (VN) - Sano detectado correctamente   : {matriz['VN']}")

        print("\n" + "=" * 60)
        print("MÉTRICAS DE RENDIMIENTO CLÍNICO")
        print("=" * 60)
        print(f"  Exactitud (Accuracy)          : {metricas['exactitud'] * 100:6.2f}%")
        print(f"  >>> SENSIBILIDAD (Recall)     : {metricas['sensibilidad'] * 100:6.2f}%  "
              "<-- métrica prioritaria en tamizaje médico")
        print(f"  Especificidad                  : {metricas['especificidad'] * 100:6.2f}%")
        print(f"  Precisión                      : {metricas['precision'] * 100:6.2f}%")
        print(f"  F1-Score                        : {metricas['f1'] * 100:6.2f}%")
        print("=" * 60)

        return matriz, metricas




def dividir_desarrollo_test(X: np.ndarray, y: np.ndarray,
                             proporcion_test: float = 0.2,
                             semilla: int = 42
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    rng = np.random.default_rng(semilla)
    n_muestras = X.shape[0]
    indices_permutados = rng.permutation(n_muestras)

    n_test = int(round(n_muestras * proporcion_test))
    idx_test = indices_permutados[:n_test]
    idx_dev = indices_permutados[n_test:]

    X_dev, y_dev = X[idx_dev], y[idx_dev]
    X_test, y_test = X[idx_test], y[idx_test]

    return X_dev, y_dev, X_test, y_test




if __name__ == "__main__":

    NOMBRES_CARACTERISTICAS = [
        "mediar", "mediag", "mediab", "hhr", "ent", "b",
        "g1", "g2", "g3", "g4", "g5", "mean_r_g",
    ]

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ruta_csv_features = os.path.join(base_dir, "resultados_fase1", "caracteristicas_extraidas.csv")
    ruta_csv_hb = os.path.join(base_dir, "dataset", "CP-Anemic", "Datos_Resumido_Dataset.csv")
    ruta_csv_unido = os.path.join(base_dir, "data", "dataset_unido.csv")

    os.makedirs(os.path.dirname(ruta_csv_unido), exist_ok=True)

    gestor = GestorDatos(
        columnas_features=NOMBRES_CARACTERISTICAS,
        columna_hb="HB_LEVEL"
    )

    print("Uniendo los CSV reales de características y Hb...\n")
    X, y, ids = gestor.unir_y_guardar(
        ruta_csv_features=ruta_csv_features,
        ruta_csv_hb=ruta_csv_hb,
        ruta_csv_salida=ruta_csv_unido,
        sep_features=",", decimal_features='.',
        sep_hb=';', decimal_hb=','
    )

    print(f"CSV unido guardado en: {ruta_csv_unido}")
    print(f"Datos unidos y ordenados: {X.shape[0]} muestras, {X.shape[1]} características.")
    print(f"Primeros IDs tras el ordenamiento natural: {ids[:5]}\n")

    X_dev, y_dev, X_test, y_test = dividir_desarrollo_test(
        X, y, proporcion_test=0.2, semilla=42
    )
    print(f"Split inicial -> Desarrollo: {X_dev.shape[0]} muestras | "
          f"Test final (aislado): {X_test.shape[0]} muestras\n")

    print("=" * 60)
    print("VALIDACIÓN CRUZADA (5-FOLD) SOBRE EL CONJUNTO DE DESARROLLO")
    print("=" * 60)

    validador = ValidadorCruzado(k_pliegues=5, semilla=42)
    mse_pliegues = validador.ejecutar(
        X_dev, y_dev,
        n_entradas=12, n_ocultas=16,
        epocas=150, tasa_aprendizaje=0.001, tamano_lote=16,
        semilla_modelo=42, verbose=True,
    )

    print(f"\nMSE por pliegue: {[round(m, 4) for m in mse_pliegues]}")
    print(f"MSE promedio (5-Fold CV): {np.mean(mse_pliegues):.4f} "
          f"(+/- {np.std(mse_pliegues):.4f})")

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO DEL MODELO FINAL (100% del conjunto de desarrollo)")
    print("=" * 60)

    escalador_global = EstandarizadorZScore()
    X_dev_std = escalador_global.ajustar_transformar(X_dev)
    X_test_std = escalador_global.transformar(X_test)

    modelo_final = PerceptronMultiCapa(n_entradas=12, n_ocultas=16, semilla=42)
    modelo_final.entrenar(
        X_dev_std, y_dev,
        epocas=200, tasa_aprendizaje=0.001, tamano_lote=16,
        semilla=42, verbose=True, imprimir_cada=25,
    )

    y_pred_test = modelo_final.predecir(X_test_std)

    mse_test = float(np.mean((y_pred_test - y_test) ** 2))
    print(f"\nMSE sobre el Test Final (regresión continua de Hb): {mse_test:.4f}")

    evaluador = EvaluadorClinico(umbral_hb=11.0)
    matriz_final, metricas_finales = evaluador.reporte(y_test, y_pred_test)