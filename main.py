"""
Punto de entrada principal del proyecto.
Ejecuta el pipeline completo: carga de datos -> entrenamiento con
cross-validation -> modelo final -> evaluación clínica.
"""

import sys
import os
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from entrenamiento import (
    GestorDatos,
    EstandarizadorZScore,
    PerceptronMultiCapa,
    ValidadorCruzado,
    EvaluadorClinico,
    dividir_desarrollo_test,
)

NOMBRES_CARACTERISTICAS = [
    "mediar", "mediag", "mediab", "hhr", "ent", "b",
    "g1", "g2", "g3", "g4", "g5", "mean_r_g",
]


def main() -> None:
    print("=" * 60)
    print(" SISTEMA DE ESTIMACIÓN DE HEMOGLOBINA Y DETECCIÓN DE ANEMIA")
    print(" Red Neuronal Artificial (MLP) desde cero")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. CARGA Y UNIÓN DE DATOS
    # ------------------------------------------------------------------
    base_dir = os.path.abspath(os.path.dirname(__file__))
    ruta_csv_features = os.path.join(base_dir, "resultados_fase1", "caracteristicas_extraidas.csv")
    ruta_csv_hb = os.path.join(base_dir, "dataset", "CP-Anemic", "Datos_Resumido_Dataset.csv")
    ruta_csv_unido = os.path.join(base_dir, "data", "dataset_unido.csv")

    os.makedirs(os.path.dirname(ruta_csv_unido), exist_ok=True)

    gestor = GestorDatos(
        columnas_features=NOMBRES_CARACTERISTICAS,
        columna_hb="HB_LEVEL",
    )

    print("\n[1/5] Cargando y uniendo CSVs (características + hemoglobina)...")
    X, y, ids = gestor.unir_y_guardar(
        ruta_csv_features=ruta_csv_features,
        ruta_csv_hb=ruta_csv_hb,
        ruta_csv_salida=ruta_csv_unido,
        sep_features=",", decimal_features=".",
        sep_hb=";", decimal_hb=",",
    )
    print(f"       CSV unido guardado en: {ruta_csv_unido}")
    print(f"       Muestras: {X.shape[0]} | Características: {X.shape[1]}")

    # ------------------------------------------------------------------
    # 2. DIVISIÓN 80% DESARROLLO / 20% TEST
    # ------------------------------------------------------------------
    print("\n[2/5] Dividiendo datos: 80% desarrollo / 20% test final...")
    X_dev, y_dev, X_test, y_test = dividir_desarrollo_test(
        X, y, proporcion_test=0.2, semilla=42
    )
    print(f"       Desarrollo: {X_dev.shape[0]} muestras")
    print(f"       Test final (aislado): {X_test.shape[0]} muestras")

    # ------------------------------------------------------------------
    # 3. VALIDACIÓN CRUZADA (5-FOLD)
    # ------------------------------------------------------------------
    print("\n[3/5] Ejecutando validación cruzada (5-Fold)...")
    print("-" * 60)

    validador = ValidadorCruzado(k_pliegues=5, semilla=42)
    mse_pliegues = validador.ejecutar(
        X_dev, y_dev,
        n_entradas=12, n_ocultas=32,
        epocas=500, tasa_aprendizaje=0.0005, tamano_lote=16,
        semilla_modelo=42, verbose=True,
    )

    print(f"\n       MSE por pliegue: {[round(m, 4) for m in mse_pliegues]}")
    print(f"       MSE promedio (5-Fold CV): {np.mean(mse_pliegues):.4f} "
          f"(+/- {np.std(mse_pliegues):.4f})")

    # ------------------------------------------------------------------
    # 4. ENTRENAMIENTO DEL MODELO FINAL (100% desarrollo)
    # ------------------------------------------------------------------
    print("\n[4/5] Entrenando modelo final con 100% del conjunto de desarrollo...")
    print("-" * 60)

    escalador_global = EstandarizadorZScore()
    X_dev_std = escalador_global.ajustar_transformar(X_dev)
    X_test_std = escalador_global.transformar(X_test)

    modelo_final = PerceptronMultiCapa(n_entradas=12, n_ocultas=32, semilla=42)
    modelo_final.entrenar(
        X_dev_std, y_dev,
        epocas=1000, tasa_aprendizaje=0.0005, tamano_lote=16,
        semilla=42, verbose=True, imprimir_cada=100,
    )

    # ------------------------------------------------------------------
    # 5. EVALUACIÓN FINAL SOBRE TEST
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluando sobre el conjunto de test (nunca visto)...")
    print("-" * 60)

    y_pred_test = modelo_final.predecir(X_test_std)

    mse_test = float(np.mean((y_pred_test - y_test) ** 2))
    print(f"\n       MSE sobre Test Final (regresión de Hb): {mse_test:.4f}")

    evaluador = EvaluadorClinico(umbral_hb=11.0)
    evaluador.reporte(y_test, y_pred_test)

    # ------------------------------------------------------------------
    # 6. GUARDAR MODELO Y PARÁMETROS DE ESTANDARIZACIÓN
    # ------------------------------------------------------------------
    ruta_modelo = os.path.join(base_dir, "modelos", "modelo_final.npz")
    os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)

    np.savez(
        ruta_modelo,
        # Pesos y sesgos de la red
        W1=modelo_final.W1,
        b1=modelo_final.b1,
        W2=modelo_final.W2,
        b2=modelo_final.b2,
        # Parámetros del estandarizador (necesarios para normalizar datos nuevos)
        mu=escalador_global.mu,
        sigma=escalador_global.sigma,
    )
    print(f"\n[✓] Modelo y parámetros guardados en: {ruta_modelo}")

    print("\n" + "=" * 60)
    print(" Pipeline finalizado.")
    print("=" * 60)


def cargar_modelo(ruta_modelo: str = None):
    """
    Carga el modelo entrenado y el estandarizador desde el archivo .npz.

    Returns:
        modelo: PerceptronMultiCapa con los pesos cargados
        escalador: EstandarizadorZScore con mu/sigma cargados
    """
    if ruta_modelo is None:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        ruta_modelo = os.path.join(base_dir, "modelos", "modelo_final.npz")

    datos = np.load(ruta_modelo)

    modelo = PerceptronMultiCapa(n_entradas=12, n_ocultas=32)
    modelo.W1 = datos["W1"]
    modelo.b1 = datos["b1"]
    modelo.W2 = datos["W2"]
    modelo.b2 = datos["b2"]

    escalador = EstandarizadorZScore()
    escalador.mu = datos["mu"]
    escalador.sigma = datos["sigma"]

    return modelo, escalador


if __name__ == "__main__":
    main()
