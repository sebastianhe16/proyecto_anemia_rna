import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from entrenamiento import preparar_dataset_real


def main() -> None:
    print("=" * 60)
    print("PIPELINE REAL DE PREPARACIÓN DEL DATASET")
    print("=" * 60)

    ruta_csv_features = "resultados_fase1/caracteristicas_extraidas.csv"
    ruta_csv_hb = "dataset/CP-Anemic/Datos_Resumido_Dataset.csv"
    ruta_salida_csv = "data/dataset_real_normalizado.csv"
    ruta_parametros = "data/parametros_normalizacion.json"

    preparar_dataset_real(
        ruta_csv_features=ruta_csv_features,
        ruta_csv_hb=ruta_csv_hb,
        ruta_salida_csv=ruta_salida_csv,
        ruta_parametros=ruta_parametros,
    )

    print("\nListo. El archivo preparado está en:")
    print(f"- {ruta_salida_csv}")
    print(f"- {ruta_parametros}")


if __name__ == "__main__":
    main()
