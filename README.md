# Proyecto: Detección de anemia mediante redes neuronales artificiales

Este proyecto implementa una red neuronal artificial desde cero para clasificar anemia a partir de características extraídas de imágenes de conjuntiva palpebral. El sistema usa Python y sigue una estructura simple para exposición y desarrollo académico.

## Enfoque del proyecto
- Se usan imágenes del dataset Eyes-Defy-Anemia.
- Se extraen características de color desde la zona útil de la imagen.
- Se aplica un preprocesamiento básico de imagen para resaltar la región de interés y mejorar la estabilidad del análisis.
- Se entrena una red neuronal MLP desde cero, sin usar TensorFlow, Keras ni scikit-learn.

## Archivos principales
- [main.py](main.py): punto de entrada del sistema.
- [src/extraccion_caracteristicas.py](src/extraccion_caracteristicas.py): extrae características desde las imágenes y genera el dataset.
- [src/preprocesamiento.py](src/preprocesamiento.py): normaliza los datos y genera los conjuntos train/val/test.
- [src/red_neuronal.py](src/red_neuronal.py): implementación manual de la red neuronal.
- [src/entrenamiento.py](src/entrenamiento.py): orquesta el entrenamiento y guarda los pesos.
- [src/metricas.py](src/metricas.py): calcula accuracy, precision, recall y F1.

## Ejecución rápida
1. Instalar dependencias:
   - pip install -r requirements.txt
2. Ejecutar el sistema:
   - python main.py
3. Elegir la opción 4 para ejecutar todo el pipeline.

## Resultado esperado
El proyecto genera:
- [data/dataset.csv](data/dataset.csv)
- [data/dataset_train.csv](data/dataset_train.csv)
- [data/dataset_val.csv](data/dataset_val.csv)
- [data/dataset_test.csv](data/dataset_test.csv)
- [modelos/pesos_entrenados.json](modelos/pesos_entrenados.json)
- [resultados/historial_entrenamiento.csv](resultados/historial_entrenamiento.csv)
- [resultados/reporte_metricas.txt](resultados/reporte_metricas.txt)
