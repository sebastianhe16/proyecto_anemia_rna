proyecto_anemia_rna/
│
├── data/
│   ├── raw_images/              # Tus imágenes originales (subcarpetas: anemico/ no_anemico/)
│   │   ├── anemico/
│   │   └── no_anemico/
│   ├── dataset.csv              # Generado automáticamente: features numéricas + etiqueta
│   └── README_datos.md          # Documentas de dónde salieron las imágenes (dataset, fuente, etc.)
│
├── src/
│   ├── __init__.py
│   ├── extraccion_caracteristicas.py   # Único archivo que usa PIL/numpy (lee imagen -> RGB/HSV/edad/sexo/Hb)
│   ├── preprocesamiento.py             # Normalización Min-Max, train/val/test split, balanceo (manual)
│   ├── red_neuronal.py                 # LA RED NEURONAL DESDE CERO (forward, backprop, Adam) — solo math
│   ├── metricas.py                     # Accuracy, Precision, Recall, F1, matriz de confusión (manual)
│   ├── entrenamiento.py                # Orquesta: carga dataset.csv, entrena, guarda pesos
│   └── utilidades.py                   # Funciones auxiliares (guardar/cargar pesos en JSON, logging, etc.)
│
├── modelos/
│   └── pesos_entrenados.json    # Pesos y bias guardados tras entrenar (para no reentrenar cada vez)
│
├── resultados/
│   ├── historial_entrenamiento.csv   # Loss/accuracy por época
│   └── reporte_metricas.txt
│
├── main.py                      # Punto de entrada: menú o pipeline completo (preparar datos -> entrenar -> evaluar)
├── requirements.txt             # numpy, Pillow, matplotlib (opcional, solo para gráficos)
└── README.md                    # Explicación general del proyecto para el informe