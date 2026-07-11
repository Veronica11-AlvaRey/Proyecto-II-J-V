# Chatbot UG - Admisión y Nivelación

Proyecto de PLN clásico que recupera respuestas sobre oferta académica, admisión y nivelación de la Universidad de Guayaquil.

## Técnicas
- Normalización de texto (minúsculas, sin tildes, sin puntuación)
- Tokenización, eliminación de stopwords (lista propia en español) y stemming
  (NLTK `SnowballStemmer`, sin depender de descargas externas)
- TF-IDF con unigramas y bigramas, sobre el texto ya preprocesado
- Similitud coseno
- Umbral de confianza y fallback (con mensaje explícito de "no encontré respuesta")
- Expresión regular para último dígito de cédula/pasaporte
- Expresión regular para detección de saludo y despedida/agradecimiento
- Interfaz Gradio y consola
- Manejo de errores con `try/except` para que ninguna consulta detenga el agente

## Intenciones cubiertas (12 tipos + reglas de saludo/despedida)
`carrera`, `admision`, `nivelacion`, `cronograma` (por cédula), `matricula_ordinaria`
(incluye estudiantes de niveles regulares como cuarto semestre), `tercera_matricula`,
`retiro_asignaturas`, `homologacion`, `becas_ayudas`, `carnet_estudiantil`,
`biblioteca`, `contacto_soporte`.

## Evaluación
Ejecutar:
```bash
python evaluar.py
```
Sobre 25 consultas de prueba (`data/consultas_prueba.csv`) el agente alcanza
**92% de accuracy** y **F1-macro de 0.94**. Los 2 errores restantes son
consultas con errores tipográficos severos (p. ej. "aprwebo", "nivelasion")
que no comparten raíz ni vocabulario suficiente con las utterances de la
base de conocimiento — limitación esperable de TF-IDF + stemming sin
corrección ortográfica ni embeddings semánticos, y un punto válido para
discutir en la sección de limitaciones del informe.

## Ejecución local
```bash
pip install -r requirements.txt
python app_consola.py
python evaluar.py   # evaluación con accuracy y F1-macro
```

## Ejecución en Colab
1. Abra `notebooks/Chatbot_UG_Colab_Gradio.ipynb`.
2. Ejecute las celdas.
3. Cuando se solicite, suba `data/base_conocimiento.json`.
4. Abra el enlace público de Gradio.

## Estructura
- `data/`: base JSON y CSV.
- `src/`: módulos de preprocesamiento, entidades y chatbot.
- `notebooks/`: notebook para Colab y Gradio.
- `docs/`: informe.
- `app_consola.py`: prueba local.

## Nota
El prototipo utiliza respuestas recuperadas, no generación abierta. Las fechas deben validarse contra la fuente oficial antes de uso real.
