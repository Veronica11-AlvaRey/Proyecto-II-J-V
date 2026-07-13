"""
Evaluación del agente conversacional (RF-08).

Ejecuta el chatbot contra data/consultas_prueba.csv y reporta:
- accuracy
- F1-macro
- tabla de aciertos/errores

Uso:
    python evaluar.py
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.chatbot import ChatbotUG


def evaluar(ruta_json="data/base_conocimiento.json", ruta_csv="data/consultas_prueba.csv"):
    bot = ChatbotUG(Path(ruta_json))
    df = pd.read_csv(ruta_csv)

    y_true, y_pred, filas = [], [], []
    for _, row in df.iterrows():
        _, intencion, score = bot.responder(row["consulta"], debug=True)
        y_true.append(row["categoria_esperada"])
        y_pred.append(intencion)
        filas.append((row["consulta"], row["categoria_esperada"], intencion, round(score, 3)))

    print(f"{'Consulta':<45} {'Esperado':<20} {'Obtenido':<20} {'Score':<6}")
    print("-" * 95)
    for consulta, esperado, obtenido, score in filas:
        marca = "OK" if esperado == obtenido else "X"
        print(f"{consulta[:44]:<45} {esperado:<20} {obtenido:<20} {score:<6} {marca}")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n=== Resultados ===")
    print(f"Accuracy : {acc:.2%}")
    print(f"F1-macro : {f1:.4f}")
    print("\n=== Reporte por clase ===")
    print(classification_report(y_true, y_pred, zero_division=0))

    return acc, f1


if __name__ == "__main__":
    evaluar()
