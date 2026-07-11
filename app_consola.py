
from pathlib import Path
from src.chatbot import ChatbotUG

bot = ChatbotUG(Path("data/base_conocimiento.json"))

print("Chatbot UG. Escribe 'salir' para terminar.")
while True:
    consulta = input("\nTú: ")
    if consulta.lower().strip() == "salir":
        break
    respuesta, intencion, score = bot.responder(consulta, debug=True)
    print(f"Bot: {respuesta}")
    print(f"[intención={intencion} | similitud={score:.4f}]")
