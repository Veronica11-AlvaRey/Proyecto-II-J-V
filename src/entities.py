
import re

def extraer_ultimo_digito(texto: str):
    numeros = re.findall(r"\d{10}", str(texto))
    if numeros:
        return numeros[0][-1]
    m = re.search(r"(?:termina en|ultimo digito(?: es)?|digito)\s*(\d)", str(texto).lower())
    return m.group(1) if m else None


# Palabras/frases que activan la regla de saludo. Se usa \b para no confundir
# "hola" dentro de otra palabra, y una lista amplia de variantes comunes.
_PATRON_SALUDO = re.compile(
    r"\b(hola+|buenos dias|buenas tardes|buenas noches|buenas|"
    r"que tal|hey|saludos|hi|hello)\b"
)

# Palabras/frases que activan la regla de despedida o agradecimiento.
_PATRON_DESPEDIDA = re.compile(
    r"\b(adios|chao|hasta luego|nos vemos|gracias|muchas gracias|"
    r"eso es todo|bye)\b"
)


def es_saludo(texto: str) -> bool:
    """Detecta si el mensaje del usuario es un saludo, mediante expresión regular."""
    return bool(_PATRON_SALUDO.search(str(texto).lower().strip()))


def es_despedida(texto: str) -> bool:
    """Detecta si el mensaje del usuario es una despedida o agradecimiento."""
    return bool(_PATRON_DESPEDIDA.search(str(texto).lower().strip()))
