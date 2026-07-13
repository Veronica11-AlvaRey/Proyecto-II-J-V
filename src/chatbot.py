
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocessing import normalizar_texto, preprocesar_texto
from .entities import extraer_ultimo_digito, es_saludo, es_despedida

# Respuesta fija para saludos (regla, no pasa por TF-IDF/similitud coseno).
SALUDO_RESPUESTA = (
    "¡Hola! Soy el asistente virtual de Admisión y Nivelación de la UG. \n"
    "Puedo ayudarte con temas como: carreras y modalidades, requisitos de "
    "admisión, cronograma según tu cédula, matrícula (nivelación y ordinaria), "
    "homologación, becas y trámites en el SIUG.\n"
    "¿Sobre qué te gustaría preguntar?"
)

# Respuesta fija para despedidas/agradecimientos.
DESPEDIDA_RESPUESTA = (
    "¡Con gusto! Si tienes otra consulta sobre admisión, nivelación o vida "
    "universitaria, aquí estaré. ¡Éxitos! "
)

# Mensaje de fallback: se muestra siempre que la similitud no supere el umbral,
# para que el usuario sepa exactamente qué puede reformular.
FALLBACK_RESPUESTA = (
    "No encontré una respuesta suficientemente clara para tu consulta.\n"
    "Intenta reformularla mencionando, por ejemplo: una carrera, 'requisitos "
    "de admisión', 'cronograma', 'nivelación', 'matrícula ordinaria', "
    "'homologación' o 'becas'."
)


class ChatbotUG:
    def __init__(self, ruta_json, umbral=0.18):
        self.umbral = umbral
        self.registros = json.loads(Path(ruta_json).read_text(encoding="utf-8"))
        self.textos = [self._texto_registro(r) for r in self.registros]
        self.vectorizador = TfidfVectorizer(ngram_range=(1, 2))
        self.matriz = self.vectorizador.fit_transform(self.textos)

    def _texto_registro(self, r):
        campos = ["tipo","titulo","carrera","modalidad","descripcion","detalle","facultad","bloque_nombre"]
        return preprocesar_texto(" ".join(str(r.get(c,"")) for c in campos))

    def _regla_cedula(self, consulta):
        ultimo = extraer_ultimo_digito(consulta)
        if ultimo is None:
            return None
        mapa = {
            "1":"lunes 13-jul-26","2":"lunes 13-jul-26",
            "3":"martes 14-jul-26","4":"martes 14-jul-26",
            "5":"miércoles 15-jul-26","6":"miércoles 15-jul-26",
            "7":"jueves 16-jul-26","8":"jueves 16-jul-26",
            "9":"viernes 17-jul-26","0":"viernes 17-jul-26"
        }
        return f"Tu documento termina en {ultimo}. Puedes ingresar desde las 09:00 am el {mapa[ultimo]}. Para todos también se habilita del 18 al 20 de julio de 2026."

    def responder(self, consulta, debug=False):
        try:
            if not str(consulta).strip():
                msg = "Escribe una consulta. Por ejemplo: '¿Qué modalidad tiene Medicina?'"
                return (msg, "vacio", 0.0) if debug else msg

            # Regla 1: saludo (se evalúa antes que nada, incluso si la frase
            # trae signos de puntuación o va acompañada de otra palabra).
            if es_saludo(consulta):
                return (SALUDO_RESPUESTA, "saludo", 1.0) if debug else SALUDO_RESPUESTA

            # Regla 2: despedida o agradecimiento.
            if es_despedida(consulta):
                return (DESPEDIDA_RESPUESTA, "despedida", 1.0) if debug else DESPEDIDA_RESPUESTA

            # Regla 3: cronograma por último dígito de cédula/pasaporte.
            regla = self._regla_cedula(consulta)
            if regla and any(x in normalizar_texto(consulta) for x in ["cedula","pasaporte","digito","inscrib","postul"]):
                return (regla, "cronograma_cedula", 1.0) if debug else regla

            # Búsqueda por similitud coseno sobre TF-IDF (con stopwords/stemming aplicados).
            q = self.vectorizador.transform([preprocesar_texto(consulta)])
            sims = cosine_similarity(q, self.matriz).flatten()
            idx = int(sims.argmax())
            score = float(sims[idx])
            if score < self.umbral:
                return (FALLBACK_RESPUESTA, "fallback", score) if debug else FALLBACK_RESPUESTA
        except Exception as e:
            msg = f"Ocurrió un problema al procesar tu consulta ({e}). Intenta reformularla."
            return (msg, "error", 0.0) if debug else msg
        r = self.registros[idx]
        partes = []
        if r.get("carrera"): partes.append(f"Carrera: {r['carrera']}")
        if r.get("modalidad"): partes.append(f"Modalidad: {r['modalidad']}")
        if r.get("titulo"): partes.append(f"Tema: {r['titulo']}")
        if r.get("descripcion"): partes.append(r["descripcion"])
        if r.get("detalle"): partes.append(r["detalle"])
        msg = "\n\n".join(partes)
        return (msg, r.get("tipo","consulta"), score) if debug else msg
