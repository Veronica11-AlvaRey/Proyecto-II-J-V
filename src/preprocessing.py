
import re
import unicodedata

try:
    from nltk.stem.snowball import SnowballStemmer
    _STEMMER = SnowballStemmer("spanish")
except ImportError:
    _STEMMER = None

# Lista propia de stopwords en español (artículos, preposiciones, pronombres,
# conjunciones y verbos auxiliares muy frecuentes). Se define localmente en
# lugar de descargar el corpus de NLTK para que el proyecto funcione sin
# conexión a internet y para controlar exactamente qué palabras se eliminan.
STOPWORDS_ES = {
    "a","al","algo","algunas","algunos","ante","antes","como","con","contra",
    "cual","cuando","de","del","desde","donde","durante","e","el","ella",
    "ellas","ellos","en","entre","era","erais","eramos","eran","eras","eres",
    "es","esa","esas","ese","eso","esos","esta","estaba","estabais","estaban",
    "estabas","estad","estada","estadas","estado","estados","estais","estamos",
    "estan","estando","estar","estara","estaran","estaras","estare","estareis",
    "estaremos","estaria","estariais","estariamos","estarian","estarias",
    "estas","este","esto","estos","estoy","fue","fuera","fuerais","fueramos",
    "fueran","fueras","fueron","fui","fuimos","ha","habia","habiais",
    "habiamos","habian","habias","habida","habidas","habido","habidos",
    "habiendo","habra","habran","habras","habre","habreis","habremos",
    "habria","habriais","habriamos","habrian","habrias","han","has","hasta",
    "hay","haya","hayamos","hayan","hayas","he","hemos","hube","hubiera",
    "hubierais","hubieramos","hubieran","hubieras","hubieron","hubiese",
    "la","las","le","les","lo","los","mas","me","mi","mia","mias","mientras",
    "mio","mios","mis","mucho","muchos","muy","nada","ni","no","nos",
    "nosotras","nosotros","nuestra","nuestras","nuestro","nuestros","o",
    "os","otra","otras","otro","otros","para","pero","poco","por","porque",
    "que","quien","quienes","se","sea","seamos","sean","seas","sera",
    "seran","seras","sere","sereis","seremos","seria","seriais","seriamos",
    "serian","serias","si","sido","siendo","sin","sobre","sois","somos",
    "son","soy","su","sus","suya","suyas","suyo","suyos","tambien","tanto",
    "te","tenia","teniamos","tenian","tenias","ti","tiene","tienen",
    "tienes","todo","todos","tu","tus","tuya","tuyas","tuyo","tuyos",
    "un","una","uno","unos","vosotras","vosotros","vuestra","vuestras",
    "vuestro","vuestros","y","ya","yo",
}


def normalizar_texto(texto: str) -> str:
    """Limpieza básica: minúsculas, sin tildes, sin signos/caracteres especiales."""
    texto = str(texto).lower().strip()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def preprocesar_texto(texto: str) -> str:
    """
    Pipeline completo de preprocesamiento (RF-02):
    1) normalización (minúsculas, sin tildes/puntuación)
    2) tokenización (split simple por espacios)
    3) eliminación de stopwords
    4) stemming (SnowballStemmer en español)
    Devuelve un string ya listo para vectorizar con TF-IDF.
    """
    texto_normalizado = normalizar_texto(texto)
    tokens = texto_normalizado.split()
    tokens = [t for t in tokens if t not in STOPWORDS_ES and len(t) > 1]
    if _STEMMER is not None:
        tokens = [_STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)
