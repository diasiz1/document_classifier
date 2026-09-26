"""Text preprocessing shared by all models: normalize -> tokenize -> stem -> fixed-size chunks."""
import re

CHUNK_SIZE = 200  # tokens per chunk
CHUNK_OVERLAP = 50  # tokens shared between neighbouring chunks
STEM_LEN = 6  # crude stemming: keep first N chars (handles Kazakh/Russian suffixes); 0 disables

# Russian alphabet + Kazakh-specific letters + Latin + digits
TOKEN_RE = re.compile(r"[a-zа-яәғқңөұүһі0-9]+")

STOPWORDS = {
    # russian
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
    "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему",
    "или", "ли", "если", "уже", "для", "при", "это", "этот", "также", "их", "который",
    # kazakh
    "және", "мен", "бен", "пен", "бұл", "осы", "сол", "ол", "де", "та", "те",
    "үшін", "бойынша", "немесе", "әр", "бір", "екі", "оның", "олар", "ретінде",
    "туралы", "арқылы", "болып", "болады", "емес", "қана", "ғана", "тиіс",
}


def tokenize(text, stem_len=STEM_LEN):
    text = text.lower().replace("ё", "е")
    tokens = [t for t in TOKEN_RE.findall(text) if len(t) > 1 and t not in STOPWORDS]
    if stem_len:
        tokens = [t[:stem_len] for t in tokens]
    return tokens


def chunk_tokens(tokens, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if len(tokens) <= size:
        return [tokens] if tokens else []
    step = size - overlap
    return [tokens[i:i + size] for i in range(0, len(tokens) - overlap, step)]


def preprocess(text):
    """Raw text -> list of chunk strings (tokens joined by single spaces)."""
    return [" ".join(chunk) for chunk in chunk_tokens(tokenize(text))]


def to_chunks(texts, labels=None):
    """Expand documents into chunks. Returns (chunks, chunk_labels, doc_ids)."""
    chunks, chunk_labels, doc_ids = [], [], []
    for i, text in enumerate(texts):
        for chunk in preprocess(text):
            chunks.append(chunk)
            doc_ids.append(i)
            if labels is not None:
                chunk_labels.append(labels[i])
    return chunks, chunk_labels, doc_ids
