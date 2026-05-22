import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text: str) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens