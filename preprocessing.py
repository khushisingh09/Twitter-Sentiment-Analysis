
import re
import html
import nltk
import emoji

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Download NLTK stopwords only if not already available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


STOP_WORDS = set(stopwords.words("english"))
PORTER = PorterStemmer()



def preprocess(text):

    if not isinstance(text, str):
        return ""

    # 1. Decode HTML entities
    text = html.unescape(text)

    # 2. Remove URLs BEFORE modifying colons
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # 3. Convert emojis into readable text
    text = emoji.demojize(text, language="en")

    # 4. Convert emoji delimiters and underscores into spaces
    text = text.replace(":", " ")
    text = text.replace("_", " ")

    # 5. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 6. Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", " ", text)

    # 7. Keep only alphabetic characters
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # 8. Convert to lowercase
    text = text.lower()

    # 9. Remove stopwords and apply stemming
    tokens = [
        PORTER.stem(word)
        for word in text.split()
        if word not in STOP_WORDS
    ]

    # 10. Return cleaned text
    return " ".join(tokens)

