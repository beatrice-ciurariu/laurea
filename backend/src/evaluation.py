import re
import mistune
from bs4 import BeautifulSoup


def remove_markdown(text):
    if not text:
        return ""
    html = mistune.html(text)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(True):
        tag.unwrap()
    plain_text = re.sub(r"[ \t]+", " ", str(soup))
    return re.sub(r"\n+", "\n", plain_text).strip()

def tokenize(text):
    if not text:
        return set()
    text = remove_markdown(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return set(text.split())

def evaluate(parsed, gold):
    t_p = tokenize(parsed)
    t_g = tokenize(gold)

    inter = t_p & t_g

    precision = len(inter) / len(t_p) if t_p else 0
    recall = len(inter) / len(t_g) if t_g else 0
    
    # Calcolo F1 standard
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return {
        "token_level_eval": {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    }
