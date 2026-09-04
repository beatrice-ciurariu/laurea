from parsers.wikipedia import parse_wikipedia
from parsers.fifa import parse_fifa
from parsers.espn import parse_espn
from parsers.yahoo import parse_yahoo
from urllib.parse import urlparse

def get_parser(url: str):
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    parsers = {
        "en.wikipedia.org": parse_wikipedia,
        "www.fifa.com": parse_fifa,
        "www.espn.com": parse_espn,
        "finance.yahoo.com": parse_yahoo,
    }
    try:
        return parsers[host]
    except KeyError as exc:
        raise ValueError("Domain not supported") from exc
