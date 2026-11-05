from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Sequence


DEFAULT_BERTSCORE_MODEL = "bert-base-multilingual-cased"
DEFAULT_BERTSCORE_LAYERS = 9
DEFAULT_BERTSCORE_MAX_TOKENS = 510
DEFAULT_BERTSCORE_BATCH_SIZE = 8
DEFAULT_BERTSCORE_MAX_WINDOW_PAIRS = 4096
DEFAULT_BERTSCORE_MAX_WINDOWS_PER_TEXT = 128

_BERTSCORE_ASSETS_LOCK = Lock()
_BERTSCORE_SCORE_LOCK = Lock()
_BERTSCORE_TOKENIZER_KEY: str | None = None
_BERTSCORE_TOKENIZER: Any | None = None
_BERTSCORE_SCORER_KEY: tuple[str, int, int] | None = None
_BERTSCORE_SCORER: Any | None = None


class MetricDependencyError(RuntimeError):
    pass


class MetricInputSizeError(ValueError):
    pass

def _normalize(text: str) -> str:
    from evaluation import remove_markdown

    return remove_markdown(text)


def _load_meteor_scorer(language: str) -> Callable[[Sequence[Sequence[str]], Sequence[str]], float]:
    if language not in {"en", "it"}:
        raise ValueError("METEOR language must be 'en' or 'it'.")

    try:
        from nltk.corpus import wordnet
        from nltk.stem import PorterStemmer, SnowballStemmer
        from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
    except ImportError as exc:
        raise MetricDependencyError(
            "METEOR requires NLTK, which is unavailable in this backend image."
        ) from exc

    try:
        wordnet.ensure_loaded()
        if language == "en":
            stemmer = PorterStemmer()
            language_wordnet = wordnet
        else:
            import nltk

            nltk.data.find("corpora/omw-1.4/")
            stemmer = SnowballStemmer("italian")

            class ItalianWordNet:
                @staticmethod
                def synsets(word: str) -> list[Any]:
                    return wordnet.synsets(word, lang="ita")

            language_wordnet = ItalianWordNet()
    except LookupError as exc:
        corpus = "WordNet" if language == "en" else "WordNet and OMW 1.4 Italian"
        raise MetricDependencyError(
            f"METEOR requires the NLTK {corpus} corpus; the Docker asset initializer did not make it available."
        ) from exc

    def score(references: Sequence[Sequence[str]], hypothesis: Sequence[str]) -> float:
        return float(
            nltk_meteor_score(
                references,
                hypothesis,
                stemmer=stemmer,
                wordnet=language_wordnet,
            )
        )

    return score


def meteor_eval(parsed_text: str, gold_text: str, language: str = "en") -> dict[str, Any]:

    if language not in {"en", "it"}:
        raise ValueError("METEOR language must be 'en' or 'it'.")

    hypothesis = _normalize(parsed_text).lower().split() if parsed_text else []
    reference = _normalize(gold_text).lower().split() if gold_text else []
    if not hypothesis or not reference:
        return {"meteor": {"score": 0.0, "language": language}}

    scorer = _load_meteor_scorer(language)
    score = scorer([reference], hypothesis)
    return {"meteor": {"score": float(score), "language": language}}


def _token_chunks(text: str, tokenizer: Any, max_tokens: int) -> list[tuple[str, int]]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be greater than zero.")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(token_ids), max_tokens):
        window = token_ids[start : start + max_tokens]
        chunks.append((tokenizer.decode(window, skip_special_tokens=True), len(window)))
    return chunks


def _model_location(model_name: str) -> str:
    configured_path = os.environ.get("BERTSCORE_MODEL_PATH")
    if model_name == DEFAULT_BERTSCORE_MODEL and configured_path:
        path = Path(configured_path)
        if not path.is_dir():
            raise MetricDependencyError(
                "BERTScore model assets are unavailable; the Docker asset initializer "
                f"did not populate {configured_path}."
            )
        return str(path)
    return model_name


def _load_bertscore_tokenizer(model_name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise MetricDependencyError(
            "BERTScore requires bert-score, transformers, and PyTorch; "
            "install them only after approving the dependency change."
        ) from exc

    location = _model_location(model_name)
    try:
        return AutoTokenizer.from_pretrained(location, local_files_only=True)
    except (OSError, ValueError) as exc:
        raise MetricDependencyError(
            "BERTScore tokenizer is not available in the local model cache; "
            "the Docker asset initializer may have failed."
        ) from exc


def _load_bertscore_scorer(model_name: str, num_layers: int, batch_size: int) -> Any:
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise MetricDependencyError(
            "BERTScore requires bert-score, transformers, and PyTorch; "
            "install them only after approving the dependency change."
        ) from exc

    location = _model_location(model_name)
    try:
        return BERTScorer(
            model_type=location,
            num_layers=num_layers,
            batch_size=batch_size,
            device="cpu",
            idf=False,
            rescale_with_baseline=False,
        )
    except (OSError, ValueError) as exc:
        raise MetricDependencyError(
            "BERTScore model weights are not available in the local cache; "
            "the Docker asset initializer may have failed."
        ) from exc


def _get_bertscore_tokenizer(model_name: str) -> Any:
    global _BERTSCORE_TOKENIZER_KEY, _BERTSCORE_TOKENIZER
    with _BERTSCORE_ASSETS_LOCK:
        if _BERTSCORE_TOKENIZER_KEY != model_name or _BERTSCORE_TOKENIZER is None:
            _BERTSCORE_TOKENIZER = _load_bertscore_tokenizer(model_name)
            _BERTSCORE_TOKENIZER_KEY = model_name
        return _BERTSCORE_TOKENIZER


def _get_bertscore_scorer(model_name: str, num_layers: int, batch_size: int) -> Any:
    global _BERTSCORE_SCORER_KEY, _BERTSCORE_SCORER
    key = (model_name, num_layers, batch_size)
    with _BERTSCORE_ASSETS_LOCK:
        if _BERTSCORE_SCORER_KEY != key or _BERTSCORE_SCORER is None:
            _BERTSCORE_SCORER = _load_bertscore_scorer(
                model_name, num_layers, batch_size
            )
            _BERTSCORE_SCORER_KEY = key
        return _BERTSCORE_SCORER


def _model_revision(model_name: str) -> str | None:
    configured_path = os.environ.get("BERTSCORE_MODEL_PATH")
    if model_name != DEFAULT_BERTSCORE_MODEL or not configured_path:
        return None
    manifest_path = Path(configured_path).parents[1] / "advanced_assets.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("bertscore", {}).get("resolved_revision")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def _as_float_list(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def _maximum_weight_assignment(weights: list[list[float]]) -> list[tuple[int, int]]:
    row_count = len(weights)
    if row_count == 0:
        return []
    col_count = len(weights[0])
    if col_count == 0:
        return []
    if any(len(row) != col_count for row in weights):
        raise ValueError("Assignment weight rows must have equal length.")

    size = max(row_count, col_count)
    padded_weights = [[0.0] * size for _ in range(size)]
    for row in range(row_count):
        for col in range(col_count):
            padded_weights[row][col] = max(0.0, weights[row][col])

    maximum = max(max(row) for row in padded_weights)
    costs = [[maximum - value for value in row] for row in padded_weights]
    row_potential = [0.0] * (size + 1)
    col_potential = [0.0] * (size + 1)
    matched_row = [0] * (size + 1)
    previous_col = [0] * (size + 1)

    for row in range(1, size + 1):
        matched_row[0] = row
        min_cost = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        col = 0
        while True:
            used[col] = True
            active_row = matched_row[col]
            delta = float("inf")
            next_col = 0
            for candidate_col in range(1, size + 1):
                if used[candidate_col]:
                    continue
                reduced_cost = (
                    costs[active_row - 1][candidate_col - 1]
                    - row_potential[active_row]
                    - col_potential[candidate_col]
                )
                if reduced_cost < min_cost[candidate_col]:
                    min_cost[candidate_col] = reduced_cost
                    previous_col[candidate_col] = col
                if min_cost[candidate_col] < delta:
                    delta = min_cost[candidate_col]
                    next_col = candidate_col
            for candidate_col in range(size + 1):
                if used[candidate_col]:
                    row_potential[matched_row[candidate_col]] += delta
                    col_potential[candidate_col] -= delta
                else:
                    min_cost[candidate_col] -= delta
            col = next_col
            if matched_row[col] == 0:
                break
        while True:
            next_col = previous_col[col]
            matched_row[col] = matched_row[next_col]
            col = next_col
            if col == 0:
                break

    return [
        (matched_row[col] - 1, col - 1)
        for col in range(1, size + 1)
        if matched_row[col] != 0
        and matched_row[col] - 1 < row_count
        and col - 1 < col_count
        and weights[matched_row[col] - 1][col - 1] > 0
    ]


def bertscore_eval(
    parsed_text: str,
    gold_text: str,
    *,
    model_name: str = DEFAULT_BERTSCORE_MODEL,
    num_layers: int = DEFAULT_BERTSCORE_LAYERS,
    max_tokens: int = DEFAULT_BERTSCORE_MAX_TOKENS,
    batch_size: int = DEFAULT_BERTSCORE_BATCH_SIZE,
    max_window_pairs: int = DEFAULT_BERTSCORE_MAX_WINDOW_PAIRS,
    max_windows_per_text: int = DEFAULT_BERTSCORE_MAX_WINDOWS_PER_TEXT,
) -> dict[str, Any]:
    if max_tokens < 1 or max_window_pairs < 1 or max_windows_per_text < 1:
        raise ValueError("Token and window-pair limits must be greater than zero.")

    hypothesis_text = _normalize(parsed_text) if parsed_text else ""
    reference_text = _normalize(gold_text) if gold_text else ""
    metadata = {
        "model": model_name,
        "model_revision": _model_revision(model_name),
        "num_layers": num_layers,
        "max_tokens_per_chunk": max_tokens,
        "chunk_strategy": "all_pair_window_max_weight_matching_v2",
        "max_window_pairs": max_window_pairs,
        "max_windows_per_text": max_windows_per_text,
    }
    if not hypothesis_text.strip() or not reference_text.strip():
        return {
            "bertscore": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                **metadata,
            }
        }

    tokenizer = _get_bertscore_tokenizer(model_name)
    hypothesis_chunks = _token_chunks(hypothesis_text, tokenizer, max_tokens)
    reference_chunks = _token_chunks(reference_text, tokenizer, max_tokens)

    pair_count = len(hypothesis_chunks) * len(reference_chunks)
    if (
        len(hypothesis_chunks) > max_windows_per_text
        or len(reference_chunks) > max_windows_per_text
        or pair_count > max_window_pairs
    ):
        raise MetricInputSizeError(
            f"BERTScore window alignment requires {pair_count} pairs; "
            f"the limits are {max_window_pairs} pairs and "
            f"{max_windows_per_text} windows per text."
        )

    scorer = _get_bertscore_scorer(model_name, num_layers, batch_size)

    candidate_windows = [
        hypothesis[0] for hypothesis in hypothesis_chunks for _reference in reference_chunks
    ]
    reference_windows = [
        reference[0] for _hypothesis in hypothesis_chunks for reference in reference_chunks
    ]
    with _BERTSCORE_SCORE_LOCK:
        precision_values, recall_values, f1_values = scorer.score(
            candidate_windows, reference_windows
        )
    precision_values = _as_float_list(precision_values)
    recall_values = _as_float_list(recall_values)
    f1_values = _as_float_list(f1_values)

    hypothesis_count = len(hypothesis_chunks)
    reference_count = len(reference_chunks)
    assignment_weights = [
        [0.0] * reference_count for _ in range(hypothesis_count)
    ]
    for hypothesis_index, (_, hypothesis_tokens) in enumerate(hypothesis_chunks):
        for reference_index, (_, reference_tokens) in enumerate(reference_chunks):
            pair_index = hypothesis_index * reference_count + reference_index
            assignment_weights[hypothesis_index][reference_index] = (
                f1_values[pair_index] * min(hypothesis_tokens, reference_tokens)
            )

    matched_pairs = _maximum_weight_assignment(assignment_weights)

    hypothesis_token_count = sum(count for _, count in hypothesis_chunks)
    reference_token_count = sum(count for _, count in reference_chunks)
    precision = sum(
        precision_values[hypothesis_index * reference_count + reference_index]
        * hypothesis_chunks[hypothesis_index][1]
        for hypothesis_index, reference_index in matched_pairs
    ) / hypothesis_token_count
    recall = sum(
        recall_values[hypothesis_index * reference_count + reference_index]
        * reference_chunks[reference_index][1]
        for hypothesis_index, reference_index in matched_pairs
    ) / reference_token_count
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "bertscore": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hypothesis_windows": hypothesis_count,
            "reference_windows": reference_count,
            "matched_windows": len(matched_pairs),
            **metadata,
        }
    }
