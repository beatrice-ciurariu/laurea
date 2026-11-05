from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable


MODEL_ID = "bert-base-multilingual-cased"


def _model_is_ready(path: Path) -> bool:
    return (path / "config.json").is_file() and any(
        (path / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )


def _check_nltk_corpora(nltk: Any) -> tuple[bool, str | None]:
    try:
        wordnet = getattr(nltk, "_advanced_assets_wordnet", None)
        if wordnet is None:
            from nltk.corpus import wordnet

        wordnet.ensure_loaded()
        nltk.data.find("corpora/omw-1.4/")
        return True, None
    except Exception as exc:
        return False, str(exc)


def provision_assets(
    asset_root: Path,
    *,
    nltk_module: Any | None = None,
    snapshot_download_fn: Callable[..., str] | None = None,
    model_info_fn: Callable[..., Any] | None = None,
    model_revision: str = "main",
) -> dict[str, Any]:
    asset_root = Path(asset_root)
    nltk_root = asset_root / "nltk_data"
    model_root = asset_root / "models" / MODEL_ID
    manifest_path = asset_root / "advanced_assets.json"
    asset_root.mkdir(parents=True, exist_ok=True)
    nltk_root.mkdir(parents=True, exist_ok=True)
    model_root.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "nltk": {"available": False, "corpora": ["wordnet", "omw-1.4"]},
        "bertscore": {
            "available": False,
            "model": MODEL_ID,
            "requested_revision": model_revision,
        },
    }

    try:
        if nltk_module is None:
            import nltk as nltk_module

        os.environ["NLTK_DATA"] = str(nltk_root)
        if str(nltk_root) not in nltk_module.data.path:
            nltk_module.data.path.insert(0, str(nltk_root))

        ready, detail = _check_nltk_corpora(nltk_module)
        if not ready:
            downloads_ok = True
            for corpus in ("wordnet", "omw-1.4"):
                try:
                    downloads_ok = bool(
                        nltk_module.download(
                            corpus, download_dir=str(nltk_root), quiet=True
                        )
                    ) and downloads_ok
                except Exception:
                    downloads_ok = False
            ready, detail = _check_nltk_corpora(nltk_module)
            if not downloads_ok and ready:
                detail = "NLTK downloader reported a failure despite corpus lookup succeeding"
        manifest["nltk"].update({"available": ready, "detail": detail})
    except Exception as exc:
        manifest["nltk"].update({"available": False, "detail": str(exc)})

    try:
        if snapshot_download_fn is None:
            from huggingface_hub import HfApi, snapshot_download as snapshot_download_fn

            model_info_fn = model_info_fn or HfApi().model_info

        old_manifest: dict[str, Any] = {}
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

        old_bert = old_manifest.get("bertscore", {})
        cache_matches = (
            old_bert.get("available") is True
            and old_bert.get("requested_revision") == model_revision
            and Path(old_bert.get("path", "")).resolve() == model_root.resolve()
            and _model_is_ready(model_root)
        )
        if cache_matches:
            manifest["bertscore"].update(old_bert)
        else:
            if model_info_fn is not None and not re.fullmatch(r"[0-9a-f]{40}", model_revision):
                resolved_revision = model_info_fn(
                    MODEL_ID, revision=model_revision
                ).sha
            else:
                resolved_revision = model_revision
            snapshot_download_fn(
                repo_id=MODEL_ID,
                revision=resolved_revision,
                local_dir=str(model_root),
            )
            if not _model_is_ready(model_root):
                raise RuntimeError("download completed without model config or weights")
            manifest["bertscore"].update(
                {
                    "available": True,
                    "path": str(model_root),
                    "resolved_revision": resolved_revision,
                }
            )
    except Exception as exc:
        manifest["bertscore"].update({"available": False, "detail": str(exc)})

    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def main() -> int:
    root = Path(os.environ.get("ADVANCED_ASSET_ROOT", "/opt/advanced-assets"))
    revision = os.environ.get("BERTSCORE_MODEL_REVISION", "main")
    try:
        result = provision_assets(root, model_revision=revision)
        for asset, status in result.items():
            if isinstance(status, dict) and "available" in status:
                message = "ready" if status["available"] else status.get("detail", "unavailable")
                print(f"advanced asset {asset}: {message}", flush=True)
    except Exception as exc:
        print(f"advanced assets unavailable: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
