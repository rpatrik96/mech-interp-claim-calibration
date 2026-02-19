#!/usr/bin/env python3
"""
Multi-LLM Annotation Pipeline for H2 Pilot Study

Sends papers + claims + codebook to LLMs and collects structured annotations.

Two modes:
  --mode classify  Re-classify existing claims from annotations.csv
  --mode extract   Extract new claims from paper text

Usage:
    python annotate.py --mode classify --model claude --papers all
    python annotate.py --mode extract --model gpt5 --papers 2211.00593,2202.05262
    python annotate.py --mode classify --model claude --papers 2211.00593 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas for structured output
# ---------------------------------------------------------------------------


class ClaimClassification(BaseModel):
    """Mode A: classify an existing claim."""

    method_used: str
    method_rung: Literal[1, 2, 3]
    claim_rung: Literal[1, 2, 3]
    confidence: Literal[1, 2, 3, 4, 5]
    hedge_flag: Literal[0, 1]
    reasoning: str


class ClaimExtraction(BaseModel):
    """Mode B: extract and classify a new claim."""

    claim_text: str
    claim_location: Literal[
        "abstract", "introduction", "body", "results", "discussion", "conclusion"
    ]
    claim_prominence: Literal[1, 2, 3]
    method_used: str
    method_rung: Literal[1, 2, 3]
    claim_rung: Literal[1, 2, 3]
    confidence: Literal[1, 2, 3, 4, 5]
    hedge_flag: Literal[0, 1]
    reasoning: str


class PaperAnnotation(BaseModel):
    paper_id: str
    claims: list[ClaimExtraction]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert annotation assistant for a mechanistic interpretability "
    "research study.\n"
    "Your task is to annotate empirical claims from ML papers following a "
    "structured codebook.\n"
    "Follow the codebook instructions exactly. Apply the decision rules for "
    "edge cases, including the decision trees for polysemous terms.\n"
    "Rate your confidence honestly on the 1-5 scale."
)


def _build_classify_user_prompt(
    codebook: str,
    calibration: str,
    paper_id: str,
    title: str,
    paper_text: str,
    claim_text: str,
    claim_location: str,
) -> str:
    return (
        f"## Annotation Codebook\n{codebook}\n\n"
        f"## Calibration Examples\n{calibration}\n\n"
        f"## Paper Context\nPaper ID: {paper_id}\nTitle: {title}\n"
        f"Full text:\n{paper_text}\n\n"
        f'## Claim to Classify\n"{claim_text}"\n'
        f"Location in paper: {claim_location}\n\n"
        "## Task\n"
        "Classify this claim's method rung and claim rung following the codebook."
    )


def _build_extract_user_prompt(
    codebook: str,
    calibration: str,
    paper_id: str,
    title: str,
    paper_text: str,
) -> str:
    return (
        f"## Annotation Codebook\n{codebook}\n\n"
        f"## Calibration Examples\n{calibration}\n\n"
        f"## Paper Context\nPaper ID: {paper_id}\nTitle: {title}\n"
        f"Full text:\n{paper_text}\n\n"
        "## Task\n"
        "Extract all empirical claims about model internals from this paper. "
        "For each claim, provide the claim text, its location, prominence, "
        "method used, method rung, claim rung, confidence, hedge flag, "
        "and reasoning following the codebook."
    )


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


def _call_anthropic(
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call Anthropic API with tool_use for structured output.

    Returns (parsed_dict, raw_response_metadata).
    """
    import anthropic

    client = anthropic.Anthropic()

    # Build a tool definition from the Pydantic schema.
    tool_name = schema.__name__
    tool_schema = schema.model_json_schema()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        tools=[
            {
                "name": tool_name,
                "description": schema.__doc__ or tool_name,
                "input_schema": tool_schema,
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user}],
    )

    # Extract the tool_use block.
    tool_block = next(
        (b for b in response.content if b.type == "tool_use"), None
    )
    if tool_block is None:
        raise RuntimeError("Anthropic response contained no tool_use block")

    parsed = tool_block.input  # already a dict

    metadata = {
        "model": response.model,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "stop_reason": response.stop_reason,
    }
    return parsed, metadata


def _call_openai(
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call OpenAI API with response_format for structured output.

    Returns (parsed_dict, raw_response_metadata).
    """
    import openai

    client = openai.OpenAI()

    response = client.beta.chat.completions.parse(
        model=model,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=schema,
    )

    choice = response.choices[0]
    parsed = json.loads(choice.message.content)

    metadata = {
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "finish_reason": choice.finish_reason,
    }
    return parsed, metadata


def _call_openrouter(
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call OpenRouter API (OpenAI-compatible) with JSON output.

    Uses json_object mode with the schema embedded in the prompt for broad
    model compatibility (not all providers support json_schema). Validates
    the response against the Pydantic schema after parsing.

    Returns (parsed_dict, raw_response_metadata).
    """
    import os

    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=10,
        timeout=120,
    )

    json_schema = schema.model_json_schema()

    # Embed the schema in the system prompt so the model knows the expected format
    schema_instruction = (
        "\n\nYou MUST respond with valid JSON matching this schema exactly:\n"
        f"```json\n{json.dumps(json_schema, indent=2)}\n```\n"
        "Output ONLY the JSON object, no other text."
    )

    # Some models (e.g. Qwen) are only available via providers not in
    # OpenRouter's default list.  Explicitly allow all providers.
    extra: dict[str, Any] = {"provider": {"allow_fallbacks": True}}

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system + schema_instruction},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_body=extra,
    )

    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("Model returned empty content (likely exhausted tokens on reasoning)")
    parsed = json.loads(raw)
    result = schema.model_validate(parsed)

    metadata = {
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "finish_reason": response.choices[0].finish_reason,
    }
    return result.model_dump(), metadata


def _strip_schema_titles(schema_dict: dict) -> dict:
    """Recursively remove 'title' keys from a JSON schema.

    Some providers (e.g. Google via OpenRouter) reject Pydantic metadata fields.
    """
    schema_dict = {k: v for k, v in schema_dict.items() if k != "title"}
    if "properties" in schema_dict:
        schema_dict["properties"] = {
            k: _strip_schema_titles(v) for k, v in schema_dict["properties"].items()
        }
    if "items" in schema_dict and isinstance(schema_dict["items"], dict):
        schema_dict["items"] = _strip_schema_titles(schema_dict["items"])
    if "$defs" in schema_dict:
        schema_dict["$defs"] = {
            k: _strip_schema_titles(v) for k, v in schema_dict["$defs"].items()
        }
    return schema_dict


def _coerce_schema_enums_to_strings(schema_dict: dict) -> dict:
    """Recursively convert integer enum values to strings in a JSON schema.

    Google Gemini requires all enum values to be strings.
    """
    schema_dict = schema_dict.copy()
    if "enum" in schema_dict:
        schema_dict["enum"] = [str(v) for v in schema_dict["enum"]]
        schema_dict["type"] = "string"
    if "properties" in schema_dict:
        schema_dict["properties"] = {
            k: _coerce_schema_enums_to_strings(v)
            for k, v in schema_dict["properties"].items()
        }
    if "items" in schema_dict and isinstance(schema_dict["items"], dict):
        schema_dict["items"] = _coerce_schema_enums_to_strings(schema_dict["items"])
    if "$defs" in schema_dict:
        schema_dict["$defs"] = {
            k: _coerce_schema_enums_to_strings(v)
            for k, v in schema_dict["$defs"].items()
        }
    return schema_dict


def _coerce_parsed_ints(parsed: Any, schema: type[BaseModel]) -> Any:
    """Convert string values back to ints where the Pydantic model expects int Literals."""
    if isinstance(parsed, dict):
        hints = schema.__annotations__ if hasattr(schema, "__annotations__") else {}
        result = {}
        for k, v in parsed.items():
            if isinstance(v, str) and v.lstrip("-").isdigit():
                result[k] = int(v)
            elif isinstance(v, list):
                # Handle nested models (e.g. PaperAnnotation.claims)
                item_schema = hints.get(k)
                if item_schema and hasattr(item_schema, "__args__"):
                    first_arg = item_schema.__args__[0]
                    inner = getattr(first_arg, "__args__", [None])[0]
                    if inner and hasattr(inner, "__annotations__"):
                        result[k] = [_coerce_parsed_ints(item, inner) for item in v]
                    else:
                        result[k] = v
                else:
                    result[k] = v
            else:
                result[k] = v
        return result
    return parsed


def _call_google(
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call Google Gemini API with structured output via google-genai.

    Returns (parsed_dict, raw_response_metadata).
    """
    from google import genai
    from google.genai import types

    client = genai.Client()

    # Gemini requires enum values to be strings; convert the schema.
    json_schema = _coerce_schema_enums_to_strings(schema.model_json_schema())

    response = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=json_schema,
        ),
    )

    parsed = _coerce_parsed_ints(json.loads(response.text), schema)

    metadata = {
        "model": model,
        "usage": {
            "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", None),
            "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", None),
            "total_tokens": getattr(response.usage_metadata, "total_token_count", None),
        },
        "finish_reason": str(getattr(response.candidates[0], "finish_reason", "")) if response.candidates else "",
    }
    return parsed, metadata


def annotate_claim(
    provider: str,
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Dispatch to the right provider and return (parsed, metadata)."""
    if provider == "anthropic":
        return _call_anthropic(model, system, user, schema, temperature, max_tokens)
    if provider == "openai":
        return _call_openai(model, system, user, schema, temperature, max_tokens)
    if provider == "google":
        return _call_google(model, system, user, schema, temperature, max_tokens)
    if provider == "openrouter":
        return _call_openrouter(model, system, user, schema, temperature, max_tokens)
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_api_call(
    log_dir: Path,
    paper_id: str,
    claim_id: str,
    mode: str,
    system_prompt: str,
    user_prompt: str,
    parsed: dict[str, Any],
    metadata: dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> None:
    """Write full API call details to a JSON file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{paper_id}_{claim_id}_{mode}.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper_id": paper_id,
        "claim_id": claim_id,
        "mode": mode,
        "prompt": {"system": system_prompt, "user": user_prompt},
        "parameters": {"temperature": temperature, "max_tokens": max_tokens},
        "response": parsed,
        "metadata": metadata,
    }
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.debug("Logged API call to %s", log_path)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text())


def load_annotations(csv_path: Path) -> list[dict[str, str]]:
    """Load existing annotations CSV (Mode A input)."""
    rows: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("paper_id"):
                rows.append(row)
    return rows


def load_paper_titles(csv_path: Path) -> dict[str, str]:
    """Map paper_id -> title from candidate_papers.csv."""
    titles: dict[str, str] = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("paper_id"):
                titles[row["paper_id"]] = row.get("title", "")
    return titles


def _load_existing_results(csv_path: Path) -> list[dict[str, str]]:
    """Load already-annotated results from an output CSV for resume support."""
    if not csv_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_paper_text(paper_texts_dir: Path, paper_id: str) -> str:
    path = paper_texts_dir / f"{paper_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Paper text not found: {path}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def save_run_metadata(
    output_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    model_key: str,
    mode: str,
    paper_ids: list[str],
    n_claims: int,
) -> None:
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "model_key": model_key,
        "model_config": config["models"][model_key],
        "paper_ids": paper_ids,
        "n_claims_processed": n_claims,
        "config_sha256": _file_sha256(config_path),
        "codebook_sha256": _file_sha256(
            config_path.parent / config["annotation"]["codebook_path"]
        ),
        "calibration_sha256": _file_sha256(
            config_path.parent / config["annotation"]["calibration_path"]
        ),
    }
    out = output_dir / "run_metadata.json"
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info("Run metadata saved to %s", out)


# ---------------------------------------------------------------------------
# Mode A: classify existing claims
# ---------------------------------------------------------------------------


def run_classify(
    config: dict[str, Any],
    config_path: Path,
    model_key: str,
    paper_ids: list[str] | None,
    dry_run: bool,
) -> None:
    base_dir = config_path.parent
    model_cfg = config["models"][model_key]
    ann_cfg = config["annotation"]

    codebook = (base_dir / ann_cfg["codebook_path"]).read_text(encoding="utf-8")
    calibration = (base_dir / ann_cfg["calibration_path"]).read_text(encoding="utf-8")
    paper_texts_dir = base_dir / ann_cfg["paper_texts_dir"]
    output_dir = base_dir / ann_cfg["output_dir"] / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"

    titles = load_paper_titles(base_dir / "candidate_papers.csv")
    claims = load_annotations(base_dir / "annotations.csv")

    # Filter to requested papers.
    if paper_ids is not None:
        claims = [c for c in claims if c["paper_id"] in paper_ids]
    else:
        paper_ids = sorted({c["paper_id"] for c in claims})

    # Resume: load already-annotated claim_ids and skip them.
    out_csv = output_dir / "annotations_classify.csv"
    existing_results = _load_existing_results(out_csv)
    done_ids = {r["claim_id"] for r in existing_results}
    if done_ids:
        logger.info("Resuming: %d claims already annotated, skipping", len(done_ids))
    claims = [c for c in claims if c["claim_id"] not in done_ids]

    total = len(claims)
    logger.info("Mode: classify | Model: %s | Claims: %d", model_key, total)

    results: list[dict[str, Any]] = list(existing_results)
    consecutive_failures = 0
    max_consecutive_failures = 3

    for idx, claim in enumerate(claims, 1):
        pid = claim["paper_id"]
        cid = claim["claim_id"]
        logger.info("Processing claim %d/%d: %s", idx, total, cid)

        try:
            paper_text = load_paper_text(paper_texts_dir, pid)
        except FileNotFoundError:
            logger.warning("Paper text missing for %s — skipping claim %s", pid, cid)
            continue

        user_prompt = _build_classify_user_prompt(
            codebook=codebook,
            calibration=calibration,
            paper_id=pid,
            title=titles.get(pid, ""),
            paper_text=paper_text,
            claim_text=claim["claim_text"],
            claim_location=claim.get("claim_location", ""),
        )

        if dry_run:
            print(f"\n{'='*60}")
            print(f"[DRY RUN] Claim {idx}/{total}: {cid}")
            print(f"System prompt length: {len(SYSTEM_PROMPT)} chars")
            print(f"User prompt length: {len(user_prompt)} chars")
            print(f"{'='*60}")
            continue

        try:
            parsed, metadata = annotate_claim(
                provider=model_cfg["provider"],
                model=model_cfg["model"],
                system=SYSTEM_PROMPT,
                user=user_prompt,
                schema=ClaimClassification,
                temperature=model_cfg.get("temperature", 0),
                max_tokens=model_cfg.get("max_tokens", 8192),
            )
        except Exception:
            logger.exception("API error for claim %s — skipping", cid)
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    "Aborting: %d consecutive API failures — likely a persistent issue",
                    max_consecutive_failures,
                )
                break
            continue

        consecutive_failures = 0  # reset on success

        # Log raw API call.
        _log_api_call(
            log_dir=raw_dir,
            paper_id=pid,
            claim_id=cid,
            mode="classify",
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            parsed=parsed,
            metadata=metadata,
            temperature=model_cfg.get("temperature", 0),
            max_tokens=model_cfg.get("max_tokens", 8192),
        )

        # Compute gap_score deterministically.
        gap_score = max(0, parsed["claim_rung"] - parsed["method_rung"])

        results.append(
            {
                "paper_id": pid,
                "claim_id": cid,
                "claim_text": claim["claim_text"],
                "claim_location": claim.get("claim_location", ""),
                "claim_prominence": claim.get("claim_prominence", ""),
                "method_used": parsed["method_used"],
                "method_rung": parsed["method_rung"],
                "claim_rung": parsed["claim_rung"],
                "gap_score": gap_score,
                "confidence": parsed["confidence"],
                "hedge_flag": parsed["hedge_flag"],
                "reasoning": parsed["reasoning"],
            }
        )

    if dry_run:
        logger.info("Dry run complete — no output written.")
        return

    # Write output CSV (existing + new results merged).
    fieldnames = [
        "paper_id",
        "claim_id",
        "claim_text",
        "claim_location",
        "claim_prominence",
        "method_used",
        "method_rung",
        "claim_rung",
        "gap_score",
        "confidence",
        "hedge_flag",
        "reasoning",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(
        "Wrote %d classified claims (%d new, %d existing) to %s",
        len(results), len(results) - len(existing_results), len(existing_results), out_csv,
    )

    save_run_metadata(
        output_dir, config, config_path, model_key, "classify", paper_ids, len(results)
    )


# ---------------------------------------------------------------------------
# Mode B: extract claims from paper text
# ---------------------------------------------------------------------------


def run_extract(
    config: dict[str, Any],
    config_path: Path,
    model_key: str,
    paper_ids: list[str] | None,
    dry_run: bool,
) -> None:
    base_dir = config_path.parent
    model_cfg = config["models"][model_key]
    ann_cfg = config["annotation"]

    codebook = (base_dir / ann_cfg["codebook_path"]).read_text(encoding="utf-8")
    calibration = (base_dir / ann_cfg["calibration_path"]).read_text(encoding="utf-8")
    paper_texts_dir = base_dir / ann_cfg["paper_texts_dir"]
    output_dir = base_dir / ann_cfg["output_dir"] / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"

    titles = load_paper_titles(base_dir / "candidate_papers.csv")

    # Determine which papers to process.
    if paper_ids is None:
        paper_ids = sorted(titles.keys())

    # Resume: load already-extracted paper_ids and skip them.
    out_csv = output_dir / "annotations_extract.csv"
    existing_results = _load_existing_results(out_csv)
    done_pids = {r["paper_id"] for r in existing_results}
    if done_pids:
        logger.info("Resuming: %d papers already extracted, skipping", len(done_pids))
    paper_ids = [p for p in paper_ids if p not in done_pids]

    total = len(paper_ids)
    logger.info("Mode: extract | Model: %s | Papers: %d", model_key, total)

    results: list[dict[str, Any]] = list(existing_results)
    consecutive_failures = 0
    max_consecutive_failures = 3

    for idx, pid in enumerate(paper_ids, 1):
        logger.info("Processing paper %d/%d: %s", idx, total, pid)

        try:
            paper_text = load_paper_text(paper_texts_dir, pid)
        except FileNotFoundError:
            logger.warning("Paper text missing for %s — skipping", pid)
            continue

        user_prompt = _build_extract_user_prompt(
            codebook=codebook,
            calibration=calibration,
            paper_id=pid,
            title=titles.get(pid, ""),
            paper_text=paper_text,
        )

        if dry_run:
            print(f"\n{'='*60}")
            print(f"[DRY RUN] Paper {idx}/{total}: {pid}")
            print(f"System prompt length: {len(SYSTEM_PROMPT)} chars")
            print(f"User prompt length: {len(user_prompt)} chars")
            print(f"{'='*60}")
            continue

        try:
            parsed, metadata = annotate_claim(
                provider=model_cfg["provider"],
                model=model_cfg["model"],
                system=SYSTEM_PROMPT,
                user=user_prompt,
                schema=PaperAnnotation,
                temperature=model_cfg.get("temperature", 0),
                max_tokens=model_cfg.get("max_tokens", 8192),
            )
        except Exception:
            logger.exception("API error for paper %s — skipping", pid)
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    "Aborting: %d consecutive API failures — likely a persistent issue",
                    max_consecutive_failures,
                )
                break
            continue

        consecutive_failures = 0

        # Log raw API call (use paper-level claim_id).
        _log_api_call(
            log_dir=raw_dir,
            paper_id=pid,
            claim_id="extract",
            mode="extract",
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            parsed=parsed,
            metadata=metadata,
            temperature=model_cfg.get("temperature", 0),
            max_tokens=model_cfg.get("max_tokens", 8192),
        )

        # Unpack claims from the PaperAnnotation response.
        claims_list = parsed.get("claims", [])
        for ci, claim in enumerate(claims_list, 1):
            gap_score = max(0, claim["claim_rung"] - claim["method_rung"])
            results.append(
                {
                    "paper_id": pid,
                    "claim_id": f"{pid}-ext-{ci:02d}",
                    "claim_text": claim["claim_text"],
                    "claim_location": claim["claim_location"],
                    "claim_prominence": claim["claim_prominence"],
                    "method_used": claim["method_used"],
                    "method_rung": claim["method_rung"],
                    "claim_rung": claim["claim_rung"],
                    "gap_score": gap_score,
                    "confidence": claim["confidence"],
                    "hedge_flag": claim["hedge_flag"],
                    "reasoning": claim["reasoning"],
                }
            )

    if dry_run:
        logger.info("Dry run complete — no output written.")
        return

    # Write output CSV.
    out_csv = output_dir / "annotations_extract.csv"
    fieldnames = [
        "paper_id",
        "claim_id",
        "claim_text",
        "claim_location",
        "claim_prominence",
        "method_used",
        "method_rung",
        "claim_rung",
        "gap_score",
        "confidence",
        "hedge_flag",
        "reasoning",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(
        "Wrote %d extracted claims (%d new, %d existing) to %s",
        len(results), len(results) - len(existing_results), len(existing_results), out_csv,
    )

    save_run_metadata(
        output_dir,
        config,
        config_path,
        model_key,
        "extract",
        paper_ids,
        len(results),
    )


# ---------------------------------------------------------------------------
# Batch export / collect (for Claude Code interactive annotation)
# ---------------------------------------------------------------------------


def run_export_prompts(
    config: dict[str, Any],
    config_path: Path,
    mode: str,
    paper_ids: list[str] | None,
    output_dir: Path,
) -> None:
    """Export prompts as JSON files for interactive annotation (e.g. Claude Code).

    Each file contains the system prompt, user prompt, claim metadata, and the
    expected response JSON schema so any LLM can produce a valid response.
    """
    base_dir = config_path.parent
    ann_cfg = config["annotation"]
    codebook = (base_dir / ann_cfg["codebook_path"]).read_text(encoding="utf-8")
    calibration = (base_dir / ann_cfg["calibration_path"]).read_text(encoding="utf-8")
    paper_texts_dir = base_dir / ann_cfg["paper_texts_dir"]
    titles = load_paper_titles(base_dir / "candidate_papers.csv")

    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "classify":
        claims = load_annotations(base_dir / "annotations.csv")
        if paper_ids is not None:
            claims = [c for c in claims if c["paper_id"] in paper_ids]

        schema_json = ClaimClassification.model_json_schema()

        for idx, claim in enumerate(claims, 1):
            pid = claim["paper_id"]
            cid = claim["claim_id"]

            try:
                paper_text = load_paper_text(paper_texts_dir, pid)
            except FileNotFoundError:
                logger.warning("Paper text missing for %s — skipping claim %s", pid, cid)
                continue

            user_prompt = _build_classify_user_prompt(
                codebook=codebook,
                calibration=calibration,
                paper_id=pid,
                title=titles.get(pid, ""),
                paper_text=paper_text,
                claim_text=claim["claim_text"],
                claim_location=claim.get("claim_location", ""),
            )

            prompt_file = output_dir / f"{cid}.json"
            prompt_file.write_text(
                json.dumps(
                    {
                        "claim_id": cid,
                        "paper_id": pid,
                        "claim_text": claim["claim_text"],
                        "claim_location": claim.get("claim_location", ""),
                        "claim_prominence": claim.get("claim_prominence", ""),
                        "mode": "classify",
                        "system_prompt": SYSTEM_PROMPT,
                        "user_prompt": user_prompt,
                        "response_schema": schema_json,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        logger.info("Exported %d prompt files to %s", idx, output_dir)

    elif mode == "extract":
        if paper_ids is None:
            paper_ids = sorted(titles.keys())

        schema_json = PaperAnnotation.model_json_schema()

        for idx, pid in enumerate(paper_ids, 1):
            try:
                paper_text = load_paper_text(paper_texts_dir, pid)
            except FileNotFoundError:
                logger.warning("Paper text missing for %s — skipping", pid)
                continue

            user_prompt = _build_extract_user_prompt(
                codebook=codebook,
                calibration=calibration,
                paper_id=pid,
                title=titles.get(pid, ""),
                paper_text=paper_text,
            )

            prompt_file = output_dir / f"{pid}_extract.json"
            prompt_file.write_text(
                json.dumps(
                    {
                        "paper_id": pid,
                        "mode": "extract",
                        "system_prompt": SYSTEM_PROMPT,
                        "user_prompt": user_prompt,
                        "response_schema": schema_json,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        logger.info("Exported %d prompt files to %s", idx, output_dir)


def run_collect_responses(
    prompts_dir: Path,
    responses_dir: Path,
    output_csv: Path,
) -> None:
    """Collect JSON response files and assemble into an annotations CSV.

    Expected workflow:
      1. run_export_prompts() writes prompt files to prompts_dir/
      2. An annotator (Claude Code, human, etc.) writes response JSON files
         to responses_dir/{claim_id}.json matching the schema
      3. This function reads both, validates, and writes the CSV

    Response files should contain the fields from ClaimClassification or
    ClaimExtraction (depending on mode), keyed by claim_id filename.
    """
    prompt_files = sorted(prompts_dir.glob("*.json"))
    if not prompt_files:
        logger.error("No prompt files found in %s", prompts_dir)
        return

    results: list[dict[str, Any]] = []
    n_missing = 0
    n_invalid = 0

    for pf in prompt_files:
        prompt = json.loads(pf.read_text(encoding="utf-8"))
        cid = prompt.get("claim_id", pf.stem)
        pid = prompt.get("paper_id", "")
        mode = prompt.get("mode", "classify")

        # Look for response file
        resp_file = responses_dir / pf.name
        if not resp_file.exists():
            n_missing += 1
            continue

        try:
            resp = json.loads(resp_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.error("Invalid JSON in response file: %s", resp_file)
            n_invalid += 1
            continue

        if mode == "classify":
            try:
                validated = ClaimClassification.model_validate(resp)
            except Exception as e:
                logger.error("Validation failed for %s: %s", cid, e)
                n_invalid += 1
                continue

            gap_score = max(0, validated.claim_rung - validated.method_rung)
            results.append(
                {
                    "paper_id": pid,
                    "claim_id": cid,
                    "claim_text": prompt.get("claim_text", ""),
                    "claim_location": prompt.get("claim_location", ""),
                    "claim_prominence": prompt.get("claim_prominence", ""),
                    "method_used": validated.method_used,
                    "method_rung": validated.method_rung,
                    "claim_rung": validated.claim_rung,
                    "gap_score": gap_score,
                    "confidence": validated.confidence,
                    "hedge_flag": validated.hedge_flag,
                    "reasoning": validated.reasoning,
                }
            )

        elif mode == "extract":
            try:
                raw_claims = resp.get("claims", resp if isinstance(resp, list) else [resp])
                paper_ann = PaperAnnotation.model_validate(
                    {"paper_id": pid, "claims": raw_claims}
                )
            except Exception as e:
                logger.error("Validation failed for %s: %s", pid, e)
                n_invalid += 1
                continue

            for ci, claim in enumerate(paper_ann.claims, 1):
                gap_score = max(0, claim.claim_rung - claim.method_rung)
                results.append(
                    {
                        "paper_id": pid,
                        "claim_id": f"{pid}-ext-{ci:02d}",
                        "claim_text": claim.claim_text,
                        "claim_location": claim.claim_location,
                        "claim_prominence": claim.claim_prominence,
                        "method_used": claim.method_used,
                        "method_rung": claim.method_rung,
                        "claim_rung": claim.claim_rung,
                        "gap_score": gap_score,
                        "confidence": claim.confidence,
                        "hedge_flag": claim.hedge_flag,
                        "reasoning": claim.reasoning,
                    }
                )

    logger.info(
        "Collected %d annotations (%d missing responses, %d invalid)",
        len(results), n_missing, n_invalid,
    )

    if not results:
        logger.error("No valid annotations to write.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper_id", "claim_id", "claim_text", "claim_location",
        "claim_prominence", "method_used", "method_rung", "claim_rung",
        "gap_score", "confidence", "hedge_flag", "reasoning",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Wrote %d annotations to %s", len(results), output_csv)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-LLM annotation pipeline for H2 pilot study"
    )
    sub = parser.add_subparsers(dest="command")

    # --- Default (backward-compatible) arguments on the main parser ---
    parser.add_argument(
        "--mode",
        choices=["classify", "extract"],
        help="classify: re-annotate existing claims; extract: discover new claims",
    )
    parser.add_argument(
        "--model",
        help="Model key from annotation_config.yaml (e.g. claude, gpt5, gemini)",
    )
    parser.add_argument(
        "--papers",
        default="all",
        help='Comma-separated paper IDs or "all" (default: all)',
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "annotation_config.yaml",
        help="Path to annotation_config.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling API",
    )

    # --- export-prompts subcommand ---
    export_p = sub.add_parser(
        "export-prompts",
        help="Export prompts as JSON files for interactive annotation (Claude Code, etc.)",
    )
    export_p.add_argument("--mode", required=True, choices=["classify", "extract"])
    export_p.add_argument("--papers", default="all")
    export_p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "prompts",
        help="Directory to write prompt JSON files (default: prompts/)",
    )
    export_p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "annotation_config.yaml",
    )

    # --- collect-responses subcommand ---
    collect_p = sub.add_parser(
        "collect-responses",
        help="Assemble JSON response files into an annotation CSV",
    )
    collect_p.add_argument(
        "--prompts-dir",
        type=Path,
        required=True,
        help="Directory containing the exported prompt JSON files",
    )
    collect_p.add_argument(
        "--responses-dir",
        type=Path,
        required=True,
        help="Directory containing annotator response JSON files",
    )
    collect_p.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path for the output annotation CSV",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Dispatch ---
    if args.command == "export-prompts":
        config = load_config(args.config)
        paper_ids = None
        if args.papers != "all":
            paper_ids = [p.strip() for p in args.papers.split(",") if p.strip()]
        run_export_prompts(config, args.config.resolve(), args.mode, paper_ids, args.output_dir)
        return

    if args.command == "collect-responses":
        run_collect_responses(args.prompts_dir, args.responses_dir, args.output_csv)
        return

    # --- Original API-based flow ---
    if not args.mode or not args.model:
        parser.error("--mode and --model are required for API annotation")

    config = load_config(args.config)

    if args.model not in config["models"]:
        parser.error(
            f"Unknown model '{args.model}'. "
            f"Available: {', '.join(config['models'])}"
        )

    # Validate required env vars before entering the annotation loop.
    import os

    _PROVIDER_ENV_VARS = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    provider = config["models"][args.model]["provider"]
    env_var = _PROVIDER_ENV_VARS.get(provider)
    if env_var and env_var not in os.environ:
        parser.error(f"{env_var} environment variable is not set (required for provider '{provider}')")

    paper_ids: list[str] | None = None
    if args.papers != "all":
        paper_ids = [p.strip() for p in args.papers.split(",") if p.strip()]

    if args.mode == "classify":
        run_classify(config, args.config.resolve(), args.model, paper_ids, args.dry_run)
    else:
        run_extract(config, args.config.resolve(), args.model, paper_ids, args.dry_run)


if __name__ == "__main__":
    main()
