from __future__ import annotations

import gc
import ast
import json
import os
import re
from typing import Any

from iri_tfg_program.taxonomy.preference_taxonomy import (
    PREFERENCE_SIGNALS,
    SIGNAL_SEMANTICS,
    VALID_LABELS,
)


DEFAULT_PREFERENCE_WEIGHT = 5.0


class QwenDecisionLabeler:
    """Local Qwen labeler for synthetic decision supervision.

    The implementation intentionally avoids OpenAI imports/API keys and mirrors
    the local Qwen path already used by COOPERA's `habitat.gpt.query`.
    """

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct-FP8",
        temperature: float = 0.2,
        max_new_tokens: int = 384,
        json_retries: int = 1,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.json_retries = json_retries
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

        token = os.environ.get("HUGGINGFACE_TOKEN")
        model_kwargs: dict[str, Any] = {}
        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if token:
            model_kwargs["token"] = token
            tokenizer_kwargs["token"] = token

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype="auto",
                device_map="auto",
                **model_kwargs,
            )
        except TypeError:
            model_kwargs.pop("token", None)
            if token:
                model_kwargs["use_auth_token"] = token
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype="auto",
                device_map="auto",
                **model_kwargs,
            )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **tokenizer_kwargs,
            )
        except TypeError:
            # Older transformers versions used `use_auth_token`.
            tokenizer_kwargs.pop("token", None)
            if token:
                tokenizer_kwargs["use_auth_token"] = token
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **tokenizer_kwargs,
            )

    def close(self) -> None:
        if self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
        self.model = None
        self.tokenizer = None

        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def synthesize_preferences_and_label(
        self,
        *,
        sample: dict[str, Any],
        human_profile_context: dict[str, Any],
    ) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        prompt = _build_profile_grounded_prompt(
            sample=sample,
            human_profile_context=human_profile_context,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You label assistive-robot decision training data. "
                    "Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(
            text_prompt,
            return_tensors="pt",
            truncation=False,
        ).to(self.model.device)

        import torch

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        prompt_ids = inputs["input_ids"][0]
        generated_ids = outputs[0][len(prompt_ids) :]
        generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        payload = _parse_json_object(generated)
        label = str(payload.get("label_action", "")).strip()
        if label not in VALID_LABELS:
            raise ValueError(f"Qwen returned invalid label_action={label!r}: {generated}")
        snapshot = _validate_preference_snapshot(payload.get("preference_snapshot", []))
        metadata = {"labeler": "qwen_profile", "raw_response": generated, "parsed": payload}
        return snapshot, label, metadata

    def generate_json(
        self,
        *,
        system: str,
        user: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        parse_errors: list[str] = []
        raw_responses: list[str] = []

        for attempt in range(self.json_retries + 1):
            generated = self._generate_from_messages(messages)
            raw_responses.append(generated)
            try:
                payload = _parse_json_object(generated)
                return payload, {
                    "raw_response": generated,
                    "parsed": payload,
                    "json_parse_attempts": attempt + 1,
                    "json_parse_errors": parse_errors,
                    "raw_response_attempts": raw_responses,
                }
            except Exception as exc:
                parse_errors.append(f"{type(exc).__name__}: {exc}")
                if attempt >= self.json_retries:
                    raise ValueError(
                        "Qwen returned malformed JSON after "
                        f"{attempt + 1} attempt(s): {parse_errors[-1]}\n"
                        f"Raw response:\n{generated}"
                    ) from exc
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You repair malformed JSON. Return only one valid JSON "
                            "object. Do not explain anything."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "The previous assistant response was intended to be JSON, "
                            "but parsing failed.\n\n"
                            f"Parser error:\n{parse_errors[-1]}\n\n"
                            "Malformed response:\n"
                            f"{generated}\n\n"
                            "Return the corrected JSON object only."
                        ),
                    },
                ]

        raise RuntimeError("unreachable")

    def _generate_from_messages(self, messages: list[dict[str, str]]) -> str:
        text_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(
            text_prompt,
            return_tensors="pt",
            truncation=False,
        ).to(self.model.device)

        import torch

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        prompt_ids = inputs["input_ids"][0]
        generated_ids = outputs[0][len(prompt_ids) :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def _build_profile_grounded_prompt(
    *,
    sample: dict[str, Any],
    human_profile_context: dict[str, Any],
) -> str:
    compact = {
        "user_external_id": sample.get("user_external_id"),
        "action_input": sample.get("action_input"),
        "context_input": sample.get("context_input"),
        "structured_task_features": sample.get("structured_task_features"),
        "coopera_intention": sample.get("source_metadata", {}).get("coopera_intention"),
        "coopera_thought": sample.get("source_metadata", {}).get("coopera_thought"),
        "coopera_act": sample.get("source_metadata", {}).get("coopera_act"),
    }
    allowed_signals = sorted(PREFERENCE_SIGNALS)
    signal_block = "\n".join(
        f"- {signal}: {SIGNAL_SEMANTICS.get(signal, {}).get('description', '')} "
        f"prefer(high) => {SIGNAL_SEMANTICS.get(signal, {}).get('prefer_means', '')} "
        f"avoid(high) => {SIGNAL_SEMANTICS.get(signal, {}).get('avoid_means', '')}"
        for signal in allowed_signals
    )
    return (
        "You are generating synthetic training supervision for an assistive-robot "
        "personalization model.\n\n"
        "IMPORTANT: Do NOT infer personal preferences merely from the current task "
        "or time. Preferences must be grounded in the provided COOPERA human "
        "profile, Big Five/personality summary, and the profile's behavioral "
        "rationale. If the profile does not support a preference, omit it.\n\n"
        "Your job:\n"
        "1. Select zero or more behavioral preference signals that this simulated "
        "human would likely have, grounded in the profile. Each has a polarity "
        "(prefer or avoid) and a weight from 1 to 10 (how strongly it is held).\n"
        "2. Choose what the assistive robot should do in this situation for this "
        "specific human.\n\n"
        "Allowed label_action values: do_now, do_later, tell_the_user, no_action.\n"
        "Use these meanings:\n"
        "- do_now: robot should execute/help now.\n"
        "- do_later: robot should postpone the action.\n"
        "- tell_the_user: robot should tell or ask the user, not execute directly.\n"
        "- no_action: robot should stay passive.\n\n"
        "Preference signals and their meaning:\n"
        f"{signal_block}\n\n"
        "Return exactly one JSON object with keys:\n"
        "- preference_snapshot: list of {signal_name, polarity, weight}; polarity is "
        "prefer or avoid, weight is an integer 1-10.\n"
        "- Copy signal_name values exactly from the list above.\n"
        "- label_action: one allowed label.\n"
        "- rationale: short explanation grounded in the profile.\n\n"
        "COOPERA HUMAN PROFILE CONTEXT:\n"
        f"{json.dumps(human_profile_context, ensure_ascii=False, indent=2)}\n\n"
        "SITUATION AND ROBOT ACTION CANDIDATE:\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = _strip_json_fence(stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    candidate = _extract_balanced_json_object(stripped)
    if candidate is None:
        raise ValueError(f"No JSON object found in model response: {text}")
    candidate = _strip_json_fence(candidate.strip())
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    try:
        parsed = ast.literal_eval(candidate)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    return json.loads(candidate)


def _strip_json_fence(text: str) -> str:
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else text


def _extract_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _validate_preference_snapshot(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Qwen response preference_snapshot must be a list.")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        signal = str(item.get("signal_name", "")).strip().lower()
        polarity = str(item.get("polarity", "prefer")).strip().lower()
        if signal not in PREFERENCE_SIGNALS:
            raise ValueError(f"Qwen returned unknown preference signal: {signal!r}")
        if polarity not in {"prefer", "avoid"}:
            raise ValueError(f"Qwen returned invalid polarity for {signal!r}: {polarity!r}")
        if signal in seen:
            continue
        seen.add(signal)
        out.append(
            {
                "signal_name": signal,
                "polarity": polarity,
                "weight": _coerce_weight(item.get("weight")),
            }
        )
    return out


def _coerce_weight(value: Any) -> float:
    if value is None or value == "":
        return DEFAULT_PREFERENCE_WEIGHT
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return DEFAULT_PREFERENCE_WEIGHT
    return max(1.0, min(10.0, weight))
