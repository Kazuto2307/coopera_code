from __future__ import annotations

import gc
import json
import os
import re
from typing import Any

from preference_taxonomy import PREFERENCE_SIGNALS, VALID_LABELS


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
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
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

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
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
        return payload, {"raw_response": generated, "parsed": payload}


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
    return (
        "You are generating synthetic training supervision for an assistive-robot "
        "personalization model.\n\n"
        "IMPORTANT: Do NOT infer personal preferences merely from the current task "
        "or time. For example, a breakfast task at 9 am does NOT imply the person "
        "prefers morning tasks. Preferences must be grounded in the provided "
        "COOPERA human profile, Big Five/personality summary, and the profile's "
        "behavioral rationale. If the profile does not support a preference, omit it.\n\n"
        "Your job:\n"
        "1. Select zero or more preference signals that this simulated human would "
        "likely have, grounded in the profile.\n"
        "2. Choose what the assistive robot should do in this situation for this "
        "specific human.\n\n"
        "Allowed label_action values: do_now, do_later, remind, no_action.\n"
        "Use these meanings:\n"
        "- do_now: robot should execute/help now.\n"
        "- do_later: robot should postpone the action.\n"
        "- remind: robot should remind or notify, not execute directly.\n"
        "- no_action: robot should stay passive.\n\n"
        "Allowed preference signal_name values:\n"
        f"{json.dumps(allowed_signals, ensure_ascii=False)}\n\n"
        "Return exactly one JSON object with keys:\n"
        "- preference_snapshot: list of {signal_name, polarity}; polarity is prefer or avoid.\n"
        "- Copy signal_name values exactly from the allowed list. Do not create variants like prefer_prefer_*.\n"
        "- label_action: one allowed label.\n"
        "- rationale: short explanation grounded in the profile.\n\n"
        "COOPERA HUMAN PROFILE CONTEXT:\n"
        f"{json.dumps(human_profile_context, ensure_ascii=False, indent=2)}\n\n"
        "SITUATION AND ROBOT ACTION CANDIDATE:\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response: {text}")
    return json.loads(match.group(0))


def _validate_preference_snapshot(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("Qwen response preference_snapshot must be a list.")
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        signal = _canonical_signal_name(str(item.get("signal_name", "")).strip())
        polarity = str(item.get("polarity", "prefer")).strip().lower()
        if signal not in PREFERENCE_SIGNALS:
            raise ValueError(f"Qwen returned unknown preference signal: {signal!r}")
        if polarity not in {"prefer", "avoid"}:
            raise ValueError(f"Qwen returned invalid polarity for {signal!r}: {polarity!r}")
        if signal.startswith("avoid_") and polarity == "avoid":
            polarity = "prefer"
        if signal in seen:
            continue
        seen.add(signal)
        out.append({"signal_name": signal, "polarity": polarity})
    return out


def _canonical_signal_name(signal: str) -> str:
    candidates = [signal]
    if signal.startswith("prefer_prefer_"):
        candidates.append(signal.replace("prefer_prefer_", "prefer_", 1))
    if signal.startswith("avoid_avoid_"):
        candidates.append(signal.replace("avoid_avoid_", "avoid_", 1))
    if signal.startswith("prefer_avoid_"):
        candidates.append(signal.replace("prefer_avoid_", "avoid_", 1))
    if signal.startswith("avoid_prefer_"):
        candidates.append(signal.replace("avoid_prefer_", "prefer_", 1))
    for candidate in candidates:
        if candidate in PREFERENCE_SIGNALS:
            return candidate
    return signal
