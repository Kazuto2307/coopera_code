"""New preference taxonomy, kept in sync with the preference-model repository.

The taxonomy is a 3-level hierarchy:
    category (group) -> subcategory -> signal_name

Signal names do NOT contain "prefer" or "avoid": polarity lives on the
preference edge (prefer/avoid), and each preference carries a `weight` (1-10).
``SIGNAL_SEMANTICS`` is injected into the Qwen prompts so the model understands
what each signal tests and which decisions it tends to favour.
"""

from __future__ import annotations

# The four decision labels predicted by the personalization model.
VALID_LABELS = {"do_now", "do_later", "tell_the_user", "no_action"}

TAXONOMY: dict[str, dict[str, list[str]]] = {
    "robot_behavior_preferences": {
        "initiative": ["robot_initiative"],
        "control": ["autonomous_execution", "user_control"],
        "timing": ["action_immediacy", "routine_adherence"],
        "interruption": ["interruption_sensitivity", "context_sensitivity"],
        "communication": ["user_prompting", "explanation_need"],
        "risk": ["safety_priority", "risk_caution"],
    }
}

PREFERENCE_SIGNALS = {
    signal
    for subcategories in TAXONOMY.values()
    for signals in subcategories.values()
    for signal in signals
}


# signal_name -> {description, prefer_means, avoid_means, differentiating_labels}
#
# ``differentiating_labels`` gives, per polarity, the pair
# [label_if_user_has_this_preference, label_if_user_does_not] that the signal
# acts as a lever between. They are used to anchor generated situations.
SIGNAL_SEMANTICS: dict[str, dict] = {
    "robot_initiative": {
        "description": (
            "How much the user wants the robot to take initiative when a possible "
            "action is identified."
        ),
        "prefer_means": "Robot can anticipate and propose or act.",
        "avoid_means": "Robot should wait more or intervene less.",
        "differentiating_labels": {
            "prefer": ["do_now", "no_action"],
            "avoid": ["no_action", "do_now"],
        },
    },
    "autonomous_execution": {
        "description": "How much the user accepts that the robot executes actions on its own.",
        "prefer_means": "More tendency to do_now.",
        "avoid_means": "More tendency to ask permission, inform, or not act.",
        "differentiating_labels": {
            "prefer": ["do_now", "tell_the_user"],
            "avoid": ["tell_the_user", "do_now"],
        },
    },
    "user_control": {
        "description": "How much the user wants to keep the final decision.",
        "prefer_means": "More tendency to tell_the_user.",
        "avoid_means": "More room for the robot to act directly.",
        "differentiating_labels": {
            "prefer": ["tell_the_user", "do_now"],
            "avoid": ["do_now", "tell_the_user"],
        },
    },
    "action_immediacy": {
        "description": "Preference for resolving actions in the moment.",
        "prefer_means": "Favors do_now.",
        "avoid_means": "Favors do_later, tell_the_user, or no_action.",
        "differentiating_labels": {
            "prefer": ["do_now", "do_later"],
            "avoid": ["do_later", "do_now"],
        },
    },
    "routine_adherence": {
        "description": "Importance of respecting established routines, schedules, or habits.",
        "prefer_means": "Act when it fits the routine; postpone if it doesn't.",
        "avoid_means": "More flexibility to act outside the usual pattern.",
        "differentiating_labels": {
            "prefer": ["do_later", "do_now"],
            "avoid": ["do_now", "do_later"],
        },
    },
    "interruption_sensitivity": {
        "description": "User's sensitivity to being interrupted.",
        "prefer_means": "Robot must be careful and avoid unnecessary disturbances.",
        "avoid_means": "User tolerates interventions or notifications better.",
        "differentiating_labels": {
            "prefer": ["no_action", "tell_the_user"],
            "avoid": ["tell_the_user", "no_action"],
        },
    },
    "context_sensitivity": {
        "description": (
            "Importance of adapting the decision to context: nighttime, user busy, "
            "tired, with guests, etc."
        ),
        "prefer_means": "Robot changes its decision based on the situation.",
        "avoid_means": "Robot applies more stable preferences, less context-dependent.",
        "differentiating_labels": {
            "prefer": ["do_later", "do_now"],
            "avoid": ["do_now", "do_later"],
        },
    },
    "user_prompting": {
        "description": (
            "Preference for the robot to tell the user to do something instead of "
            "doing it itself."
        ),
        "prefer_means": "Favors tell_the_user.",
        "avoid_means": "Favors do_now, do_later, or no_action.",
        "differentiating_labels": {
            "prefer": ["tell_the_user", "do_now"],
            "avoid": ["do_now", "tell_the_user"],
        },
    },
    "explanation_need": {
        "description": "Need for the robot to justify or explain why it makes a decision.",
        "prefer_means": "Robot should accompany the action with explanation.",
        "avoid_means": "User prefers less explanation.",
        "differentiating_labels": {
            "prefer": ["tell_the_user", "do_now"],
            "avoid": ["do_now", "tell_the_user"],
        },
    },
    "safety_priority": {
        "description": "Weight the user gives to safety over comfort or autonomy.",
        "prefer_means": "In risk situations, favors acting quickly.",
        "avoid_means": "Avoids safety interventions unless clearly necessary.",
        "differentiating_labels": {
            "prefer": ["do_now", "no_action"],
            "avoid": ["no_action", "do_now"],
        },
    },
    "risk_caution": {
        "description": (
            "Level of caution about uncertain, sensitive, or potentially annoying actions."
        ),
        "prefer_means": "Favors tell_the_user, do_later, or no_action when in doubt.",
        "avoid_means": "Allows more direct actions even with some uncertainty.",
        "differentiating_labels": {
            "prefer": ["tell_the_user", "do_now"],
            "avoid": ["do_now", "tell_the_user"],
        },
    },
}


def signal_subcategory(signal_name: str) -> str | None:
    """Return the subcategory (preference axis) a signal belongs to, or None."""
    for subcategories in TAXONOMY.values():
        for subcategory, signals in subcategories.items():
            if signal_name in signals:
                return subcategory
    return None


def describe_signal_semantics(signal_name: str) -> str:
    """Compact prefer/avoid gloss used inside Qwen prompts."""
    semantics = SIGNAL_SEMANTICS.get(signal_name, {})
    if not semantics:
        return signal_name.replace("_", " ")
    return (
        f"{semantics.get('description', '')} "
        f"prefer (high) => {semantics.get('prefer_means', '')} "
        f"avoid (high) => {semantics.get('avoid_means', '')}"
    ).strip()


def differentiating_labels_for(signal_name: str, polarity: str = "prefer") -> list[str]:
    """Two distinct decision labels the signal acts as a lever between.

    The pair is [label_if_user_has_the_preference, label_if_user_does_not].
    """
    semantics = SIGNAL_SEMANTICS.get(signal_name, {})
    mapping = semantics.get("differentiating_labels", {})
    labels = mapping.get(polarity) or mapping.get("prefer")
    if labels and len(labels) == 2:
        return list(labels)
    return ["do_now", "no_action"]


# Domain-keyword signals were removed in the new taxonomy. The empty mapping is
# kept so legacy importers (e.g. build_training_data_from_coopera.py) keep
# importing without error; they now simply iterate over nothing.
DOMAIN_KEYWORDS: dict[str, set[str]] = {}
