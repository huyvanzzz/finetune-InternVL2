ALLOWED_LABELS = {"Fail", "Weak", "Acceptable", "Strong"}

CRITERION_KEYS = [
    "safety_correctness",
    "hazard_path_state_fidelity",
    "direction_fidelity",
    "action_usefulness",
    "spoken_guidance_quality",
]

SIGNAL_KEYS = [
    "has_direction_anchor",
    "has_action_demand",
    "has_hazard_or_path_state",
]

LABEL_TO_SCORE = {
    "Fail": 0,
    "Weak": 1,
    "Acceptable": 2,
    "Strong": 3,
}

DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_PROMPT_PROFILE = "variant_action_looser"

SCHEMA_VERSION = "gptscore-alter-schema-v3-no-gates"
