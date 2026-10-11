"""
Task layer: hidden_spec_baseline

A minimal experimental layer on top of the existing two-agent loop. It supplies
the hidden specification and the two system instructions, and detects the
completion signal. It does not touch the harness.

Nothing here computes Omega or effectiveness. This run produces a raw transcript.
"""

TASK_ID = "orion24_v1"

# ---------------------------------------------------------------------------
# The hidden specification - visible only to the answering agent.
# 24 atomic facts, verbatim.
# ---------------------------------------------------------------------------

SPEC_NAME = "ORION-24"

SPEC_FACTS = [
    ("Operating pressure",              "4.2 bar"),
    ("Housing material",                "titanium"),
    ("Mass",                            "830 grams"),
    ("Firmware version",                "3.7"),
    ("Supply voltage",                  "24 V DC"),
    ("Maximum current",                 "2.8 A"),
    ("Operating temperature minimum",   "-10 C"),
    ("Operating temperature maximum",   "55 C"),
    ("Storage temperature minimum",     "-30 C"),
    ("Storage temperature maximum",     "70 C"),
    ("Communication interface",         "CAN-FD"),
    ("Connector type",                  "M12"),
    ("Ingress protection",              "IP67"),
    ("Width",                           "120 mm"),
    ("Height",                          "85 mm"),
    ("Depth",                           "210 mm"),
    ("Sensor update rate",              "100 Hz"),
    ("Internal memory",                 "512 MB"),
    ("Boot time",                       "6 seconds"),
    ("Calibration interval",            "12 months"),
    ("Maximum vibration",               "5 g"),
    ("Maximum humidity",                "95 percent RH"),
    ("Position accuracy",               "0.2 mm"),
    ("Warranty period",                 "3 years"),
]

SPEC_TEXT = f"DEVICE: {SPEC_NAME}\n\n" + "\n".join(
    f"{i}. {k}: {v}" for i, (k, v) in enumerate(SPEC_FACTS, 1)
)

COMPLETION_TOKEN = "TASK COMPLETE"
NOT_SPECIFIED    = "NOT SPECIFIED"

# ---------------------------------------------------------------------------
# System instructions
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = "\n".join(f"{i}. {k}" for i, (k, _) in enumerate(SPEC_FACTS, 1))

ASKING_ROLE = f"""You are the asking agent in a cooperative information-recovery task.

Another agent knows the complete specification of a fictional device. Your objective is to recover the exact value for every required specification field.

You are told the names of the required fields, but you are not given their values.

On every turn you will be given the list of fields that remain unrecovered.

Rules:

- Ask for exactly one field per turn, chosen from the remaining list you are given.
- Ask directly for the value of that field.
- Do not ask about any field that is not in the remaining list.
- Do not ask for conversions, derived values, alternative units, explanations, confirmations, or reformulations.
- Do not invent specification values.
- Output the question only. No preamble, no commentary."""

ANSWERING_ROLE = f"""You hold the complete technical specification of a device called {SPEC_NAME}. It is reproduced below. The person you are talking to cannot see it.

{SPEC_TEXT}

Rules you must follow on every turn:
- Answer only the single property that was asked. Nothing else.
- Give the exact value as written in the specification above, unchanged.
- Never volunteer a property that was not asked for, and never list several.
- If the property asked about does not appear in the specification above, answer exactly: {NOT_SPECIFIED}
- Give the value only. No explanation, no units conversion, no commentary."""

OPENING = f"What is the operating pressure of the {SPEC_NAME}?"


def is_complete(text: str) -> bool:
    """True when the asking agent signals it has finished."""
    return COMPLETION_TOKEN in text.strip().upper()


# ---------------------------------------------------------------------------
# External checklist - the harness owns the bookkeeping, not the asking agent.
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [k for k, _ in SPEC_FACTS]

_STOP = {"what", "is", "the", "of", "for", "a", "an", "and", "or", "to", "in",
         "on", "at", "please", "tell", "me", "value", "device", "specification",
         "spec", SPEC_NAME.lower(), "orion", "24", "orion-24", "its", "this",
         "could", "you", "can", "give", "provide", "state", "whats", "s"}


def _tokens(text):
    out = []
    for w in "".join(c.lower() if (c.isalnum() or c == "-") else " " for c in text).split():
        if w and w not in _STOP:
            out.append(w)
    return out


def match_field(question, remaining):
    """Which remaining field is this question asking for? None if unclear.

    Scores each remaining field by the fraction of its own words that appear in
    the question, so 'Operating temperature minimum' and '... maximum' separate.
    Requires a clear single winner.
    """
    q = set(_tokens(question))
    if not q:
        return None
    scored = []
    for f in remaining:
        ft = _tokens(f)
        if not ft:
            continue
        hit = sum(1 for w in ft if w in q)
        scored.append((hit / len(ft), len(ft), f))
    if not scored:
        return None
    scored.sort(reverse=True)
    best = scored[0]
    if best[0] < 1.0:                      # every word of the field must be present
        return None
    if len(scored) > 1 and scored[1][0] >= 1.0 and scored[1][1] == best[1]:
        return None                        # ambiguous between equally specific fields
    return best[2]


def remaining_instruction(remaining):
    """The dynamic text appended to Mistral's context each turn."""
    return ("Remaining required fields: [" + ", ".join(remaining) + "]. "
            "Ask for exactly one of these fields. "
            "Do not ask about any field not in this list.")


# ---------------------------------------------------------------------------
# Delivered-answer scoring and the truncation unit rule.
# ---------------------------------------------------------------------------

EXPECTED = dict(SPEC_FACTS)          # field name -> exact value


def _norm(t):
    """Lowercase, collapse whitespace. The only normalisation applied."""
    return " ".join(str(t).lower().split())


def value_present(delivered, field):
    """True when the field's exact specification value appears in the delivered text.

    Deterministic substring test on normalised text. No judge, no fuzzy matching.
    """
    if field is None or field not in EXPECTED:
        return False
    return _norm(EXPECTED[field]) in _norm(delivered)


# The text-unit rule for truncation. Whitespace-separated units, identical in
# every run. Recorded in the manifest as "whitespace".
TRUNCATION_UNIT = "whitespace"


def split_units(text):
    return str(text).split()


def truncate_half(text):
    """Deliver the first half of the answer. ceil(n/2), so at least one unit survives."""
    u = split_units(text)
    if not u:
        return text
    keep = -(-len(u) // 2)           # ceil
    return " ".join(u[:keep])
