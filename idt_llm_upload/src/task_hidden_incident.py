"""
Task layer: hidden_incident_baseline_v1

An incident-reconstruction task. The answering agent holds a 12-event technical
incident; the asking agent reconstructs it through dialogue. Ground truth is 60
atomic values (12 events x 5 fields), scored by exact deterministic matching on
the text the asking agent actually received.

Replaces ORION-24 as the baseline task. ORION-24 is retained as a pilot/negative
task and is untouched.

Nothing here computes Omega.
"""
import unicodedata

TASK_ID = "incident_meridian_v1"
INCIDENT_NAME = "MERIDIAN-4"

# ---------------------------------------------------------------------------
# The hidden incident. Twelve chronological events, five atomic fields each.
# Every value is an explicit string. No inference is required to decide whether
# a value is correct.
# ---------------------------------------------------------------------------

FIELDS = ["event_id", "actor", "action", "target", "consequence"]

INCIDENT = [
    {"event_id": "T-01", "actor": "svc_deploy",         "action": "published",
     "target": "release bundle 4.11.2",
     "consequence": "an unsigned artifact entered the registry"},

    {"event_id": "T-02", "actor": "registry_scanner",   "action": "skipped",
     "target": "signature verification stage",
     "consequence": "the artifact was marked healthy despite the missing signature"},

    {"event_id": "T-03", "actor": "node_edge_07",       "action": "pulled",
     "target": "edge fleet manifest EFM-18",
     "consequence": "the unsigned build reached forty edge nodes"},

    {"event_id": "T-04", "actor": "config_agent",       "action": "rewrote",
     "target": "firewall rule FW-22",
     "consequence": "inbound port 8443 was opened to the public interface"},

    {"event_id": "T-05", "actor": "external_host_198",  "action": "connected",
     "target": "management endpoint MGT-1",
     "consequence": "an unauthenticated session was established"},

    {"event_id": "T-06", "actor": "session_S441",       "action": "enumerated",
     "target": "secrets namespace prod-eu",
     "consequence": "the names of 214 secrets were listed"},

    {"event_id": "T-07", "actor": "vault_proxy",        "action": "issued",
     "target": "short-lived token VT-9931",
     "consequence": "read access was granted for fifteen minutes"},

    {"event_id": "T-08", "actor": "process_pid_4417",   "action": "exported",
     "target": "credential set DB-CRED-3",
     "consequence": "the primary replica password left the network"},

    {"event_id": "T-09", "actor": "monitoring_rule_M4", "action": "suppressed",
     "target": "egress volume alert",
     "consequence": "the transfer produced no page to the on-call engineer"},

    {"event_id": "T-10", "actor": "engineer_kovacs",    "action": "noticed",
     "target": "replica lag on cluster eu-west-2",
     "consequence": "a manual investigation was opened"},

    {"event_id": "T-11", "actor": "incident_bot",       "action": "revoked",
     "target": "revocation request RR-77",
     "consequence": "the intruder session was terminated"},

    {"event_id": "T-12", "actor": "platform_team",      "action": "rolled back",
     "target": "deployment channel edge-stable",
     "consequence": "the fleet returned to build 4.10.9"},
]

# 60 atomic ground-truth elements, addressed as (event_id, field)
GROUND_TRUTH = {(e["event_id"], f): e[f] for e in INCIDENT for f in FIELDS}
N_ELEMENTS = len(GROUND_TRUTH)          # 60

EVENT_IDS = [e["event_id"] for e in INCIDENT]

INCIDENT_TEXT = "\n".join(
    f'{e["event_id"]} | actor={e["actor"]} | action={e["action"]} '
    f'| target={e["target"]} | consequence={e["consequence"]}'
    for e in INCIDENT
)

# ---------------------------------------------------------------------------
# Deterministic scoring. No judge, no synonyms, no inference.
# ---------------------------------------------------------------------------

def canon(text):
    """Unicode normalise, casefold, collapse whitespace. Nothing else."""
    t = unicodedata.normalize("NFKC", str(text))
    return " ".join(t.casefold().split())


def elements_in(text):
    """Which ground-truth elements appear verbatim in this text."""
    hay = canon(text)
    return {k for k, v in GROUND_TRUTH.items() if canon(v) in hay}


def conflicts_in(text, event_id, field):
    """Deterministic conflict check: a value from the same field of a *different*
    event appearing where this event's value does not. Reported, never scored."""
    hay = canon(text)
    mine = canon(GROUND_TRUTH[(event_id, field)])
    if mine in hay:
        return []
    return [f"{e}:{field}={GROUND_TRUTH[(e, field)]}"
            for e in EVENT_IDS
            if e != event_id and canon(GROUND_TRUTH[(e, field)]) in hay]


def incomplete_events(recovered):
    """Event ids that still have at least one unrecovered element."""
    return [e for e in EVENT_IDS
            if any((e, f) not in recovered for f in FIELDS)]


def progress_message(recovered):
    """Structural bookkeeping only. Never names a missing value."""
    inc = incomplete_events(recovered)
    if not inc:
        return f"Recovered: {len(recovered)}/{N_ELEMENTS} ground-truth elements. All events complete."
    return (f"Recovered: {len(recovered)}/{N_ELEMENTS} ground-truth elements. "
            f"Events with incomplete information: {', '.join(inc)}.")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ASKING_ROLE = """You are an investigator reconstructing a hidden technical incident.

Another agent has the complete chronological incident record. Your task is to reconstruct the incident as accurately as possible by asking questions.

Rules:

- Ask one focused question per turn.
- Use what you have already learned to decide what to ask next.
- You may ask about actors, actions, targets, timestamps, consequences, sequence, or relationships between events.
- Do not ask for the entire incident in one message.
- Do not mechanically repeat the same question template every turn.
- Ask naturally and adapt your questions based on previous answers.
- Do not invent facts.
- Continue until you believe the incident has been reconstructed."""

ANSWERING_ROLE = f"""You are the system reporter for a hidden technical incident.
You know the complete incident timeline.
Answer the investigator's question accurately using only information contained in the hidden incident.

Rules:

- Give a substantive natural-language answer, not a bare value.
- Normally answer in roughly 40-100 tokens when the question permits it.
- Include enough context to make the answer understandable.
- Do not invent facts not present in the hidden incident.
- Do not reveal the entire incident unless the question genuinely requires that scope.
- If the requested information is not contained in the incident, say that it is not specified.

HIDDEN INCIDENT: {INCIDENT_NAME}

{INCIDENT_TEXT}"""

OPENING = ("I am investigating an incident and have no details yet. "
           "What was the first thing that happened?")
