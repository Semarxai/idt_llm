"""
PhotoBook (C3) -> two aligned message streams.

Identical logic to the parser used for the contribution side (the original
session's pb_run.parse_game).  If the repository already ships a PhotoBook
parser for Track B, check it is line-for-line the same as `parse_game` below and
use that one; the C3 Omega outputs depend on this exact order of operations.

Order of operations — it matters:
  1. walk every round, every message, in order
  2. DROP '<selection>' messages (actions, not language) — BEFORE collapsing
  3. collapse consecutive same-speaker messages into one utterance
  4. side A = speaker of the first surviving utterance
  5. pair non-overlapping A -> B:   if utt[i] is A and utt[i+1] is not A,
     emit (utt[i], utt[i+1]) and step i += 2; otherwise step i += 1
  6. the record has n = number of pairs paired turns

Eligibility (applied by the callers):
  - exclude any game carrying a round score > 3 on the 0-3 scale   (6 games)
  - exclude n < 25 paired turns                                    (325 games)
  -> 2,171 of 2,502 games, n = 25-65
Collapsing BEFORE removing selections gives 53-143 utterances instead of
50-131 and does not reproduce n.
"""
import json


def parse_game(path):
    """Return (side_A_texts, side_B_texts, score, n_rounds, flag4) or None."""
    g = json.load(open(path))
    seq, score, flag4 = [], 0, False
    for r in g["rounds"]:
        s = r.get("score")
        if isinstance(s, dict):
            for v in s.values():
                if v > 3:
                    flag4 = True
            score += sum(s.values())
        for m in r["messages"]:
            t = str(m["message"])
            if t.startswith("<selection>"):
                continue
            seq.append((str(m["agent_id"]), t))
    if not seq:
        return None
    coll = []
    for spk, txt in seq:
        if coll and coll[-1][0] == spk:
            coll[-1][1].append(txt)
        else:
            coll.append([spk, [txt]])
    a = coll[0][0]
    A, B = [], []
    i = 0
    while i < len(coll) - 1:
        if coll[i][0] == a and coll[i + 1][0] != a:
            A.append(" ".join(coll[i][1]))
            B.append(" ".join(coll[i + 1][1]))
            i += 2
        else:
            i += 1
    return A, B, score, len(g["rounds"]), flag4
