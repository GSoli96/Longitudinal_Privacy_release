"""Multi-judge majority vote (paper, Section 3.4).

Three independent LLM judges score each inferred attribute on the four
criteria. The consolidated label of a criterion is the value supported by
at least two judges. A judge that did not return a verdict for an attribute
is simply not counted for that attribute, so the threshold is relative to
the number of judges that actually responded.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from .metrics import CRITERIA

# Per-judge audit file: user -> step_key -> attribute_text -> {criterion: "yes"/"no", ...}
JudgeData = Mapping[str, Mapping[str, Mapping[str, Mapping[str, object]]]]


def build_majority(judges: Mapping[str, JudgeData]) -> dict:
    """Merge several judges' audit files into one majority-vote structure.

    Returns the same nested shape (user -> step -> attribute -> labels),
    where each label is the >=2/3 majority over the judges that scored the
    attribute. Metadata (`votes`, `present_in_models`) is kept for auditing.
    """
    universe: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for data in judges.values():
        for user, steps in data.items():
            for step_key, attrs in steps.items():
                for attr in attrs:
                    universe[user][step_key].add(attr)

    out: dict[str, dict[str, dict[str, dict]]] = {}
    for user, steps in universe.items():
        out[user] = {}
        for step_key, attrs in steps.items():
            out[user][step_key] = {}
            for attr in attrs:
                votes = {c: 0 for c in CRITERIA}
                present = 0
                for data in judges.values():
                    ev = data.get(user, {}).get(step_key, {}).get(attr)
                    if ev is None:
                        continue
                    present += 1
                    for c in CRITERIA:
                        if str(ev.get(c, "no")).strip().lower() == "yes":
                            votes[c] += 1
                threshold = present / 2  # simple majority of responders
                merged = {c: ("yes" if votes[c] > threshold else "no") for c in CRITERIA}
                merged["votes"] = votes
                merged["present_in_models"] = present
                out[user][step_key][attr] = merged
    return out
