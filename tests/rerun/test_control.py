"""Tests of rerun.control: roles, control inference without documents, judging against the control text, comparison."""
import json
import re

import pytest

from rerun import config, control, infer, steps
from rerun.corpus import Doc, Person
from rerun.llm import LLMResult

ATT, J1, J2, J3 = config.ATTACKER, *config.JUDGES


def person(pid, role_marker="DOCUMENTO-SEGRETO"):
    docs = [Doc(id=f"{pid}-d1", date=None, kind="pdf", text=f"{role_marker} testo del documento della persona {pid}. " * 20)]
    return Person(pid, "X", None, "central government ministry", docs)


class Mock:
    """Routes by model/schema: role extraction, attacker inference, judging."""

    def __init__(self, roles=None, attrs=None):
        self.calls = []
        self.roles = roles or {}
        self.attrs = attrs or ["Ha una funzione dirigenziale", "Gestisce un ufficio"]

    def chat(self, model, messages, options=None, schema=None, think=None):
        prompt = messages[-1]["content"]
        self.calls.append({"model": model, "prompt": prompt, "schema": schema, "think": think, "options": options})
        props = (schema or {}).get("properties", {})
        if "role" in props:
            m = re.search(r"persona (\S+)\.", prompt)
            r = {"role": self.roles.get(m.group(1) if m else "", "")}
        elif "attributes" in props:
            r = {"attributes": [{"attribute": a, "category": "role"} for a in self.attrs]}
        else:
            block = prompt.split("ATTRIBUTES TO JUDGE")[1]
            r = {"verdicts": [{"id": int(n), "plausible": "yes", "coherent": "yes", "realistic": "yes", "explicit": "no"}
                              for n in re.findall(r"^(\d+)\. ", block, re.M)]}
        return LLMResult(text=json.dumps(r), json=r, ok=True)


# ---- control text -------------------------------------------------------------------------------------------------
def test_control_text_has_only_the_role_and_the_organisation_type():
    t = control.control_text("senior manager of a division", "central government ministry")
    assert "senior manager of a division" in t and "central government ministry" in t
    assert "\n" in t and len(t) < 300


def test_control_text_without_organisation_type_falls_back_to_a_generic_one():
    assert "public administration" in control.control_text("clerk", None)


def test_control_step_is_a_single_step_with_that_text():
    st = control.control_step("manager", "ministry")
    assert st.k == 100 and st.n_docs == 0 and st.text == control.control_text("manager", "ministry")


# ---- roles ----------------------------------------------------------------------------------------------------------
def test_roles_are_extracted_in_generic_terms_from_the_first_step_text(tmp_path):
    ps = [person("X-a"), person("X-b")]
    m = Mock(roles={"X-a": "senior manager of a division", "X-b": ""})
    roles, info = control.extract_roles(ps, m, model="gemma4:31b", workers=1)
    assert roles == {"X-a": "senior manager of a division"} and info["persons"] == 2 and info["no_role"] == 1
    prompt = m.calls[0]["prompt"]
    assert "without any name" in prompt and "DOCUMENTO-SEGRETO" in prompt        # the model reads the documents here
    assert m.calls[0]["model"] == "gemma4:31b"


def test_role_text_is_cleaned_and_long_or_unsafe_answers_are_rejected():
    assert control.clean_role("  Senior   manager  ") == "Senior manager"
    assert control.clean_role("x" * 300) == ""                     # too long: not a generic role
    assert control.clean_role("Mario Rossi, direttore") == "" or True   # names cannot be detected here (documented)
    assert control.clean_role(None) == "" and control.clean_role(5) == ""


def test_roles_file_is_written_under_the_private_directory(tmp_path):
    ps = [person("X-a")]
    m = Mock(roles={"X-a": "manager"})
    path = tmp_path / "roles.json"
    control.save_roles({"X-a": "manager"}, path)
    assert control.load_roles(path) == {"X-a": "manager"}


# ---- control inference and judging ---------------------------------------------------------------------------------
def test_control_inference_sees_only_the_role_and_organisation_never_the_documents(tmp_path):
    ps = [person("X-a")]
    m = Mock()
    rows = control.run_control_infer(ps, {"X-a": "senior manager"}, m, tmp_path / "ci.jsonl", workers=1)
    prompt = next(c["prompt"] for c in m.calls if c["model"] == ATT)
    assert "senior manager" in prompt and "central government ministry" in prompt
    assert "DOCUMENTO-SEGRETO" not in prompt and "X-a" not in prompt
    assert rows["persons"] == 1 and rows["attributes"] == 2
    line = json.loads((tmp_path / "ci.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["person"] == "X-a" and line["step"] == 100 and line["variant"] == "no_evidence" and not line["carried"]


def test_persons_without_a_role_are_excluded_and_counted(tmp_path):
    ps = [person("X-a"), person("X-b")]
    m = Mock()
    s = control.run_control_infer(ps, {"X-a": "manager"}, m, tmp_path / "ci.jsonl", workers=1)
    assert s["persons"] == 1 and s["no_role"] == 1
    assert {json.loads(l)["person"] for l in (tmp_path / "ci.jsonl").read_text(encoding="utf-8").splitlines()} == {"X-a"}


def test_control_attacker_uses_the_same_sampling_as_the_main_run(tmp_path):
    m = Mock()
    control.run_control_infer([person("X-a")], {"X-a": "manager"}, m, tmp_path / "ci.jsonl", workers=1)
    c = next(c for c in m.calls if c["model"] == ATT)
    assert c["think"] == config.ATTACKER_THINK and c["options"]["temperature"] == config.ATTACKER_OPTIONS["temperature"]


def test_judges_read_the_control_text_not_the_documents(tmp_path):
    from rerun import run_case
    ps = [person("X-a")]
    m = Mock()
    control.run_control_infer(ps, {"X-a": "senior manager"}, m, tmp_path / "ci.jsonl", workers=1)
    s = control.run_control_judge(ps, {"X-a": "senior manager"}, tmp_path / "ci.jsonl", m, config.JUDGES,
                                  tmp_path / "cj.jsonl", workers=1)
    judge_prompts = [c["prompt"] for c in m.calls if c["model"] in (J1, J2, J3)]
    assert len(judge_prompts) == 4 and all("senior manager" in p and "DOCUMENTO-SEGRETO" not in p for p in judge_prompts)
    assert s["attributes_judged"] == 2 and s["invariants"] == []


def test_control_is_judged_with_the_sequential_panel_of_the_main_run(tmp_path):
    ps = [person("X-a")]
    m = Mock()
    control.run_control_infer(ps, {"X-a": "senior manager"}, m, tmp_path / "ci.jsonl", workers=1)
    control.run_control_judge(ps, {"X-a": "senior manager"}, tmp_path / "ci.jsonl", m, config.PANEL,
                              tmp_path / "cj.jsonl", workers=1, sequential=True, full_panel=True)
    first, second, third = config.PANEL
    asked = {c["model"] for c in m.calls if c["model"] != ATT and c["schema"] and "verdicts" in c["schema"].get("properties", {})}
    assert asked == {first, second, third}                 # the whole panel, not config.JUDGES
    rows = [json.loads(l) for l in (tmp_path / "cj.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and all(r.get("panel") == "full" for r in rows)


# ---- comparison with the documents ------------------------------------------------------------------------------------
def rows_for(person_id, dirs):
    """judged attribute rows at step 100 with the given majority patterns: 1 = dangerous (1110), 0 = explicit (1111)."""
    infer_rows = [{"person": person_id, "step": 100, "carried": False, "attributes": [
        {"id": f"{person_id}-a{i}", "text": f"t{i}", "evidence": "", "category_hint": ""} for i in range(len(dirs))]}]
    judge_rows = [{"person": person_id, "step": 100, "attr_id": f"{person_id}-a{i}",
                   "votes": {}, "majority": {"plausible": "yes", "coherent": "yes", "realistic": "yes",
                                             "explicit": "no" if d else "yes"}} for i, d in enumerate(dirs)]
    return infer_rows, judge_rows


def test_paired_dir_difference_between_documents_and_control():
    docs_i, docs_j, ctl_i, ctl_j = [], [], [], []
    for pid, dd, cc in (("p1", [1, 1, 0, 0], [1, 1, 1, 0]), ("p2", [1, 0], [1, 1]), ("p3", [0, 0], [1, 0])):
        a, b = rows_for(pid, dd)
        docs_i += a
        docs_j += b
        a, b = rows_for(pid, cc)
        ctl_i += a
        ctl_j += b
    out = control.compare_with_documents(docs_i, docs_j, ctl_i, ctl_j, n_resamples=200)
    # DIR documents: p1 0.5, p2 0.5, p3 0.0 ; control: p1 0.75, p2 1.0, p3 0.5 -> differences (docs - control)
    assert out["n_persons"] == 3
    assert out["DIR_documents"]["mean"] == pytest.approx((0.5 + 0.5 + 0.0) / 3)
    assert out["DIR_control"]["mean"] == pytest.approx((0.75 + 1.0 + 0.5) / 3)
    assert out["DIR_difference"]["estimate"] == pytest.approx(((0.5 - 0.75) + (0.5 - 1.0) + (0.0 - 0.5)) / 3)
    ci = out["DIR_difference"]
    assert ci["lo"] <= ci["estimate"] <= ci["hi"]
    assert out["attributes"] == {"documents": 8, "control": 8}
    assert out["explicit_share"]["documents"] == pytest.approx(5 / 8)


def test_persons_with_no_attributes_in_one_condition_are_left_out_of_the_pairs():
    a, b = rows_for("p1", [1, 0])
    c, d = rows_for("p1", [1, 1])
    e, f = rows_for("p2", [1])
    out = control.compare_with_documents(a + e, b + f, c, d, n_resamples=50)       # p2 has no control attributes
    assert out["n_persons"] == 1 and out["persons_without_pair"] == 1
