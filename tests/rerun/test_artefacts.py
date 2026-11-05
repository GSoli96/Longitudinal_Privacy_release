from rerun import artefacts


def test_placeholders_are_removed_and_counted():
    t, c = artefacts.clean_text("Gentile [NOME], apra {{link}} e <qui> ______ fine")
    assert "[" not in t and "{" not in t and "<" not in t and "___" not in t
    assert c["placeholder"] == 4


def test_organisation_matches_whole_words_only():
    t, c = artefacts.clean_text("Presente nell'ente: il portale Ente e ente.it sono qui", organisation=["Ente"],
                                domains=["ente.it"])
    assert "Presente" in t and "portale" in t and c["organisation"] == 3
    assert "ente.it" not in t.lower().replace("nell'ente", "")


def test_no_organisation_given_removes_nothing_of_that_kind():
    t, c = artefacts.clean_text("Ente e ente", organisation=[], domains=[""])
    assert t == "Ente e ente" and c["organisation"] == 0


def test_urgency_formulae_are_removed():
    t, c = artefacts.clean_text("Serve una risposta urgente, entro oggi e subito. Scadenza vicina. ASAP")
    for w in ("urgent", "entro oggi", "subito", "scadenza", "asap"):
        assert w not in t.lower()
    assert c["urgency"] == 5


def test_signature_block_is_cut_from_the_last_closing():
    body = "Buongiorno,\nvi scrivo per il progetto.\n\nCordiali saluti\nUfficio Personale\ntel 123"
    t, c = artefacts.clean_text(body)
    assert t == "Buongiorno,\nvi scrivo per il progetto." and c["signature"] == 1


def test_text_without_closing_keeps_everything():
    t, c = artefacts.clean_text("Solo una frase.")
    assert t == "Solo una frase." and c["signature"] == 0


def test_clean_record_does_not_mutate_and_reports_counts():
    rec = {"id": "a", "y": 1, "subject": "Urgente [X]", "body": "Testo\n\nCordiali saluti\nMario"}
    out = artefacts.clean_record(rec)
    assert rec["subject"] == "Urgente [X]" and out["subject"] == ""
    assert out["body"] == "Testo"
    assert out["cleaned_counts"] == {"placeholder": 1, "organisation": 0, "urgency": 1, "signature": 1}


def test_clean_records_summary_by_class_and_empty():
    recs = [{"y": 1, "subject": "s", "body": "Urgente subito"}, {"y": 0, "subject": "s", "body": "Tutto ok"}]
    cleaned, s = artefacts.clean_records(recs)
    assert s["n_records"] == 2 and s["n_empty"] == 1
    assert s["by_class"]["1"]["urgency"] == 2 and s["by_class"]["0"]["urgency"] == 0
    assert cleaned[1]["body"] == "Tutto ok"
