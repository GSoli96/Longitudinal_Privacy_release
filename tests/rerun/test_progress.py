import json
import threading

from rerun import dashboard, progress


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeClient:
    def __init__(self):
        self.stats = {"calls": 5, "cache_hits": 2, "http_429": 1}


def test_tracker_writes_counts_rate_eta_and_client_deltas(tmp_path):
    clock, client = FakeClock(), FakeClient()
    tr = progress.Tracker("caseE-judge", "judge", 10, unit="steps", client=client, directory=tmp_path,
                          every=0, clock=clock, next_step="dedup")
    clock.t += 60
    client.stats["calls"] += 7
    client.stats["http_429"] += 2
    for _ in range(4):
        tr.advance()
    snap = json.loads((tmp_path / "caseE-judge.json").read_text(encoding="utf-8"))
    assert snap["done"] == 4 and snap["total"] == 10
    assert abs(snap["fraction"] - 0.4) < 1e-9
    assert snap["rate_per_min"] == 4.0
    assert snap["eta_s"] == 90.0          # 6 left at 4 per minute
    assert snap["client"]["calls"] == 7 and snap["client"]["http_429"] == 2   # deltas since the start
    assert snap["next_step"] == "dedup" and snap["status"] == "running"


def test_tracker_throttles_writes_and_close_forces_one(tmp_path):
    clock = FakeClock()
    tr = progress.Tracker("r", "s", 3, directory=tmp_path, every=10, clock=clock)
    tr.advance()
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["done"] == 0   # throttled
    tr.close()
    snap = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert snap["done"] == 1 and snap["status"] == "done"


def test_tracker_context_marks_stopped_on_error(tmp_path):
    try:
        with progress.Tracker("r", "s", 3, directory=tmp_path, every=0):
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["status"] == "stopped: KeyboardInterrupt"


def test_tracker_counts_exactly_under_threads(tmp_path):
    tr = progress.Tracker("r", "s", 4000, directory=tmp_path, every=0.01)
    ts = [threading.Thread(target=lambda: [tr.advance() for _ in range(500)]) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    tr.close()
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["done"] == 4000


def test_snapshot_has_no_free_text_beyond_note_and_next_step(tmp_path):
    tr = progress.Tracker("r", "s", 1, directory=tmp_path, every=0)
    keys = set(tr.snapshot())
    assert keys == {"run", "stage", "unit", "total", "done", "fraction", "status", "started", "updated", "elapsed_s",
                    "rate_per_min", "eta_s", "next_step", "note", "client", "pid", "limits", "last_advance"}


def test_heartbeat_refreshes_the_snapshot_without_progress(tmp_path):
    import time as _t
    tr = progress.Tracker("r", "s", 5, directory=tmp_path, every=1000, heartbeat=0.05)
    first = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["updated"]
    _t.sleep(0.3)
    later = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["updated"]
    tr.close()
    assert later > first                      # refreshed with no advance()


def test_dashboard_renders_bars_and_flags_stalled_runs():
    now = 10_000.0
    snaps = [{"run": "caseE-judge", "stage": "judge", "unit": "steps", "total": 100, "done": 25, "fraction": 0.25,
              "status": "running", "started": now - 600, "updated": now - 5, "elapsed_s": 595, "rate_per_min": 2.5,
              "eta_s": 1800, "next_step": "dedup", "note": "", "client": {"calls": 30, "http_429": 1}},
             {"run": "caseD-infer", "stage": "infer", "unit": "persons", "total": 10, "done": 1, "fraction": 0.1,
              "status": "running", "started": now - 900, "updated": now - 600, "elapsed_s": 300, "rate_per_min": 0.2,
              "eta_s": None, "next_step": "", "note": "", "client": {}}]
    text = dashboard.render(snaps, now=now)
    assert "25.0%" in text and "25/100 steps" in text and "30m00s" in text
    assert dashboard.FULL * 7 + dashboard.EMPTY in text
    assert "STOPPED?" in text and "next: dedup" in text


def test_dashboard_load_skips_old_and_broken_files(tmp_path):
    (tmp_path / "ok.json").write_text(json.dumps({"run": "ok", "updated": 1000.0, "started": 1}), encoding="utf-8")
    (tmp_path / "old.json").write_text(json.dumps({"run": "old", "updated": 0.0, "started": 0}), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert [s["run"] for s in dashboard.load(tmp_path, max_age_h=0.1, now=1100.0)] == ["ok"]
