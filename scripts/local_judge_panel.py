"""Simple console panel for the local judge run (stage judge-local, all cases in sequence).

Read only: reads data/private/progress/case<X>-2a-judge-local.json and `ollama ps`; closing the window stops nothing.
usage: python scripts/local_judge_panel.py [seconds]
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / "data" / "private" / "progress"
ORDER = ["E", "D", "A", "B", "C", "F", "G"]
# steps per case after the 120-person sample of A and B (D41); the snapshot total wins once a case has started
STEPS = {"E": 1046, "D": 260, "A": 155, "B": 443, "C": 343, "F": 61, "G": 178}
every = int(sys.argv[1]) if len(sys.argv) > 1 else 10
history = []
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def bar(f, n=40):
    f = max(0.0, min(1.0, f))
    return "█" * int(f * n) + "░" * (n - int(f * n))


def snapshot(case):
    p = PROGRESS / f"case{case}-2a-judge-local.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ollama_line():
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return "ollama is not responding"
    rows = [r for r in out[1:] if r.strip()]
    if not rows:
        return "no model loaded"
    parts = rows[0].split()
    # NAME ID SIZE(2 tokens) PROCESSOR(2 tokens) CONTEXT ...
    return f"{parts[0]}  {parts[4]} {parts[5]}  context {parts[6]}" if len(parts) > 6 else rows[0]


def age(ts):
    s = time.time() - ts
    return f"{s:.0f} s ago" if s < 120 else f"{s / 60:.0f} min ago"


while True:
    now = time.time()
    snaps = {c: snapshot(c) for c in ORDER}
    done_tot = steps_tot = failed = 0
    current = None
    lines = []
    for c in ORDER:
        s = snaps[c]
        total = s["total"] if s else STEPS[c]
        done = s["done"] if s else 0
        if s and s["status"] == "running" and now - s["updated"] < 600:
            state, current = "RUNNING", s
        elif s and done >= total:
            state = "done"
        elif s:
            state = "stopped"
        else:
            state = "to do"
        if s:
            failed += s.get("client", {}).get("failed", 0)
        done_tot += done
        steps_tot += total
        lines.append(f" Case {c}   {bar(done / max(1, total))} {100 * done / max(1, total):5.1f}%  {done:>5,} / {total:<5,}   {state}")
    history = [x for x in history + [(now, done_tot)] if now - x[0] <= 3600]
    if len(history) > 1 and history[-1][0] - history[0][0] >= 600:
        per_h = (history[-1][1] - history[0][1]) * 3600 / (history[-1][0] - history[0][0])
        src = "last hour" if history[-1][0] - history[0][0] >= 3300 else "since the panel was opened"
    elif current:
        per_h, src = current.get("rate_per_min", 0) * 60, "average of the running case"
    else:
        per_h, src = 0, ""
    os.system("cls")
    print(f" LOCAL JUDGE (qwen3.5:9b, no reasoning)      {time.strftime('%d/%m %H:%M:%S')}")
    print(f" refreshes every {every} s; closing the window stops nothing\n")
    print("\n".join(lines))
    print(f"\n Total    {bar(done_tot / max(1, steps_tot))} {100 * done_tot / max(1, steps_tot):5.1f}%  {done_tot:>5,} / {steps_tot:,} step\n")
    if current:
        last = current.get("last_advance") or current["updated"]
        print(f" Last step completed: {age(last)}   snapshot updated {age(current['updated'])}")
    else:
        print(" No case running")
    print(f" Failed calls: {failed}")
    print(f" Ollama: {ollama_line()}")
    if per_h > 0:
        rest_h = (steps_tot - done_tot) / per_h
        print(f" Rate: {per_h:.0f} steps/hour ({src})")
        end = time.localtime(now + rest_h * 3600)
        print(f" Expected end (approximate): {DAYS[end.tm_wday]} {time.strftime('%d/%m %H:%M', end)}"
              f"  (~{rest_h:.0f} hours; long texts slow it down)")
    print("\n Next: cloud judges when the quota is back (26-27/10)")
    time.sleep(every)
