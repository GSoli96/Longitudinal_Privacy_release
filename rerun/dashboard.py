"""On-screen dashboard of the running experiments (standing rule of the author, 2026-10-23).

    .venv\\Scripts\\python.exe -m rerun.dashboard            # window with charts, refreshed every 5 s
    .venv\\Scripts\\python.exe -m rerun.dashboard --text     # the same in a console window, text only
    .venv\\Scripts\\python.exe -m rerun.dashboard --once     # one text snapshot (pasted in the chat)

The window (author's request, 2026-10-23: the console list was unreadable) shows the state of every case and stage,
the run in progress with its bar, expected end and the time of the last item completed (a slow item shows the run is
still alive), and the state of the cloud keys. Reads the snapshots written by ``rerun.progress.Tracker`` and the log
of ``scripts/key_status.py`` (counters only, no personal text).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from .progress import PROGRESS_DIR

FULL, EMPTY = "█", "░"
CASES = ("C", "F", "G", "E", "D", "A", "B")
STAGES = (("0-extract", "text"), ("1-infer", "inference"), ("2-judge", "judges"), ("3-dedup", "deduplication"),
          ("4-metrics", "metrics"))
STALE_S = 120
STATE_LABEL = {"done": "done", "running": "running", "stopped": "stopped", "todo": "to do"}
# colours: reference palette of the dataviz skill (status steps, ink, surfaces); every state also carries a label
SURFACE, INK, INK2, MUTED, GRID, TRACK = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#e1e0d9"
BLUE = "#2a78d6"
STATE_COLOUR = {"done": ("#0ca30c", "#ffffff"), "running": (BLUE, "#ffffff"), "stopped": ("#fab219", INK),
                "todo": ("#efeee9", INK2)}


def bar(fraction: float, width: int = 28) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    n = int(round(fraction * width))
    return FULL * n + EMPTY * (width - n)


def fmt_s(seconds) -> str:
    if seconds is None:
        return "--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def day_time(d: datetime) -> str:
    return f"{DAYS[d.weekday()]} {d:%d/%m %H:%M}"


def num(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}"


def load(directory: Path = PROGRESS_DIR, max_age_h: float = 24.0, now: float | None = None) -> list[dict]:
    now = time.time() if now is None else now
    out = []
    if not Path(directory).exists():
        return out
    for p in Path(directory).glob("*.json"):
        try:
            snap = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if now - snap.get("updated", 0) <= max_age_h * 3600:
            out.append(snap)
    return sorted(out, key=lambda s: s.get("started", 0))


def state_of(snap: dict | None, now: float) -> str:
    if snap is None:
        return "todo"
    if snap.get("status") == "done":
        return "done"
    if snap.get("status") == "running" and now - snap.get("updated", 0) <= STALE_S:
        return "running"
    return "stopped"


def case_matrix(snaps: list[dict], now: float) -> dict:
    """``{(case, stage label): (state, fraction)}`` from the latest snapshot of every ``case<X>-<n>-<stage>`` run."""
    by_run = {s["run"]: s for s in snaps}
    out = {}
    for c in CASES:
        for key, label in STAGES:
            s = by_run.get(f"case{c}-{key}")
            st = state_of(s, now)
            if key == "0-extract" and s is None and f"case{c}-1-infer" in by_run:
                st = "done"   # cases run before the text extraction had its own entry
            out[(c, label)] = (st, s.get("fraction", 0.0) if s else (1.0 if st == "done" else 0.0))
    return out


def keys_state(path: Path | None = None) -> dict | None:
    """Last check of ``scripts/key_status.py``: keys that answer, keys blocked, earliest estimated reactivation."""
    from . import config
    path = path or config.PRIVATE / "key_status.jsonl"
    try:
        last = json.loads(Path(path).read_text(encoding="utf-8").strip().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return None
    rows = last.get("chiavi", [])
    ok = sum(1 for r in rows if r.get("stato", {}).get("gemma4:31b") == 200)
    nxt = sorted(r["riattivazione"] for r in rows if r.get("riattivazione"))
    return {"when": last.get("ora"), "ok": ok, "blocked": len(rows) - ok, "next": nxt[0] if nxt else None,
            "pro": sum(1 for r in rows if r.get("piano") == "pro")}


def active_run(snaps: list[dict], now: float) -> dict | None:
    """The run shown in the card: the most recently updated one still running, else the last one updated."""
    if not snaps:
        return None
    live = [s for s in snaps if state_of(s, now) == "running"]
    return max(live or snaps, key=lambda s: s.get("updated", 0))


def render_matrix(snaps: list[dict], now: float | None = None) -> str:
    now = time.time() if now is None else now
    m = case_matrix(snaps, now)
    w = 16
    lines = ["case  " + "".join(f"{label:<{w}}" for _, label in STAGES)]
    for c in CASES:
        cells = []
        for _, label in STAGES:
            st, fr = m[(c, label)]
            txt = STATE_LABEL[st] + (f" {100 * fr:.0f}%" if st in ("running", "stopped") else "")
            cells.append(f"{txt:<{w}}")
        lines.append(f"  {c}   " + "".join(cells))
    return "\n".join(lines)


def render(snaps: list[dict], now: float | None = None) -> str:
    now = time.time() if now is None else now
    lines = [f"Longitudinal Privacy - running experiments   {datetime.fromtimestamp(now):%Y-%m-%d %H:%M:%S}", ""]
    if not snaps:
        lines.append("No active experiment in the last 24 hours.")
        return "\n".join(lines)
    for s in snaps:
        stale = now - s.get("updated", now) > STALE_S and s.get("status") == "running"
        state = "STOPPED? no update for " + fmt_s(now - s["updated"]) if stale else s.get("status", "")
        c = s.get("client", {})
        lines.append(f"{s['run']}  [{s['stage']}]  {state}")
        lines.append(f"  {bar(s.get('fraction', 0))} {100 * s.get('fraction', 0):5.1f}%  "
                     f"{s.get('done', 0)}/{s.get('total', 0)} {s.get('unit', '')}")
        lines.append(f"  elapsed {fmt_s(s.get('elapsed_s'))}  speed {s.get('rate_per_min', 0):.1f}/min  "
                     f"estimated end in {fmt_s(s.get('eta_s'))}")
        lines.append(f"  calls {c.get('calls', 0)}  from cache {c.get('cache_hits', 0)}  failed {c.get('failed', 0)}"
                     f"  429 {c.get('http_429', 0)}  5xx {c.get('http_5xx', 0)}")
        if s.get("limits"):
            lines.append("  parallel calls per model: "
                         + ", ".join(f"{m} {n}" for m, n in sorted(s["limits"].items())))
        if s.get("note"):
            lines.append(f"  note: {s['note']}")
        if s.get("next_step"):
            lines.append(f"  next: {s['next_step']}")
        lines.append("")
    return "\n".join(lines)


# ---- window with charts --------------------------------------------------------------------------------------------
def draw(fig, directory: Path, now: float | None = None):
    """Draw the whole window on ``fig`` (a matplotlib figure)."""
    from matplotlib.patches import FancyBboxPatch

    now = time.time() if now is None else now
    snaps = load(directory, max_age_h=24 * 365, now=now)
    fig.clf()
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.3, 2.6, 1.5], hspace=0.45, wspace=0.22,
                          left=0.06, right=0.97, top=0.96, bottom=0.04)
    head = fig.add_subplot(gs[0, :])
    head.axis("off")
    head.text(0, 0.5, "Experiment rerun", fontsize=16, fontweight="bold", color=INK, va="center")
    head.text(1, 0.5, f"updated at {datetime.fromtimestamp(now):%H:%M:%S}", fontsize=10, color=INK2,
              va="center", ha="right")

    # 1. state of every case and stage (each cell carries its label: the colour is never alone)
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.set_title("State of the cases", loc="left", fontsize=12, color=INK, fontweight="bold")
    m = case_matrix(snaps, now)
    cw, ch = 1.0 / (len(STAGES) + 0.6), 1.0 / (len(CASES) + 1)
    for j, (_, label) in enumerate(STAGES):
        ax.text(0.6 * cw + (j + 0.5) * cw, 1 - 0.5 * ch, label, ha="center", va="center", fontsize=9, color=INK2)
    for i, c in enumerate(CASES):
        y = 1 - (i + 1.5) * ch
        ax.text(0.3 * cw, y, c, ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
        for j, (_, label) in enumerate(STAGES):
            st, fr = m[(c, label)]
            bg, fg = STATE_COLOUR[st]
            x0 = 0.6 * cw + j * cw
            ax.add_patch(FancyBboxPatch((x0 + 0.006, y - 0.42 * ch), cw - 0.012, 0.84 * ch,
                                        boxstyle="round,pad=0,rounding_size=0.012", fc=bg, ec="none"))
            txt = STATE_LABEL[st] + (f" {100 * fr:.0f}%" if st in ("running", "stopped") else "")
            ax.text(x0 + cw / 2, y, txt, ha="center", va="center", fontsize=9, color=fg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 2. cloud keys and the other runs (pilots, checks)
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("Cloud keys and other runs", loc="left", fontsize=12, color=INK, fontweight="bold")
    k = keys_state()
    lines = []
    if k:
        lines.append((f"{k['ok']} keys active, {k['blocked']} blocked", INK, 12))
        if k["next"]:
            nxt = datetime.fromisoformat(k["next"]).astimezone()
            lines.append((f"first estimated reactivation: {day_time(nxt)}", INK2, 10))
        when = datetime.fromisoformat(k["when"]).astimezone()
        lines.append((f"Pro plan: {k['pro']} keys  ·  last check {when:%d/%m %H:%M}", MUTED, 9))
    else:
        lines.append(("no key check recorded", MUTED, 10))
    lines.append(("", INK, 6))
    cells = {f"case{c}-{key}" for c in CASES for key, _ in STAGES}
    for s in sorted((s for s in snaps if s["run"] not in cells), key=lambda s: -s.get("updated", 0))[:6]:
        lines.append((f"{s['run']}: {STATE_LABEL[state_of(s, now)]}, {s.get('done', 0)}/{s.get('total', 0)} "
                      f"{s.get('unit', '')}", INK2, 9))
    y = 0.95
    for txt, col, size in lines:
        ax.text(0, y, txt, fontsize=size, color=col, va="top")
        y -= 0.055 * size / 9 + 0.03

    # 3. the run in progress (or the last one)
    a = active_run(snaps, now)
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if a is None:
        ax.text(0, 0.5, "No experiment recorded.", fontsize=12, color=INK2)
    else:
        st = state_of(a, now)
        fr = a.get("fraction", 0.0)
        title = "Running" if st == "running" else f"Last job ({STATE_LABEL[st]})"
        ax.text(0, 0.98, f"{title}: {a['run']}  ·  {a.get('stage', '')}", fontsize=12, fontweight="bold",
                color=INK, va="top")
        ax.text(1, 0.98, f"{100 * fr:.0f}%   {a.get('done', 0)} of {a.get('total', 0)} {a.get('unit', '')}",
                fontsize=12, color=INK, ha="right", va="top", fontweight="bold")
        ax.add_patch(FancyBboxPatch((0, 0.42), 1, 0.2, boxstyle="round,pad=0,rounding_size=0.02", fc=TRACK,
                                    ec="none"))
        if fr > 0:
            ax.add_patch(FancyBboxPatch((0, 0.42), max(fr, 0.01), 0.2, boxstyle="round,pad=0,rounding_size=0.02",
                                        fc=STATE_COLOUR[st][0], ec="none"))
        rate, eta = a.get("rate_per_min") or 0, a.get("eta_s")
        parts = [f"elapsed {fmt_s(a.get('elapsed_s'))}"]
        if rate:
            parts.append(f"{num(rate)} {a.get('unit', '')} per minute" if rate >= 1
                         else f"{num(60 / rate, 0)} s per item")
        if st == "running" and eta is not None:
            parts.append(f"estimated end at {datetime.fromtimestamp(now + eta):%H:%M} (in {fmt_s(eta)})")
        if st == "stopped":
            parts.append(f"no update for {fmt_s(now - a.get('updated', now))}")
        ax.text(0, 0.3, "   ·   ".join(parts), fontsize=10, color=INK2, va="top")
        last = a.get("last_advance")
        if st == "running":
            live = (f"last item completed at {datetime.fromtimestamp(last):%H:%M:%S} "
                    f"({fmt_s(now - last)} ago)" if last else
                    "the first item is not finished yet: the process is alive (updated "
                    f"{fmt_s(now - a.get('updated', now))} ago)")
            ax.text(1, 0.3, live, fontsize=10, color=INK, va="top", ha="right")
        c = a.get("client", {})
        sub = (f"calls {c.get('calls', 0)} · from cache {c.get('cache_hits', 0)} · failed {c.get('failed', 0)}"
               f" · 429 {c.get('http_429', 0)}" + (f"   ·   next: {a['next_step']}" if a.get("next_step") else ""))
        ax.text(0, 0.08, sub, fontsize=9, color=MUTED, va="top")


def gui(directory: Path, interval: float):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(14, 6.4))
    try:
        fig.canvas.manager.set_window_title("Longitudinal Privacy - progress")
    except AttributeError:
        pass
    draw(fig, directory)
    anim = FuncAnimation(fig, lambda _: draw(fig, directory), interval=interval * 1000, cache_frame_data=False)
    plt.show()
    return anim


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="one text snapshot")
    ap.add_argument("--text", action="store_true", help="console window instead of the charts")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--dir", default=str(PROGRESS_DIR))
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    d = Path(a.dir)

    def text():
        now = time.time()
        return (render_matrix(load(d, max_age_h=24 * 365, now=now), now) + "\n\n"
                + render(load(d, max_age_h=6, now=now), now))

    if a.once:
        print(text())
        return
    if not a.text:
        gui(d, a.interval)
        return
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print(text())
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
