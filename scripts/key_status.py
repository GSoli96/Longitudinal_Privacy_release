"""State of the Ollama Cloud keys: plan, whether a call goes through, usage, estimated reactivation.

Prints key indices only: never keys, account names or server messages (they contain the account name).
Each run is appended to data/private/key_status.jsonl; --history reads that log and prints, per key, when it
went from blocked to active, so the real reset times can be measured instead of estimated.

    .venv\\Scripts\\python.exe scripts\\key_status.py              # all keys, probe gemma4:31b
    .venv\\Scripts\\python.exe scripts\\key_status.py --minimax    # also probe minimax-m3.1-flash (paid model)
    .venv\\Scripts\\python.exe scripts\\key_status.py --history     # only read the log
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
KEYS_PATH = ROOT / ".secrets" / "ollama_keys.json"
LOG = ROOT / "data" / "private" / "key_status.jsonl"
URL = "https://ollama.com"
FREE_PERIOD_USD = 2.4   # a free account is blocked at about 2.50 USD of usage (observed 2026-10-23)


def get(key, path, **params):
    r = requests.get(f"{URL}{path}", params=params, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    return r.json()


def plan_of(key):
    try:
        r = requests.post(f"{URL}/api/me", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        if r.status_code != 200:
            return f"http{r.status_code}"
        return str({k.lower(): v for k, v in r.json().items()}.get("plan", "?"))
    except requests.RequestException as e:
        return type(e).__name__


def probe(key, model):
    body = {"model": model, "messages": [{"role": "user", "content": "ok"}], "stream": False,
            "options": {"num_predict": 1, "temperature": 0}}
    if model.startswith("minimax"):
        body["think"] = False
    try:
        r = requests.post(f"{URL}/api/chat", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
        return r.status_code
    except requests.RequestException as e:
        return type(e).__name__


def ts(s):
    return datetime.fromisoformat(s[:19]).replace(tzinfo=timezone.utc)


def used(buckets):
    return [(ts(b["from"]), b["usage_usd"]) for b in buckets if b["request_count"]]


def estimate(status, hours, days7, days30, usd7):
    """Reason of the block and estimated reactivation (UTC); None when the key works or nothing fits."""
    if status == 200:
        return "attiva", None
    if status != 429:
        return f"http{status}", None
    if usd7 >= FREE_PERIOD_USD and days7:
        return "settimanale", days7[0][0] + timedelta(days=7)
    if hours:
        return "5 ore?", hours[0][0] + timedelta(hours=5)
    first = days30[0][0] if days30 else None
    now = datetime.now(timezone.utc)
    month = datetime(now.year + now.month // 12, now.month % 12 + 1, 1, tzinfo=timezone.utc)
    return "mensile?", min(month, first + timedelta(days=30)) if first else month


def row(i, k, models):
    key = k["key"]
    status = {m: probe(key, m) for m in models}
    out = {"idx": i, "tipo": k.get("type", "free").lower(), "piano": plan_of(key), "stato": status}
    try:
        h = get(key, "/api/usage", range="24h")
        d7 = get(key, "/api/usage", range="7d")
        d30 = get(key, "/api/usage", range="30d")
    except requests.RequestException as e:
        out["errore_usage"] = type(e).__name__
        return out
    hours, days7, days30 = used(h["buckets"]), used(d7["buckets"]), used(d30["buckets"])
    recent = [x for x in hours if x[0] >= datetime.now(timezone.utc) - timedelta(hours=5)]
    usd7 = d7["totals"]["usage_usd"]
    why, when = estimate(status[models[0]], recent, days7, days30, usd7)
    out.update(usd_24h=round(h["totals"]["usage_usd"], 2), usd_7d=round(usd7, 2),
               usd_30d=round(d30["totals"]["usage_usd"], 2), blocco=why,
               riattivazione=when.isoformat() if when else None)
    return out


def local(iso):
    return datetime.fromisoformat(iso).astimezone().strftime("%a %d/%m %H:%M") if iso else ""


def history():
    if not LOG.exists():
        print("no log")
        return
    runs = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    last = {}
    for run in runs:
        for r in run["chiavi"]:
            s = r["stato"].get("gemma4:31b")
            prev = last.get(r["idx"])
            if prev and prev[1] == 429 and s == 200:
                print(f"key {r['idx']:>2}: blocked at {local(prev[0])}, active at {local(run['ora'])}"
                      f" (estimate was {local(prev[2])})")
            last[r["idx"]] = (run["ora"], s, r.get("riattivazione"))
    print(f"{len(runs)} checks in the log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minimax", action="store_true", help="also probe minimax-m3.1-flash")
    ap.add_argument("--history", action="store_true", help="only print the reactivations seen in the log")
    a = ap.parse_args()
    if a.history:
        history()
        return
    keys = json.loads(KEYS_PATH.read_text(encoding="utf-8"))["CLOUD_API_KEYS"]
    models = ["gemma4:31b"] + (["minimax-m3.1-flash"] if a.minimax else [])
    now = datetime.now(timezone.utc)
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(lambda ik: row(ik[0], ik[1], models), enumerate(keys)))
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ora": now.isoformat(), "chiavi": rows}) + "\n")

    print(now.astimezone().strftime("%a %d/%m %H:%M"))
    print(f"{'idx':>3} {'type':<4} {'plan':<6} " + " ".join(f"{m.split(':')[0][:7]:<7}" for m in models)
          + f" {'$24h':>5} {'$7d':>5} {'$30d':>6}  block        reactivation")
    for r in rows:
        print(f"{r['idx']:>3} {r['tipo']:<4} {r['piano']:<6} " + " ".join(f"{r['stato'][m]!s:<7}" for m in models)
              + f" {r.get('usd_24h', ''):>5} {r.get('usd_7d', ''):>5} {r.get('usd_30d', ''):>6}"
              f"  {r.get('blocco', r.get('errore_usage', '')):<12} {local(r.get('riattivazione'))}")
    print("plans:", dict(Counter(r["piano"] for r in rows)))
    for m in models:
        print(f"{m}:", dict(Counter(str(r["stato"][m]) for r in rows)))
    print(f"total 7 days {sum(r.get('usd_7d', 0) for r in rows):.2f} USD")


if __name__ == "__main__":
    main()
