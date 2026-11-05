import threading
import time

from rerun.llm import AdaptiveLimit


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_halves_on_a_burst_of_429_and_not_below_floor():
    c = Clock()
    lim = AdaptiveLimit(16, floor=2, period=30, window=60, clock=c)
    for _ in range(20):
        lim.record(True)
    c.t = 31
    lim.record(True)
    assert lim.limit == 8
    for step in range(5):
        for _ in range(20):
            lim.record(True)
        c.t += 31
        lim.record(True)
    assert lim.limit == 2                 # floor


def test_few_events_do_not_trigger_a_decrease():
    c = Clock()
    lim = AdaptiveLimit(16, period=30, min_events=10, clock=c)
    lim.record(True)
    c.t = 31
    lim.record(True)
    assert lim.limit == 16


def test_grows_by_one_only_when_clean_and_saturated_up_to_ceiling():
    c = Clock()
    lim = AdaptiveLimit(6, start=4, period=30, clock=c)
    for _ in range(20):
        lim.record(False)
    c.t = 31
    lim.record(False)
    assert lim.limit == 4                 # clean but never saturated: no growth
    with lim, lim, lim, lim:              # four calls in flight = saturated
        pass
    for _ in range(20):
        lim.record(False)
    c.t = 62
    lim.record(False)
    assert lim.limit == 5
    from contextlib import ExitStack
    for _ in range(3):
        with ExitStack() as st:           # fill the current limit exactly (saturated, never blocked)
            for _ in range(lim.limit):
                st.enter_context(lim)
        c.t += 31
        lim.record(False)
    assert lim.limit == 6                 # ceiling


def test_never_more_calls_in_flight_than_the_limit():
    lim = AdaptiveLimit(3, clock=time.monotonic)
    peak, cur, lock = [0], [0], threading.Lock()

    def work():
        with lim:
            with lock:
                cur[0] += 1
                peak[0] = max(peak[0], cur[0])
            time.sleep(0.02)
            with lock:
                cur[0] -= 1

    ts = [threading.Thread(target=work) for _ in range(12)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert peak[0] <= 3
