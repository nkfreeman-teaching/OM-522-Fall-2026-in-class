"""A private discrete-event scheduler for FlowSim's customer and server processes.

The scheduler keeps a heap of (time, priority, sequence, event) entries and
processes the smallest entry first. Events at equal times run in priority order,
and events with equal priority run in the order they were queued (not the order
they were created). A process is a generator that yields events and resumes
when each event is processed. These rules follow the SimPy 4.1.2 scheduler that
FlowSim used earlier, and the scheduler was checked against SimPy 4.1.2 at commit
32e96d3.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from functools import partial
from heapq import heappop, heappush
from itertools import count

# Process starts and the stop of run(until=time) come before every other event
# at the same time. FlowSim's shift changes come next, and succeeded events and
# timeouts default to NORMAL. Abandonment follows NORMAL, so a server that frees
# at the instant a customer's patience runs out serves that customer.
URGENT = 0
SHIFT_PRIORITY = 1
NORMAL = 2
ABANDON_PRIORITY = 3


class Event:
    """Something that happens once. Callbacks run when the event is processed."""

    __slots__ = ("scheduler", "callbacks", "triggered", "value")

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        # None after processing, which is how callers tell "processed" apart
        # from "triggered", i.e., queued but not yet popped from the heap.
        self.callbacks: list[Callable[[Event], None]] | None = []
        self.triggered = False
        self.value: object = None

    def succeed(self) -> None:
        if self.triggered:
            raise RuntimeError("An event can succeed only once.")
        self.scheduler.schedule(
            event=self,
            priority=NORMAL,
            delay=0,
        )


class AnyOf(Event):
    """Succeed when the first of several events is processed.

    The value lists the events processed by the time this event is processed,
    so an event tied at the same time is included only if it was popped first.
    """

    __slots__ = ("events",)

    def __init__(self, scheduler: Scheduler, events: list[Event]):
        super().__init__(scheduler)
        self.events = events
        for event in events:
            if event.callbacks is None:
                self.check(event)
            else:
                event.callbacks.append(self.check)
        self.callbacks = [self.build_value]

    def check(self, event: Event) -> None:
        if not self.triggered:
            self.succeed()

    def build_value(self, event: Event) -> None:
        for other in self.events:
            if other.callbacks is not None and self.check in other.callbacks:
                other.callbacks.remove(self.check)
        self.value = [other for other in self.events if other.callbacks is None]


def _raise(error: Exception, event: Event) -> None:
    raise error


class _Process:
    __slots__ = ("scheduler", "generator")

    def __init__(self, scheduler: Scheduler, generator: Generator[Event, object, None]):
        self.scheduler = scheduler
        self.generator = generator
        start = Event(scheduler)
        start.callbacks = [self.resume]
        scheduler.schedule(
            event=start,
            priority=URGENT,
            delay=0,
        )

    def resume(self, event: Event) -> None:
        while True:
            try:
                event = self.generator.send(event.value)
            except StopIteration:
                return
            except Exception as error:
                # Raise the error when its own event is processed, so a run
                # that stops earlier at this time still returns its results.
                # SimPy also defers BaseException, but FlowSim lets Ctrl-C and
                # SystemExit stop the run at once, by the owner's choice.
                failure = Event(self.scheduler)
                failure.callbacks = [partial(_raise, error)]
                self.scheduler.schedule(
                    event=failure,
                    priority=NORMAL,
                    delay=0,
                )
                return
            if event.callbacks is not None:
                event.callbacks.append(self.resume)
                return


class _Stop(Exception):
    pass


def _stop(event: Event) -> None:
    raise _Stop


class Scheduler:
    def __init__(self):
        self.now = 0.0
        self.queue: list[tuple[float, int, int, Event]] = []
        self.sequence = count()

    def schedule(self, event: Event, priority: int, delay: float) -> None:
        # Store now + delay, not an absolute time, so a delay computed as
        # target - now gives the same floating-point time SimPy gave.
        event.triggered = True
        heappush(self.queue, (self.now + delay, priority, next(self.sequence), event))

    def event(self) -> Event:
        return Event(self)

    def timeout(self, delay: float, priority: int = NORMAL) -> Event:
        if delay < 0:
            raise ValueError(f"Negative delay {delay!r}.")
        event = Event(self)
        self.schedule(
            event=event,
            priority=priority,
            delay=delay,
        )
        return event

    def any_of(self, events: list[Event]) -> AnyOf:
        return AnyOf(self, events)

    def process(self, generator: Generator[Event, object, None]) -> None:
        _Process(self, generator)

    def step(self) -> None:
        self.now, _, _, event = heappop(self.queue)
        callbacks, event.callbacks = event.callbacks, None
        # An event is queued once, so it still holds its callbacks here.
        assert callbacks is not None
        for callback in callbacks:
            callback(event)

    def run(self, until: float | Event) -> None:
        """Process events until a time, or until an event is processed.

        A stop at a time runs before other events at that time, which wait for
        the next call to run.
        """
        if isinstance(until, Event):
            if until.callbacks is None:
                return
            until.callbacks.append(_stop)
        else:
            if until <= self.now:
                raise ValueError(f"Cannot run until {until!r} from {self.now!r}.")
            stop = Event(self)
            stop.callbacks = [_stop]
            self.schedule(
                event=stop,
                priority=URGENT,
                delay=until - self.now,
            )
        try:
            while self.queue:
                self.step()
        except _Stop:
            return
        raise RuntimeError("No scheduled events remain, and the stop event never ran.")
