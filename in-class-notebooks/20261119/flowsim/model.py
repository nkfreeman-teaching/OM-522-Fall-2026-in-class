"""Student-facing model definitions, independent of notebook software."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np

from flowsim.diagram import chain_diagram, network_diagram
from flowsim.display import display_html, display_text
from flowsim.distributions import Distribution
from flowsim.engine import simulate
from flowsim.policies import OrderUpTo
from flowsim.records import SUPPLIER
from flowsim.results import (
    Experiment,
    Results,
)
from flowsim.schedules import validate_schedule
from flowsim.validation import Duration, integer, number
from flowsim.validation import duration as duration_check

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

# Closed string choices. Each station and stock-point argument that takes one
# is still validated at run time.
ShiftEnd = Literal["finish", "pause"]
ExternalOverflow = Literal["reject", "wait"]
FailureClock = Literal["available", "busy"]
AbandonFrom = Literal["waiting", "waiting_and_outside"]
Shortage = Literal["backorder", "lost"]
ForecastFrom = Literal["orders", "end_demand"]


@dataclass(
    frozen=True,
    eq=False,
    repr=False,
)
class Node:
    """A source, station, exit, demand, or stock point belonging to one model.

    Model.source, Model.station, Model.demand, Model.stock_point, and
    Model.exit create nodes. A node is immutable and compares by identity.
    Routes take the node objects themselves, not their names.

    Attributes
    ----------
    name : str
        The node's unique name within its model.
    kind : str
        One of 'source', 'station', 'exit', 'demand', or 'stock_point'.

    The remaining fields hold the arguments the node was created with, and a
    field that does not apply to the node's kind is None. Printing a node, or
    showing it in a notebook, lists the fields that are set.

    Example
    -------
    >>> model = flowsim.Model()
    >>> desk = model.station(name="Desk", capacity=2, service_time=4)
    >>> desk.kind, desk.capacity
    ('station', 2)
    """

    name: str
    kind: str
    interarrival: Duration | None = None
    # A station's one duration, or a read-only mapping from customer type to
    # duration.
    service_time: Duration | Mapping[str, Duration] | None = None
    # A station's read-only mapping from customer type to rank, where a lower
    # rank is served first, or None for first-come, first-served.
    priority: Mapping[str, float] | None = None
    # A station's patience, as one duration or a read-only mapping from customer
    # type to duration, and the queues a customer may leave when it runs out.
    patience: Duration | Mapping[str, Duration] | None = None
    abandon_from: AbandonFrom | None = None
    capacity: int = 0
    first_arrival: float | None = None
    limit: int | None = None
    # A source's read-only mapping from customer type to probability, or None
    # for customers without a type.
    types: Mapping[str, float] | None = None
    arrival_rates: tuple[tuple[float, float], ...] | None = None
    staffing: tuple[tuple[float, int], ...] | None = None
    shift_end: ShiftEnd | None = None
    buffer: int | None = None
    external_overflow: ExternalOverflow | None = None
    time_to_failure: Duration | None = None
    repair_time: Duration | None = None
    failure_clock: FailureClock | None = None
    per_period: Duration | None = None
    initial: float | None = None
    # None at a stock point that produces at stations, whose jobs hold at most
    # batch_size units each.
    lead_time: Duration | None = None
    batch_size: float | None = None
    shortage: Shortage | None = None
    policy: OrderUpTo | None = None
    forecast_from: ForecastFrom | None = None

    def _fields(self) -> list[tuple[str, str]]:
        """List the fields that are set, and a station's capacity."""
        shown = []
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in {"name", "kind"}:
                continue
            if value is None or (field.name == "capacity" and self.kind != "station"):
                continue
            shown.append((field.name, _show(value)))
        return shown

    def __repr__(self) -> str:
        return display_text(*self._display())

    def _repr_html_(self) -> str:
        return display_html(*self._display())

    def _display(self) -> tuple[str, list[tuple[str, str]]]:
        return (
            f"{self.kind.replace('_', ' ').capitalize()} {self.name!r}",
            self._fields(),
        )


def _show(value: object) -> str:
    """Show a field value: a distribution's repr, a number, or <function>."""
    if isinstance(value, Mapping):
        return (
            "{"
            + ", ".join(f"{key!r}: {_show(item)}" for key, item in value.items())
            + "}"
        )
    if callable(value) and not isinstance(value, Distribution):
        return "<function>"
    return repr(value)


# Node kinds that customers visit along routes. Demand and stock points belong
# to a supply chain instead, although a stock point that produces releases jobs
# to stations and receives them back.
NETWORK_KINDS = {"source", "station", "exit"}

# One routing rule's destinations and probabilities, in the order given.
Choices = tuple[tuple[str, float], ...]


def _type_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Customer types must be nonempty names, not {value!r}.")
    return value


# How far branch and type probabilities may sum from one. The builder page
# reads it too.
BRANCH_SUM_TOLERANCE = 1e-12


def _type_mix(types: object) -> Mapping[str, float]:
    if not isinstance(types, Mapping) or not types:
        raise ValueError("types must map each customer type to its probability.")
    probabilities = {
        _type_name(kind): number(
            value=probability,
            label=f"Probability of type {kind!r}",
            positive=True,
        )
        for kind, probability in types.items()
    }
    if not math.isclose(
        a=sum(probabilities.values()),
        b=1.0,
        rel_tol=0,
        abs_tol=BRANCH_SUM_TOLERANCE,
    ):
        raise ValueError("Type probabilities must sum to one.")
    return MappingProxyType(probabilities)


def _typed_durations(times: Mapping, label: str) -> Mapping[str, Duration]:
    """Check a mapping from customer type to duration, labeled like "Service time"."""
    if not times:
        raise ValueError(f"A {label.lower()} mapping needs at least one customer type.")
    durations = {}
    for kind, duration in times.items():
        name = _type_name(kind)
        duration_check(duration, f"{label} for type {name!r}")
        durations[name] = duration
    return MappingProxyType(durations)


def _ranks(priority: object) -> Mapping[str, float]:
    if not isinstance(priority, Mapping) or not priority:
        raise ValueError("priority must map each customer type to its rank.")
    return MappingProxyType(
        {
            _type_name(kind): number(rank, f"Rank of type {kind!r}")
            for kind, rank in priority.items()
        }
    )


def _route_types(types: object) -> tuple[str | None, ...]:
    """Return the rule keys for a route or branch, where None means every type."""
    if types is None:
        return (None,)
    if isinstance(types, str) or not isinstance(types, Sequence) or not types:
        raise ValueError("types must be a nonempty list of customer types.")
    names = tuple(_type_name(kind) for kind in types)
    if len(set(names)) != len(names):
        raise ValueError("A route cannot list a customer type twice.")
    return names


def _rule(table: Mapping[str | None, Choices], kind: str | None) -> Choices | None:
    """Return the rule a customer of this type follows from one node."""
    return table[None] if None in table else table.get(kind)


@dataclass(
    frozen=True,
    eq=False,
)
class SupplyChain:
    """Consumer demand and the stock points that order up the chain from it."""

    demand: Node
    stock_points: tuple[Node, ...]
    period: float


class Model:
    """A network of service stations, first-come, first-served unless ranked.

    Each customer type's routes are acyclic. Routes across types may form a
    cycle only through stations with unlimited buffers. A model is either a
    service network (sources, stations, and routes) or a supply chain (demand
    and stock points linked by supply), not both.

    Parameters
    ----------
    time_unit : str, default 'minutes'
        A nonempty label for the model's time unit. Every duration, rate, and
        schedule time uses this unit. The label does not convert units.

    Attributes
    ----------
    exit : Node
        The node where completed customers leave the network.
    time_unit : str
        The time unit label.
    nodes : mapping of str to Node
        Read-only view of the nodes in definition order.
    graph : networkx.DiGraph
        Frozen copy of the routes across every customer type.
    routing : mapping
        Each node's destinations and probabilities by customer type.
    supply_chain : SupplyChain or None
        The linked demand and stock points, or None for a service network.

    The common next calls are source, station, route, branch, validate, diagram,
    run, and experiment. A supply chain uses demand, stock_point, and supply.

    Example
    -------
    >>> model = flowsim.Model(time_unit="minutes")
    >>> arrivals = model.source(name="Arrivals", interarrival=5)
    >>> desk = model.station(name="Desk", capacity=1, service_time=4)
    >>> model.route([arrivals, desk, model.exit])
    >>> results = model.run(duration=480)
    """

    def __init__(self, time_unit: str = "minutes"):
        if not isinstance(time_unit, str) or not time_unit.strip():
            raise ValueError("time_unit must be a nonempty label.")
        self.time_unit = time_unit.strip()
        self._nodes: dict[str, Node] = {}
        self._graph = nx.DiGraph()
        # Each origin's routing rules by customer type, where the key None holds
        # one rule for every type.
        self._routing: dict[str, dict[str | None, Choices]] = {}
        self._supply: SupplyChain | None = None
        self.exit = self._add(
            Node(
                name="Exit",
                kind="exit",
            )
        )

    @property
    def nodes(self) -> Mapping[str, Node]:
        """Return a read-only view of the nodes by name, in definition order."""
        return MappingProxyType(self._nodes)

    @property
    def graph(self) -> nx.DiGraph:
        """Return a frozen copy of the routes across every customer type.

        An edge from an untyped route or branch holds its probability. An edge
        from a typed one instead holds types, a mapping from each customer type
        that may take the edge to its probability.
        """
        return nx.freeze(self._graph.copy())

    @property
    def routing(self) -> Mapping[str, Mapping[str | None, Choices]]:
        """Return each node's destinations and probabilities by customer type.

        The key None holds a node's one route for every type.
        """
        return MappingProxyType(
            {
                origin: MappingProxyType(dict(table))
                for origin, table in self._routing.items()
            }
        )

    @property
    def supply_chain(self) -> SupplyChain | None:
        """Return the supply chain, or None for a service network."""
        return self._supply

    def _add(self, node: Node) -> Node:
        if not isinstance(node.name, str) or not node.name.strip():
            raise ValueError("Each node needs a nonempty name.")
        if node.name in self._nodes:
            raise ValueError(f"The name {node.name!r} already exists in this model.")
        if node.kind == "source" and self._supply is not None:
            raise ValueError(_ONE_KIND)
        self._nodes[node.name] = node
        if node.kind in NETWORK_KINDS:
            self._graph.add_node(node.name)
        return node

    def _has_sources(self) -> bool:
        return any(node.kind == "source" for node in self._nodes.values())

    def _owned(self, node: Node) -> None:
        if isinstance(node, str):
            raise ValueError(
                "Routes take the nodes that model.source(...), model.station(...), "
                f"and model.exit return, not names. You passed the string {node!r}; "
                "pass the variable that holds that node."
            )
        if isinstance(node, Node) and self._nodes.get(node.name) is not node:
            raise ValueError(
                "Routes must use nodes created by this model, but "
                f"{node.name!r} is a node created by a different Model."
            )
        if not isinstance(node, Node):
            raise ValueError("Routes must use nodes created by this model.")

    def source(
        self,
        name: str,
        interarrival: Duration | None = None,
        *,
        first_arrival: float | None = None,
        limit: int | None = None,
        rates: Sequence[tuple[float, float]] | None = None,
        types: Mapping[str, float] | None = None,
    ) -> Node:
        """Create a source of customer arrivals.

        Give exactly one of interarrival (renewal arrivals) or rates (a
        piecewise constant Poisson process). The first arrival occurs after one
        sampled interarrival time, or at the first arrival the rate schedule
        yields, unless first_arrival is given. A source is not allowed in a
        model with a supply chain.

        Parameters
        ----------
        name : str
            Unique nonempty name within the model.
        interarrival : float or callable, optional
            Time between arrivals in model time units. A positive number, or a
            function of one argument (the random generator) that returns a
            positive number, e.g., flowsim.exponential(mean=4). Required
            unless rates is given.
        first_arrival : float, optional
            Time of the first arrival, at least 0. It overrides the sampled
            first time, including during a zero-rate period.
        limit : int, optional
            Maximum number of customers the source creates, at least 0.
            Without it, arrivals continue until the run closes.
        rates : sequence of (float, float), optional
            Arrival rate schedule of (start time, customers per time unit)
            pairs. It must begin at time 0 and have strictly increasing start
            times, and each rate is at least 0. Replaces interarrival.
        types : mapping of str to float, optional
            Customer types and the probability, greater than 0, that an arrival
            has each type. The probabilities must sum to one. Without types,
            customers have no type.

        Returns
        -------
        Node
            The source, to pass to route or branch.

        Example
        -------
        >>> model = flowsim.Model()
        >>> arrivals = model.source(
        ...     name="Arrivals",
        ...     interarrival=flowsim.exponential(mean=5),
        ...     limit=100,
        ... )
        """
        if (interarrival is None) == (rates is None):
            raise ValueError("Provide either interarrival or rates, but not both.")
        arrival_rates = None
        if rates is not None:
            arrival_rates = validate_schedule(
                points=rates,
                label="Arrival rate schedule",
            )
        else:
            duration_check(
                value=interarrival,
                label="Interarrival time",
                positive=True,
            )
        first = (
            None
            if first_arrival is None
            else number(first_arrival, "First arrival time")
        )
        count = (
            None
            if limit is None
            else integer(
                value=limit,
                label="Arrival limit",
                minimum=0,
            )
        )
        return self._add(
            Node(
                name=name,
                kind="source",
                interarrival=interarrival,
                first_arrival=first,
                limit=count,
                types=None if types is None else _type_mix(types),
                arrival_rates=arrival_rates,
            )
        )

    def station(
        self,
        name: str,
        capacity: int,
        service_time: Duration | Mapping[str, Duration],
        *,
        staffing: Sequence[tuple[float, float]] | None = None,
        shift_end: ShiftEnd | None = None,
        buffer: int | None = None,
        external_overflow: ExternalOverflow | None = None,
        time_to_failure: Duration | None = None,
        repair_time: Duration | None = None,
        failure_clock: FailureClock | None = None,
        priority: Mapping[str, float] | None = None,
        patience: Duration | Mapping[str, Duration] | None = None,
        abandon_from: AbandonFrom | None = None,
    ) -> Node:
        """Create a station of identical servers with one waiting line.

        Service is first-come, first-served unless priority ranks customer
        types, and it is never preempted. Schedules, finite buffers,
        abandonment, and failures are off unless requested, and each requires
        explicit companion arguments, listed below.

        service_time is one duration, or a mapping from customer type to the
        duration for that type. priority maps each customer type to a rank. A
        lower rank leaves the waiting line first, and equal ranks keep the order
        customers joined it. A higher-priority customer never interrupts a
        service in progress, while a staffing schedule with shift_end='pause'
        and server failures can pause it.

        patience is one duration, or a mapping from customer type to duration,
        after which a customer who has not started service leaves the network.
        abandon_from='waiting' starts the clock when the customer enters the
        waiting line. 'waiting_and_outside' also starts it when an outside
        arrival joins the queue outside a full buffer, and the clock keeps
        running after admission.

        Parameters
        ----------
        name : str
            Unique nonempty name within the model.
        capacity : int
            Number of installed servers, at least 1.
        service_time : float, callable, or mapping
            Service duration in model time units. A number of at least 0, a
            function of one argument (the random generator), or a mapping from
            customer type to either, e.g., flowsim.exponential(mean=4).
        staffing : sequence of (float, int), optional
            Schedule of (start time, servers on duty) pairs. It must begin at
            time 0, have strictly increasing times, and keep each level between
            0 and capacity. Requires shift_end.
        shift_end : {'finish', 'pause'}, optional
            What an off-duty server does with its current job. 'finish' lets
            it complete the job and its downstream transfer, then blocks new
            work. 'pause' suspends processing until that same server is
            scheduled again. Required with staffing, and an error without it.
        buffer : int, optional
            Maximum number of jobs waiting, at least 0 (0 allows direct service
            only). Without it, the waiting room is unlimited. Requires
            external_overflow.
        external_overflow : {'reject', 'wait'}, optional
            What a source arrival does when the buffer is full. 'reject'
            records a rejected arrival. 'wait' holds it in an outside queue
            until admission is possible. Required with buffer, and an error
            without it.
        time_to_failure : float or callable, optional
            Lifetime of a server before it fails, a positive duration in model
            time units. Requires repair_time and failure_clock.
        repair_time : float or callable, optional
            Repair duration, at least 0. Requires time_to_failure and
            failure_clock.
        failure_clock : {'available', 'busy'}, optional
            When a server's lifetime runs down. 'available' advances while the
            server is up and either on duty or, with shift_end='finish',
            finishing a job after its shift, including idle and blocked time.
            'busy' advances only during processing. Required with
            time_to_failure, and an error without it.
        priority : mapping of str to float, optional
            Rank for each customer type. A lower rank is served first, and
            equal ranks are served first-come, first-served.
        patience : float, callable, or mapping, optional
            Time a customer waits before leaving without service, as one
            duration or a mapping from customer type to duration. Requires
            abandon_from.
        abandon_from : {'waiting', 'waiting_and_outside'}, optional
            Where the patience clock starts. 'waiting' starts it when the
            customer joins the waiting line. 'waiting_and_outside' also starts
            it when an outside arrival joins the queue outside a full buffer,
            and it requires buffer with external_overflow='wait'. Required with
            patience, and an error without it.

        Returns
        -------
        Node
            The station, to pass to route or branch.

        Example
        -------
        >>> model = flowsim.Model()
        >>> clinic = model.station(
        ...     name="Clinic",
        ...     capacity=3,
        ...     service_time=flowsim.exponential(mean=12),
        ...     buffer=5,
        ...     external_overflow="reject",
        ... )
        """
        count = integer(capacity, "Station capacity")
        if isinstance(service_time, Mapping):
            service_time = _typed_durations(service_time, "Service time")
        else:
            duration_check(service_time, "Service time")
        staff = None
        if staffing is not None:
            if shift_end not in {"finish", "pause"}:
                raise ValueError(
                    "Scheduled staffing requires shift_end='finish' or 'pause'."
                )
            # With counts=True every level is already an int.
            staff = tuple(
                (time, int(level))
                for time, level in validate_schedule(
                    points=staffing,
                    label="Staffing schedule",
                    counts=True,
                    maximum=count,
                )
            )
        elif shift_end is not None:
            raise ValueError("shift_end requires a staffing schedule.")
        waiting_limit = None
        if buffer is not None:
            waiting_limit = integer(
                value=buffer,
                label="Buffer capacity",
                minimum=0,
            )
            if external_overflow not in {"reject", "wait"}:
                raise ValueError(
                    "A finite buffer requires external_overflow='reject' or 'wait'."
                )
        elif external_overflow is not None:
            raise ValueError("external_overflow requires a finite buffer.")
        if (time_to_failure is None) != (repair_time is None):
            raise ValueError("Provide both time_to_failure and repair_time.")
        if time_to_failure is not None:
            if failure_clock not in {"available", "busy"}:
                raise ValueError(
                    "Failures require failure_clock='available' or 'busy'."
                )
            duration_check(
                value=time_to_failure,
                label="Time to failure",
                positive=True,
            )
            duration_check(repair_time, "Repair time")
        elif failure_clock is not None:
            raise ValueError("failure_clock requires time_to_failure and repair_time.")
        if patience is not None:
            if abandon_from not in {"waiting", "waiting_and_outside"}:
                raise ValueError(
                    "Patience requires abandon_from='waiting' or 'waiting_and_outside'."
                )
            if abandon_from == "waiting_and_outside" and external_overflow != "wait":
                raise ValueError(
                    "abandon_from='waiting_and_outside' requires a finite buffer "
                    "with external_overflow='wait'."
                )
            if isinstance(patience, Mapping):
                patience = _typed_durations(patience, "Patience")
            else:
                duration_check(patience, "Patience")
        elif abandon_from is not None:
            raise ValueError("abandon_from requires patience.")
        return self._add(
            Node(
                name=name,
                kind="station",
                service_time=service_time,
                capacity=count,
                staffing=staff,
                shift_end=shift_end,
                buffer=waiting_limit,
                external_overflow=external_overflow,
                time_to_failure=time_to_failure,
                repair_time=repair_time,
                failure_clock=failure_clock,
                priority=None if priority is None else _ranks(priority),
                patience=patience,
                abandon_from=abandon_from,
            )
        )

    def demand(self, name: str, per_period: Duration) -> Node:
        """Create consumer demand, a quantity drawn once per review period.

        Demand belongs to a supply chain. Link it to stock points with supply.

        Parameters
        ----------
        name : str
            Unique nonempty name within the model.
        per_period : float or callable
            Units demanded in one review period. A number of at least 0, or a
            function of one argument (the random generator) that returns one,
            e.g., flowsim.poisson(mean=20).

        Returns
        -------
        Node
            The demand, the first item of the chain passed to supply.

        Example
        -------
        >>> model = flowsim.Model(time_unit="days")
        >>> sales = model.demand(name="Sales", per_period=flowsim.poisson(mean=20))
        """
        duration_check(per_period, "Demand per period")
        return self._add(
            Node(
                name=name,
                kind="demand",
                per_period=per_period,
            )
        )

    def stock_point(
        self,
        name: str,
        *,
        initial: float,
        lead_time: Duration | None,
        shortage: Shortage,
        policy: OrderUpTo,
        forecast_from: ForecastFrom,
        batch_size: float | None = None,
    ) -> Node:
        """Create a stock point that fills orders from stock and reorders each period.

        lead_time is the delay from its supplier's shipment to its own receipt.
        Every argument is a behavioral choice, so none has a default.

        lead_time=None makes the top stage produce at stations instead. Its
        route leaves it and returns to it, each order splits into jobs of
        batch_size units (the last one smaller), and a finished job enters
        stock at the first review at or after it leaves the last station.

        Parameters
        ----------
        name : str
            Unique nonempty name within the model.
        initial : float
            Stock on hand at time 0, at least 0, with nothing on order.
        lead_time : float, callable, or None
            Delay from the supplier's shipment to this stage's receipt, in model
            time units. A positive constant that is a whole number of review
            periods (checked by supply), or a function of one argument (the
            random generator). None makes the stage produce at stations, which
            requires batch_size and an explicit cover in the policy. A sampled
            lead time also requires an explicit cover. A constant lead time
            forbids cover.
        shortage : {'backorder', 'lost'}
            What happens to demand that stock cannot fill. 'backorder' keeps
            it and ships it first when stock arrives. 'lost' discards it, and
            the downstream stage stops expecting it.
        policy : OrderUpTo
            The ordering rule, from flowsim.order_up_to.
        forecast_from : {'orders', 'end_demand'}
            What the stage forecasts from. 'orders' uses the orders it
            receives. 'end_demand' uses consumer demand (information sharing).
        batch_size : float, optional
            Units per production job, greater than 0. Required when
            lead_time=None, and an error otherwise.

        Returns
        -------
        Node
            The stock point, to list in the chain passed to supply.

        Example
        -------
        >>> model = flowsim.Model(time_unit="days")
        >>> store = model.stock_point(
        ...     name="Store",
        ...     initial=100,
        ...     lead_time=2,
        ...     shortage="backorder",
        ...     policy=flowsim.order_up_to(
        ...         forecast=flowsim.moving_average(window=5),
        ...         safety_factor=1.0,
        ...     ),
        ...     forecast_from="orders",
        ... )
        """
        stock = number(initial, "Initial stock")
        if not isinstance(policy, OrderUpTo):
            raise ValueError("policy must come from flowsim.order_up_to.")
        size = None
        if lead_time is None:
            if policy.cover is None:
                raise ValueError(
                    "A stock point that produces (lead_time=None) needs an "
                    "explicit cover in order_up_to, since its stations set its "
                    "lead time."
                )
            size = number(
                value=batch_size,
                label="Batch size",
                positive=True,
            )
        elif batch_size is not None:
            raise ValueError(
                "batch_size requires lead_time=None, i.e., a stock point that "
                "produces at stations."
            )
        else:
            duration_check(
                value=lead_time,
                label="Lead time",
                positive=True,
            )
            if callable(lead_time) and policy.cover is None:
                raise ValueError(
                    "A sampled lead time needs an explicit cover in order_up_to, "
                    "since FlowSim does not estimate the lead time's mean."
                )
            if not callable(lead_time) and policy.cover is not None:
                raise ValueError(
                    "With a constant lead time, cover is lead_time / period. Give "
                    "cover only for a sampled lead time."
                )
        if shortage not in {"backorder", "lost"}:
            raise ValueError("shortage must be 'backorder' or 'lost'.")
        if forecast_from not in {"orders", "end_demand"}:
            raise ValueError("forecast_from must be 'orders' or 'end_demand'.")
        return self._add(
            Node(
                name=name,
                kind="stock_point",
                initial=stock,
                lead_time=lead_time,
                batch_size=size,
                shortage=shortage,
                policy=policy,
                forecast_from=forecast_from,
            )
        )

    def supply(self, chain: Sequence[Node], *, period: float) -> None:
        """Link demand to stock points, each ordering from the next in the list.

        The last stock point orders from a supplier with unlimited stock that
        ships in the same period, unless it produces at stations. Every stage
        reviews once per period. A model with a supply chain has no sources.

        Parameters
        ----------
        chain : sequence of Node
            Demand from Model.demand first, then one or more stock points from
            Model.stock_point, ordered from the one nearest consumers to the
            one nearest the supplier. Only the last may produce
            (lead_time=None).
        period : float
            Review period in model time units, greater than 0. Each constant
            lead time must be a whole number of periods.

        Returns
        -------
        None

        Example
        -------
        >>> model.supply([sales, store], period=1)
        """
        if self._supply is not None:
            raise ValueError("This model already has a supply chain.")
        if self._has_sources():
            raise ValueError(_ONE_KIND)
        if len(chain) < 2:
            raise ValueError(
                "A supply chain needs demand and at least one stock point."
            )
        for node in chain:
            self._owned(node)
        if any(node.name == SUPPLIER for node in chain):
            raise ValueError(
                f"The name {SUPPLIER!r} is reserved for the unlimited top supplier."
            )
        if len({node.name for node in chain}) != len(chain):
            raise ValueError("A supply chain cannot list a node twice.")
        demand, *stock_points = chain
        if demand.kind != "demand":
            raise ValueError("A supply chain starts with demand from model.demand.")
        if any(node.kind != "stock_point" for node in stock_points):
            raise ValueError("After demand, a supply chain lists only stock points.")
        length = number(
            value=period,
            label="Review period",
            positive=True,
        )
        if any(node.lead_time is None for node in stock_points[:-1]):
            raise ValueError(
                "Only the top stage can produce (lead_time=None), since every "
                "other stage orders from the stage above it."
            )
        for node in stock_points:
            if node.lead_time is None or callable(node.lead_time):
                continue
            periods = round(node.lead_time / length)
            if periods < 1 or not math.isclose(
                a=periods * length,
                b=node.lead_time,
                rel_tol=1e-9,
                abs_tol=0,
            ):
                raise ValueError(
                    f"The constant lead time at {node.name!r} must be a whole number "
                    "of review periods, since receipts happen at reviews."
                )
        self._supply = SupplyChain(
            demand=demand,
            stock_points=tuple(stock_points),
            period=length,
        )

    def route(
        self,
        nodes: Sequence[Node],
        *,
        types: Sequence[str] | None = None,
    ) -> None:
        """Connect a sequence of nodes, without replacing an existing route.

        Repeating a node's existing route is allowed. Giving a different route
        from a node that already has one raises an error. A stock point that
        produces starts and ends its route at itself.

        Parameters
        ----------
        nodes : sequence of Node
            At least two nodes created by this model, in visiting order,
            normally ending at model.exit.
        types : sequence of str, optional
            Customer types that follow the route. Without it, every customer
            does.

        Returns
        -------
        None

        Example
        -------
        >>> model.route([arrivals, desk, model.exit])
        """
        if len(nodes) < 2:
            raise ValueError("A route needs at least two nodes.")
        for node in nodes:
            self._owned(node)
        keys = _route_types(types)
        edges = list(zip(nodes[:-1], nodes[1:]))
        for origin, destination in edges:
            self._check_edge(origin, destination)
        # A production route leaves its stock point and returns to it.
        loop = nodes[0] is nodes[-1] and nodes[0].kind == "stock_point"
        if len({node.name for node in nodes}) != len(nodes) - loop:
            self._cycle_error(keys[0])
        rules = []
        for origin, destination in edges:
            choices = ((destination.name, 1.0),)
            for key in keys:
                # Repeating a node's existing route is allowed.
                if self._routing.get(origin.name, {}).get(key) == choices:
                    continue
                self._check_free(
                    origin=origin,
                    key=key,
                    message=(
                        f"{origin.name!r} already has a different route"
                        + ("." if key is None else f" for type {key!r}.")
                    ),
                )
                rules.append((origin.name, key, choices))
        self._commit(rules)

    def _check_edge(self, origin: Node, destination: Node) -> None:
        if origin.kind == "exit":
            raise ValueError("The exit cannot have an outgoing route.")
        if destination.kind == "source":
            raise ValueError("A source cannot receive routed customers.")
        if "demand" in {origin.kind, destination.kind}:
            raise ValueError("Demand belongs to a supply chain and has no routes.")
        if origin.kind == "stock_point" and destination.kind != "station":
            raise ValueError("A stock point releases production jobs to a station.")
        if destination.kind == "stock_point" and origin.kind != "station":
            raise ValueError(
                "Only a station can return finished jobs to a stock point."
            )

    def branch(
        self,
        origin: Node,
        destinations: Mapping[Node, float],
        *,
        types: Sequence[str] | None = None,
    ) -> None:
        """Choose one destination per customer using fixed probabilities.

        Parameters
        ----------
        origin : Node
            The node customers leave. It must not already have a route for the
            same customer types.
        destinations : mapping of Node to float
            Each destination and its probability, greater than 0. The
            probabilities must sum to one.
        types : sequence of str, optional
            Customer types that follow the branch. Without it, every customer
            does.

        Returns
        -------
        None

        Example
        -------
        >>> model.branch(desk, {exam: 0.7, model.exit: 0.3})
        """
        self._owned(origin)
        if not destinations:
            raise ValueError("A branch needs at least one destination.")
        keys = _route_types(types)
        for key in keys:
            self._check_free(
                origin=origin,
                key=key,
                message=(
                    f"{origin.name!r} already has an outgoing route"
                    + ("." if key is None else f" for type {key!r}.")
                ),
            )
        probabilities = []
        for destination, probability in destinations.items():
            self._owned(destination)
            self._check_edge(origin, destination)
            probabilities.append(
                number(
                    value=probability,
                    label="Probability",
                    positive=True,
                )
            )
        if not math.isclose(
            a=sum(probabilities),
            b=1.0,
            rel_tol=0,
            abs_tol=BRANCH_SUM_TOLERANCE,
        ):
            raise ValueError("Branch probabilities must sum to one.")
        choices = tuple(
            (destination.name, probability)
            for destination, probability in zip(destinations, probabilities)
        )
        self._commit([(origin.name, key, choices) for key in keys])

    def _check_free(self, origin: Node, key: str | None, message: str) -> None:
        """Reject a rule that would overlap one of the origin's existing rules."""
        current = self._routing.get(origin.name, {})
        if (key is None and current) or key in current:
            raise ValueError(message)
        if key is not None and None in current:
            raise ValueError(
                f"{origin.name!r} already has a route for every customer type."
            )

    def _cycle_error(self, key: str | None) -> None:
        if key is None:
            raise ValueError("FlowSim supports acyclic routes only.")
        raise ValueError(
            f"The routes for type {key!r} form a cycle. Each customer type's "
            "routes must be acyclic."
        )

    def _commit(self, rules: list[tuple[str, str | None, Choices]]) -> None:
        """Validate new routing rules as one edit, then add them to the model."""
        if not rules:
            return
        routing = {origin: dict(table) for origin, table in self._routing.items()}
        graph = self._graph.copy()
        for origin, key, choices in rules:
            routing.setdefault(origin, {})[key] = choices
            for destination, probability in choices:
                if key is None:
                    graph.add_edge(
                        u_of_edge=origin,
                        v_of_edge=destination,
                        probability=probability,
                    )
                    continue
                if graph.has_edge(origin, destination):
                    types = dict(graph[origin][destination]["types"])
                else:
                    types = {}
                types[key] = probability
                graph.add_edge(
                    u_of_edge=origin,
                    v_of_edge=destination,
                    types=MappingProxyType(types),
                )
        # Check the untyped routes, then each type's routes in first-use order.
        keys = dict.fromkeys(
            [None, *(key for table in routing.values() for key in table)]
        )
        # A finished job's return to its stock point ends its route, so that
        # edge closes no cycle.
        returns = [
            (origin, destination)
            for origin, destination in graph.edges
            if self._nodes[destination].kind == "stock_point"
        ]
        for key in keys:
            followed = nx.DiGraph()
            for origin, table in routing.items():
                rule = table.get(None) if key is None else _rule(table, key)
                for destination, _ in rule or ():
                    followed.add_edge(origin, destination)
            followed.remove_edges_from(returns)
            if not nx.is_directed_acyclic_graph(followed):
                self._cycle_error(key)
        # A cycle of finite buffers can deadlock, with each blocked server
        # waiting for room at the next station on the cycle.
        flows = graph.copy()
        flows.remove_edges_from(returns)
        cyclic = {
            name
            for component in nx.strongly_connected_components(flows)
            if len(component) > 1
            for name in component
        }
        for node in self._nodes.values():
            if node.name in cyclic and node.buffer is not None:
                raise ValueError(
                    f"{node.name!r} has a finite buffer and lies on a cycle of "
                    "routes across customer types. Blocked servers on such a "
                    "cycle can wait on each other forever, so its stations need "
                    "unlimited buffers."
                )
        self._routing = routing
        self._graph = graph

    def __repr__(self) -> str:
        return display_text(*self._display())

    def _repr_html_(self) -> str:
        return display_html(*self._display())

    def _route_lines(self) -> list[str]:
        """Write the routes as compact lines, one chain or branch per line."""
        keys = []
        for table in self._routing.values():
            keys += [key for key in table if key not in keys]
        keys.sort(key=lambda key: (key is not None, str(key)))
        lines = []
        for key in keys:
            rules = {
                origin: table[key]
                for origin, table in self._routing.items()
                if key in table
            }
            prefix = "" if key is None else f"[{key}] "
            targets = {name for choices in rules.values() for name, _ in choices}
            queue = [origin for origin in rules if origin not in targets]
            queue += [origin for origin in rules if origin in targets]
            done = set()
            while queue:
                current = queue.pop(0)
                if current in done:
                    continue
                text = current
                while True:
                    done.add(current)
                    choices = rules[current]
                    if len(choices) == 1:
                        following = choices[0][0]
                        text += f" -> {following}"
                        if following in rules and following not in done:
                            current = following
                            continue
                    else:
                        text += " -> " + ", ".join(
                            f"{name} ({probability:.0%})"
                            for name, probability in choices
                        )
                        queue = [name for name, _ in choices if name in rules] + queue
                    break
                lines.append(prefix + text)
        return lines

    def _display(self) -> tuple[str, list[tuple[str, str]]]:
        counts = {
            kind: sum(node.kind == kind for node in self._nodes.values())
            for kind in ["source", "station", "exit", "stock_point"]
        }
        parts = [
            f"{count} {noun}{'' if count == 1 else 's'}"
            for count, noun in [
                (counts["source"], "source"),
                (counts["station"], "station"),
                (counts["exit"], "exit"),
                (counts["stock_point"], "stock point"),
            ]
        ]
        rows = []
        if self._supply is not None:
            chain = [self._supply.demand, *self._supply.stock_points]
            rows.append(
                (
                    "Chain",
                    " -> ".join(node.name for node in chain)
                    + f" (period {self._supply.period:g} {self.time_unit})",
                )
            )
        shown = 8
        lines = self._route_lines()
        for index, line in enumerate(lines[:shown]):
            rows.append(("Routes" if index == 0 else "", line))
        if len(lines) > shown:
            rows.append(("", f"... and {len(lines) - shown} more"))
        # Validation grows with the square of the node count, so a display of
        # a very large model skips it.
        if len(self._nodes) > _DISPLAY_VALIDATE_LIMIT:
            status = "not checked for a model this large; call model.validate()"
        else:
            try:
                self.validate()
                status = "passes"
            except ValueError as error:
                status = "fails: " + " ".join(str(error).split())
        rows.append(("Validate", status))
        rows += [
            ("Next", "model.run(duration, seed=42), model.experiment(...)"),
            ("", "model.diagram()"),
        ]
        return f"Model in {self.time_unit}: " + ", ".join(parts), rows

    def validate(self) -> None:
        """Reject disconnected stations and routes that cannot reach the exit.

        run and experiment call this first. Call it directly to check a model
        while building it. A supply chain is checked for unlinked demand and
        stock points and for production routes instead.

        Returns
        -------
        None
            Nothing is returned. A ValueError describes the first problem found.

        Example
        -------
        >>> model.validate()
        """
        chain = () if self._supply is None else self._supply.stock_points
        for node in self._nodes.values():
            unlinked_stock = node.kind == "stock_point" and node not in chain
            unlinked_demand = node.kind == "demand" and (
                self._supply is None or node is not self._supply.demand
            )
            if unlinked_stock or unlinked_demand:
                raise ValueError(f"{node.name!r} is not in a supply chain.")
        if self._supply is not None:
            self._validate_production(self._supply)
            return
        sources = [node for node in self._nodes.values() if node.kind == "source"]
        if not sources:
            raise ValueError(
                "The model needs at least one arrival source or a supply chain."
            )
        reachable = self._reachable()
        # A reached node with no route out usually means a forgotten exit, which
        # would otherwise surface as an unreachable exit.
        for node in self._nodes.values():
            if node.name in reachable - set(self._routing) - {self.exit.name}:
                raise ValueError(
                    f"{node.name!r} has no onward route. End its route with "
                    "model.exit, e.g. model.route([..., service, model.exit])."
                )
        for node in self._nodes.values():
            if node.kind not in NETWORK_KINDS:
                continue
            if node.name not in reachable:
                raise ValueError(f"{node.name!r} is unreachable from every source.")
            if node is not self.exit and not nx.has_path(
                G=self._graph,
                source=node.name,
                target=self.exit.name,
            ):
                raise ValueError(f"{node.name!r} needs a route to the exit.")
        for source in sources:
            self._check_types(source)

    def _validate_production(self, supply: SupplyChain) -> None:
        """Check that the top stage's jobs reach every station and return to it."""
        top = supply.stock_points[-1]
        stations = [node for node in self._nodes.values() if node.kind == "station"]
        for node in supply.stock_points:
            if node.lead_time is not None and node.name in self._graph:
                raise ValueError(
                    f"{node.name!r} has a route, and only a stock point with "
                    "lead_time=None produces at stations."
                )
        if top.lead_time is not None:
            if stations:
                raise ValueError(
                    f"Stations in a supply chain make {top.name!r}'s orders, so "
                    f"{top.name!r} needs lead_time=None and a route through them."
                )
            return
        if top.name not in self._routing:
            raise ValueError(
                f"{top.name!r} produces, so it needs a route to its first station."
            )
        released = nx.descendants(self._graph, top.name)
        for node in stations:
            if node.name in released and node.name not in self._routing:
                raise ValueError(
                    f"{node.name!r} has no onward route. End the production route "
                    f"by returning to {top.name!r}, e.g. "
                    "model.route([factory, ..., station, factory])."
                )
        for node in stations:
            if not nx.has_path(
                G=self._graph,
                source=top.name,
                target=node.name,
            ):
                raise ValueError(f"{node.name!r} is unreachable from {top.name!r}.")
            if nx.has_path(
                G=self._graph,
                source=node.name,
                target=self.exit.name,
            ):
                raise ValueError(
                    f"{node.name!r} routes to the exit, and production jobs "
                    f"return to {top.name!r} instead."
                )
            if not nx.has_path(
                G=self._graph,
                source=node.name,
                target=top.name,
            ):
                raise ValueError(f"{node.name!r} needs a route back to {top.name!r}.")
            if node.patience is not None:
                raise ValueError(
                    f"{node.name!r} sets patience, and production jobs never abandon."
                )
            if node.external_overflow == "reject":
                raise ValueError(
                    f"{node.name!r} rejects outside arrivals, and released "
                    "production jobs cannot be rejected."
                )
        self._check_types(top)

    def _check_types(self, source: Node) -> None:
        """Follow each of a source's types and check durations, ranks, and routes.

        Routes may be added after stations, so these checks wait for a run. A
        stock point that produces is checked as the source of its untyped jobs.
        """
        for kind in (None,) if source.types is None else source.types:
            reached = set()
            stack = [source.name]
            while stack:
                name = stack.pop()
                if name in reached or name == self.exit.name:
                    continue
                reached.add(name)
                node = self._nodes[name]
                service_time = node.service_time
                if isinstance(service_time, Mapping) and kind is None:
                    raise ValueError(
                        f"{name!r} sets service times by customer type, and "
                        f"customers from {source.name!r} have no type."
                    )
                if isinstance(service_time, Mapping) and kind not in service_time:
                    raise ValueError(
                        f"{name!r} has no service time for type {kind!r}, which "
                        f"can reach it from {source.name!r}."
                    )
                if isinstance(node.patience, Mapping) and kind is None:
                    raise ValueError(
                        f"{name!r} sets patience by customer type, and customers "
                        f"from {source.name!r} have no type."
                    )
                if isinstance(node.patience, Mapping) and kind not in node.patience:
                    raise ValueError(
                        f"{name!r} has no patience for type {kind!r}, which can "
                        f"reach it from {source.name!r}."
                    )
                if node.priority is not None and kind is None:
                    raise ValueError(
                        f"{name!r} ranks customers by type, and customers from "
                        f"{source.name!r} have no type."
                    )
                if node.priority is not None and kind not in node.priority:
                    raise ValueError(
                        f"{name!r} has no rank for type {kind!r}, which can "
                        f"reach it from {source.name!r}."
                    )
                rule = _rule(self._routing[name], kind)
                if rule is None and kind is None:
                    raise ValueError(
                        f"{name!r} routes by customer type, and customers from "
                        f"{source.name!r} have no type."
                    )
                if rule is None:
                    raise ValueError(
                        f"{name!r} has no route for type {kind!r}, which can "
                        f"reach it from {source.name!r}."
                    )
                stack.extend(destination for destination, _ in rule)

    def diagram(self, check: bool = False) -> Figure:
        """Draw the model's nodes and routes, or its supply chain.

        A service network draws its sources, stations, exit, and routes. With
        check=False, an incomplete model still draws. A node that no source
        reaches and a source or station with no onward route are drawn in red,
        with the reason on hover. With check=True, the model is validated
        first, so an incomplete model raises the error that run() would raise.
        A supply chain draws the outside supplier, its stock points, and
        demand in one row, with each stock point's lead time on the arrow into
        it and a producing stage's stations below it. In either drawing, an
        arrow that would pass over another node curves around it.

        Parameters
        ----------
        check : bool, default False
            Whether to validate the model first.

        Returns
        -------
        plotly.graph_objects.Figure
            The network or supply-chain diagram.

        Example
        -------
        >>> figure = model.diagram()
        >>> figure.show()
        """
        if not isinstance(check, bool):
            raise ValueError("check must be True or False.")
        if check:
            self.validate()
        if self._supply is not None:
            return chain_diagram(self)
        if any(node.kind not in NETWORK_KINDS for node in self._nodes.values()):
            raise ValueError(
                "This model has demand or stock points but no supply chain yet. "
                "Link them with model.supply([demand, ...], period=...) to draw it."
            )
        return network_diagram(
            model=self,
            problems=self._route_problems(),
        )

    def _reachable(self) -> set[str]:
        """Return the sources and every node a source's routes reach."""
        reachable = set()
        for node in self._nodes.values():
            if node.kind == "source":
                reachable.add(node.name)
                reachable.update(nx.descendants(self._graph, node.name))
        return reachable

    def _route_problems(self) -> dict[str, str]:
        """Name each node whose routes keep the network from running, with why."""
        reachable = self._reachable()
        problems = {}
        for node in self._nodes.values():
            notes = []
            if node.kind != "source" and node.name not in reachable:
                notes.append("No source reaches it.")
            if node.kind in {"source", "station"} and node.name not in self._routing:
                notes.append(
                    "It has no onward route. End its route with model.exit, "
                    "e.g., model.route([..., station, model.exit])."
                )
            if notes:
                problems[node.name] = " ".join(notes)
        return problems

    def run(
        self,
        duration: float,
        seed: int = 42,
        *,
        warmup: float = 0,
        drain: bool = False,
    ) -> Results:
        """Simulate once and return the results.

        Arrivals occur in [0, duration). Each run starts from fresh state with
        its own random streams, so one seed reproduces one run exactly. With
        drain=False, service also stops at duration. With drain=True, arrivals
        stop at duration and the run continues until every customer who was
        not rejected, including any waiting outside a full buffer, has
        completed service or abandoned (not available for a supply chain,
        which reviews only until closing). Queue averages, utilization,
        throughput, and measurement counts use the window [warmup, duration),
        while whole-run counts such as arrivals and completed cover the entire
        run, and wait and flow-time averages use cohorts of arrivals at or
        after warmup whose outcomes were observed.

        Parameters
        ----------
        duration : float
            Closing time in model time units, greater than 0.
        seed : int, default 42
            Seed for the run's random generators, at least 0.
        warmup : float, default 0
            Length of the initial period excluded from measurement, at least 0
            and shorter than duration. The full trace is still kept.
        drain : bool, default False
            Whether to continue past duration until no customers remain.

        Returns
        -------
        Results
            The run's tables, with settings and a summary.

        Example
        -------
        >>> results = model.run(duration=480, seed=7, warmup=60, drain=True)
        """
        self.validate()
        horizon = number(
            value=duration,
            label="Run duration",
            positive=True,
        )
        start = number(warmup, "Warm-up duration")
        if start >= horizon:
            raise ValueError("Warm-up duration must be shorter than run duration.")
        if not isinstance(drain, bool):
            raise ValueError("drain must be True or False.")
        if drain and self._supply is not None:
            raise ValueError(
                "drain serves customers after closing, and a supply chain reviews "
                "only until closing."
            )
        return simulate(
            model=self,
            duration=horizon,
            warmup=start,
            drain=drain,
            seed=integer(
                value=seed,
                label="Seed",
                minimum=0,
            ),
        )

    def experiment(
        self,
        duration: float,
        replications: int = 20,
        seed: int = 42,
        *,
        warmup: float = 0,
        drain: bool = False,
        keep_records: bool = True,
    ) -> Experiment:
        """Run independent replications, keeping each run's raw records by default.

        The seed argument seeds one numpy SeedSequence, which spawns an
        independent seed for each replication. The same seed therefore repeats
        the whole experiment, and no two replications share a stream. Each
        replication is a run with the given duration, warmup, and drain.

        Parameters
        ----------
        duration : float
            Closing time of each run in model time units, greater than 0.
        replications : int, default 20
            Number of independent runs, at least 2.
        seed : int, default 42
            Seed from which each replication's seed is spawned, at least 0.
        warmup : float, default 0
            Initial period excluded from measurement in each run.
        drain : bool, default False
            Whether each run continues past duration until no customers remain.
        keep_records : bool, default True
            Whether each run keeps its raw records (customers, visits, states,
            activities, failures, inventory, and shipments). False drops them
            once the run's measures are computed, which saves memory in long
            or many runs. The experiment's tables, intervals, plots, viewer,
            and export are unchanged, while a single run's plots, viewer, and
            export then raise an error.

        Returns
        -------
        Experiment
            The runs and their replication metrics.

        Example
        -------
        >>> experiment = model.experiment(duration=480, replications=30, seed=1)
        >>> experiment.summary
        """
        if not isinstance(keep_records, bool):
            raise ValueError("keep_records must be True or False.")
        count = integer(
            value=replications,
            label="Replications",
            minimum=2,
        )
        seeds = np.random.SeedSequence(
            integer(
                value=seed,
                label="Seed",
                minimum=0,
            )
        ).spawn(count)
        runs = []
        for child in seeds:
            run = self.run(
                duration=duration,
                seed=int(child.generate_state(1)[0]),
                warmup=warmup,
                drain=drain,
            )
            # Drop each run's records before the next run starts, so memory
            # holds one run's records at a time.
            if not keep_records:
                run._drop_records()
            runs.append(run)
        return Experiment(runs)


# The largest model whose display runs validate(); 200 nodes take about 8 ms.
_DISPLAY_VALIDATE_LIMIT = 200

_ONE_KIND = (
    "A model with a supply chain has no arrival sources, since its stations "
    "serve only the top stage's production jobs."
)
