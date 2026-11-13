"""An offline page that draws a small service network and writes its FlowSim code.

The page's argument forms come from the Python signatures and docstrings of the
distribution constructors and of ``Model.source``, ``Model.station``, and
``Model.run``, so the page cannot drift from the API it writes. The page only
writes constructor calls; every draw stays in Python.
"""

from __future__ import annotations

import ast
import builtins
import collections.abc
import inspect
import json
import keyword
import re
import types
import typing
from pathlib import Path

from flowsim import __version__
from flowsim import distributions as _distributions
from flowsim.fitting import COUNT_FAMILIES
from flowsim.model import BRANCH_SUM_TOLERANCE, Model
from flowsim.viewer import Viewer

# The version of the page's saved JSON model format. Version 2 added customer
# types, schedules, buffers, abandonment, failures, and priorities; the page
# still reads version 1 files.
FORMAT_VERSION = 2

# The arguments the page edits: every argument of Model.source and
# Model.station. An argument added to either later is listed on the page as
# not edited until the page handles it.
SUPPORTED = {
    "source": ("name", "interarrival", "first_arrival", "limit", "rates", "types"),
    "station": (
        "name",
        "capacity",
        "service_time",
        "staffing",
        "shift_end",
        "buffer",
        "external_overflow",
        "time_to_failure",
        "repair_time",
        "failure_clock",
        "priority",
        "patience",
        "abandon_from",
    ),
}


def _parameter_docs(obj: object) -> dict[str, str]:
    """Return each parameter's description from a NumPy-style docstring."""
    doc = inspect.getdoc(obj) or ""
    match = re.search(
        pattern=r"Parameters\n-+\n(.*?)(?:\n\n[A-Z][a-z]+\n-+\n|\Z)",
        string=doc,
        flags=re.DOTALL,
    )
    if match is None:
        return {}
    described: dict[str, list[str]] = {}
    current = None
    for line in match.group(1).splitlines():
        heading = re.match(r"^(\w+) : ", line)
        if heading is not None:
            current = heading.group(1)
            described[current] = []
        elif current is not None and line.strip():
            described[current].append(line.strip())
    return {name: " ".join(lines) for name, lines in described.items()}


def _bounds(description: str) -> dict:
    """Read a lower bound stated in a parameter description, if any."""
    if re.search(r"greater than (zero|0)\b|\bpositive\b", description):
        return {"minimum": 0, "exclusive": True}
    match = re.search(r"at least (zero|\d+)", description)
    if match is not None:
        value = 0 if match.group(1) == "zero" else int(match.group(1))
        return {"minimum": value, "exclusive": False}
    return {}


def _members(annotation: object) -> list[object]:
    """Return the members of a union, or the annotation alone."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return [
            member for member in typing.get_args(annotation) if member is not type(None)
        ]
    return [annotation]


def _kind(annotation: object) -> tuple[str, list | None]:
    """Classify an annotation as a page field kind, with any Literal choices."""
    members = _members(annotation)

    def callable_member(member: object) -> bool:
        if typing.get_origin(member) is collections.abc.Callable:
            return True
        return any(callable_member(inner) for inner in typing.get_args(member))

    if any(callable_member(member) for member in members):
        return "duration", None
    if len(members) == 1 and typing.get_origin(members[0]) is typing.Literal:
        return "choice", list(typing.get_args(members[0]))
    kinds = {bool: "boolean", int: "integer", float: "number", str: "text"}
    if len(members) == 1 and members[0] in kinds:
        return kinds[members[0]], None
    return "other", None


def _parameters(
    function: types.FunctionType, skip: tuple[str, ...] = ("self",)
) -> list[dict]:
    """Describe each parameter: name, kind, default, choices, and bounds."""
    described = _parameter_docs(function)
    result = []
    for name, parameter in inspect.signature(
        function, eval_str=True
    ).parameters.items():
        if name in skip:
            continue
        kind, choices = _kind(parameter.annotation)
        required = parameter.default is inspect.Parameter.empty
        description = described.get(name, "")
        entry = {
            "name": name,
            "kind": kind,
            "required": required,
            "default": None if required else parameter.default,
            "description": description,
            **_bounds(description),
        }
        if choices is not None:
            entry["choices"] = choices
        result.append(entry)
    return result


def _example(function: types.FunctionType) -> dict:
    """Return the keyword values of the first call in the docstring's Example."""
    doc = inspect.getdoc(function) or ""
    for line in doc.splitlines():
        if line.startswith(f">>> {function.__name__}("):
            call = _call_keywords(line.removeprefix(">>> "))
            if call is not None:
                return call
    return {}


def _call_keywords(source: str) -> dict | None:
    """Return the literal keyword arguments of one call expression, skipping the rest."""
    try:
        node = ast.parse(source, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call):
        return None
    keywords = {}
    for item in node.keywords:
        if item.arg is None:
            continue
        try:
            keywords[item.arg] = ast.literal_eval(item.value)
        except (ValueError, TypeError):
            continue
    return keywords


def _duration_constructors() -> list[types.FunctionType]:
    """Return the named duration constructors, in definition order.

    A duration constructor is a public function in flowsim.distributions that
    returns a Distribution and takes only keyword numbers. Count families
    (demand), sequence-based constructors, and shifted are left out.
    """
    found = []
    for name, function in vars(_distributions).items():
        if name.startswith("_") or not inspect.isfunction(function):
            continue
        if function.__module__ != _distributions.__name__ or name in COUNT_FAMILIES:
            continue
        signature = inspect.signature(function, eval_str=True)
        returned = signature.return_annotation
        if not (
            inspect.isclass(returned)
            and issubclass(returned, _distributions.Distribution)
        ):
            continue
        parameters = signature.parameters.values()
        if all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and parameter.annotation in (float, int)
            for parameter in parameters
        ):
            found.append(function)
    return found


def schema() -> dict:
    """Return the argument schema that the builder page embeds.

    Returns
    -------
    dict
        The keys ``distributions`` (each named duration constructor with its
        parameters and docstring example), ``model``, ``source``, ``station``,
        and ``run`` (parameters of ``Model``, ``Model.source``,
        ``Model.station``, and ``Model.run``), ``supported`` (the arguments the
        page edits), ``reserved`` (names a generated variable must avoid),
        ``branch_sum_tolerance``, ``format_version``, and ``flowsim_version``.

    Example
    -------
    >>> [entry["name"] for entry in schema()["distributions"]][:2]
    ['exponential', 'uniform']
    """
    constructors = _duration_constructors()
    distributions = [
        {
            "name": function.__name__,
            "summary": (inspect.getdoc(function) or "").splitlines()[0],
            "parameters": _parameters(function),
            "example": _example(function),
        }
        for function in constructors
    ]
    reserved = sorted(
        {
            *keyword.kwlist,
            *keyword.softkwlist,
            *dir(builtins),
            "Model",
            "model",
            "results",
            *(function.__name__ for function in constructors),
        }
    )
    return {
        "format_version": FORMAT_VERSION,
        "flowsim_version": __version__,
        "distributions": distributions,
        "model": _parameters(Model.__init__),
        "source": _parameters(Model.source),
        "station": _parameters(Model.station),
        "run": _parameters(Model.run),
        "supported": {kind: list(names) for kind, names in SUPPORTED.items()},
        "reserved": reserved,
        "branch_sum_tolerance": BRANCH_SUM_TOLERANCE,
    }


def builder_html() -> str:
    """Return the builder page with the schema embedded."""
    # Escape '<' so docstring text cannot end the embedded JSON script.
    data = json.dumps(schema(), allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("builder.html").read_text(encoding="utf-8")
    return template.replace("__SCHEMA_JSON__", data)


def builder(path: str | Path | None = None) -> Viewer:
    """Open a page that draws a service network and writes its FlowSim code.

    The page runs offline in a browser. It adds sources, stations, and the
    exit to a canvas, connects them with routes and probabilistic branches,
    and edits every argument of Model.source and Model.station: durations
    (constants or named distributions), customer types and their routes,
    arrival-rate and staffing schedules, finite buffers, abandonment,
    failures, and priorities. It never chooses a behavior argument (shift_end,
    external_overflow, abandon_from, or failure_clock) for the student, and
    lists each missing choice under Checks. It writes a complete FlowSim
    script that it can copy or download, saves and loads its own JSON model
    file, and supports the keyboard and undo. Supply chains are not in the
    page, so they are added by editing the script.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        File to save the page to. Without it, nothing is written.

    Returns
    -------
    Viewer
        The page. A notebook displays it in an isolated frame, and a script
        calls ``.open()`` to show it in a browser.

    Example
    -------
    >>> page = flowsim.builder("output/builder.html")
    >>> page.open()
    """
    page = Viewer(
        document=builder_html(),
        kind="model builder page",
        name="builder",
        height=1100,
    )
    if path is not None:
        page.save(path)
    return page
