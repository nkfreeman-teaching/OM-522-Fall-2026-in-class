"""Text and HTML displays shared by models, results, experiments, and comparisons."""

from __future__ import annotations

import difflib
import html
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowsim.results import Settings


# Displays read tables and attributes that already exist. None of them runs a
# model, fits a distribution, or builds a viewer or figure.
def display_text(head: str, rows: list[tuple[str, str]]) -> str:
    """Join a heading and its (label, value) rows into a short text display."""
    lines = [head]
    for label, value in rows:
        lines.append(f"  {label}: {value}" if label else f"    {value}")
    return "\n".join(lines)


def display_html(head: str, rows: list[tuple[str, str]]) -> str:
    """Render the same heading and rows as a compact HTML table."""
    body = "".join(
        '<tr><th style="text-align: left; vertical-align: top; padding-right: 1em;">'
        f"{html.escape(label)}</th><td><code>{html.escape(value)}</code></td></tr>"
        for label, value in rows
    )
    return (
        f"<div><strong>{html.escape(head)}</strong>"
        f'<table style="border-collapse: collapse;">{body}</table></div>'
    )


def figure(value: object) -> str:
    """Format one table value for a display."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def window_line(settings: Settings) -> str:
    """Describe a run window and stopping rule in one line."""
    stopping = "drained" if settings.drain else "not drained"
    return (
        f"{settings.duration:g} {settings.time_unit}, "
        f"warm-up {settings.warmup:g}, {stopping}"
    )


def missing_attribute(
    owner: object,
    name: str,
    hints: dict[str, str],
) -> AttributeError:
    """Build the error for a missing public attribute, with a hint or matches."""
    message = f"{type(owner).__name__!r} object has no attribute {name!r}."
    if name in hints:
        return AttributeError(f"{message} {hints[name]}")
    public = [
        attribute
        for attribute in [*dir(type(owner)), *vars(owner)]
        if not attribute.startswith("_")
    ]
    close = difflib.get_close_matches(
        name,
        public,
        n=3,
    )
    if close:
        message += f" Close matches: {', '.join(close)}."
    return AttributeError(message)
