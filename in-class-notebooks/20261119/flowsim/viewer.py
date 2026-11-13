"""An offline report page that notebooks display and scripts save or open."""

from __future__ import annotations

import html
import webbrowser
from pathlib import Path

# An unsaved page's open() writes output/flowsim-<name>.html under the working
# directory, so every file FlowSim writes on its own sits in one folder.
OPEN_FOLDER = Path("output")


class Viewer:
    """An offline HTML report page.

    A notebook displays the page in an isolated frame, so the page's scripts
    cannot reach the notebook. A script saves it with ``save(path)`` or shows
    it with ``open()``, and ``html`` holds the complete document.

    Example
    -------
    >>> viewer = results.viewer()
    >>> viewer.save("output/viewer.html")
    """

    def __init__(self, document: str, kind: str, name: str, height: int):
        self._document = document
        self._kind = kind
        # The file stem an unsaved page's open() writes, e.g., "results".
        self._name = name
        # The frame height in a notebook, in pixels. The page scrolls inside it.
        self._height = height
        self._path: Path | None = None
        self._saved: Path | None = None

    @property
    def html(self) -> str:
        """Return the complete HTML document."""
        return self._document

    def save(self, path: str | Path) -> Path:
        """Write the page to a file, creating its folder, and return the path.

        Parameters
        ----------
        path : str or pathlib.Path
            File to write. Its folder is created if needed.

        Returns
        -------
        pathlib.Path
            The file written.

        Example
        -------
        >>> viewer.save("output/viewer.html")
        """
        destination = Path(path)
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        destination.write_text(
            data=self._document,
            encoding="utf-8",
        )
        self._path = destination
        # Resolved now, so open() writes this file after a change of working
        # directory.
        self._saved = destination.resolve()
        return destination

    def open(self) -> Path:
        """Show the page in the default web browser and return its file.

        A saved page is written again to its saved path. A page that was never
        saved is written to output/flowsim-<name>.html under the current working
        directory (e.g., output/flowsim-results.html), which each call replaces.
        The method prints the file's path.

        Returns
        -------
        pathlib.Path
            The file shown, as an absolute path for a page that was never saved.

        Example
        -------
        >>> viewer.open()
        """
        if self._path is None or self._saved is None:
            # Resolved once, so the returned path survives a later change of
            # working directory.
            path = target = (OPEN_FOLDER / f"flowsim-{self._name}.html").resolve()
        else:
            path, target = self._path, self._saved
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        target.write_text(
            data=self._document,
            encoding="utf-8",
        )
        print(f"Opening {target}")
        webbrowser.open(target.as_uri())
        return path

    def _repr_html_(self) -> str:
        # srcdoc holds the escaped document, and the sandbox without
        # allow-same-origin gives the page its own origin.
        return (
            f'<iframe srcdoc="{html.escape(self._document, quote=True)}" '
            'sandbox="allow-scripts allow-downloads" '
            f'style="width: 100%; height: {self._height}px; '
            'border: 1px solid #dce3e8;"></iframe>'
        )

    def __repr__(self) -> str:
        size = len(self._document.encode("utf-8")) / 1_000_000
        return (
            f"<Viewer: {self._kind}, {size:.1f} MB. Display it in a notebook, "
            "or call .save(path) or .open(). The document is in .html.>"
        )
