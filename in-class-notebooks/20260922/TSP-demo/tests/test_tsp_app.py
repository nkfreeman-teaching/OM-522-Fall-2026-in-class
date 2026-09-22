"""Exercise the classroom route flow through Streamlit's test runner."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_default_route_flow() -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "tsp_app.py").run(timeout=30)
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1

    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert len(app.get("plotly_chart")) == 2
    metrics = {metric.label: metric.value for metric in app.metric}
    assert "Nearest-neighbor tour" in metrics
    assert "Improved tour" in metrics
    assert any("saved 717.9 road miles" in item.value for item in app.caption)

    app.multiselect[0].set_value(["AR"])
    app.number_input[0].set_value(2)
    app.number_input[1].set_value(100)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert any("14 facilities across 1 state: AR" in item.value for item in app.caption)
    assert len(app.get("plotly_chart")) == 2
