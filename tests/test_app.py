from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_sections_and_infeasible_price():
    entrypoint = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(entrypoint, default_timeout=120).run()
    assert not app.exception
    for section in ["Valuation", "Homeowner decisions", "Causal repricing",
                    "Instrumental variables", "Profit optimization", "Two-sided extension",
                    "Pipeline and assumptions"]:
        app.sidebar.radio[0].set_value(section).run()
        assert not app.exception
        if section == "Two-sided extension":
            assert len(app.dataframe) >= 7
            app.selectbox[0].set_value(app.selectbox[0].options[1]).run()
            assert not app.exception
    app.sidebar.radio[0].set_value("Profit optimization").run()
    app.slider[2].set_value(1.15).run()
    assert not app.exception
    assert any("No feasible" in message.value for message in app.error)
    app.slider[2].set_value(0.8).run()
    assert len(app.success) == 1
