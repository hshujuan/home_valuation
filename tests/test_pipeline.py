import json

from home_valuation.pipeline import save_study


def test_pipeline_artifacts(study, tmp_path):
    save_study(study, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "market_sales.csv" in manifest["artifacts"]
    assert len(manifest["artifacts"]["market_sales.csv"]) == 64
    assert (tmp_path / "evaluator_truth" / "resale.csv").exists()
    assert (tmp_path / "figures" / "selection.png").exists()
    assert "Synthetic" in (tmp_path / "reference_results.md").read_text() or (
        "synthetic" in (tmp_path / "reference_results.md").read_text())
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["schema_version"] == 1
