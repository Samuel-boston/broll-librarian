"""M1 acceptance gate.

`broll analyse <clip>` must produce output that validates against
AnalysisResult with every controlled-vocabulary field in-vocabulary, for every
fixture clip.

By default this runs against the mock provider so it is offline and free, which
exercises the harness but not the model. To run the real gate:

    BROLL_ACCEPTANCE_PROVIDER=gemini GEMINI_API_KEY=... pytest tests/test_m1_acceptance.py -s
"""

from __future__ import annotations

import json
import os

import pytest
from typer.testing import CliRunner

from broll.analysis.schema import AnalysisResult, find_oov
from broll.cli import app

runner = CliRunner()

PROVIDER = os.environ.get("BROLL_ACCEPTANCE_PROVIDER", "mock")
MODEL = os.environ.get("BROLL_ACCEPTANCE_MODEL")


@pytest.fixture(scope="function")
def acceptance_workspace(broll_home, monkeypatch):
    monkeypatch.setenv("BROLL_HOME", str(broll_home))
    args = ["init", "--name", "Acceptance", "--provider", PROVIDER]
    if MODEL:
        args += ["--model", MODEL]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return "acceptance"


def test_every_fixture_clip_analyses_in_vocabulary(acceptance_workspace, clips, capsys):
    assert len(clips) >= 5, f"expected at least 5 fixture clips, found {len(clips)}"

    failures: list[str] = []
    report: list[str] = []

    for name, path in sorted(clips.items()):
        result = runner.invoke(app, ["analyse", str(path), "--compact"])
        if result.exit_code != 0:
            failures.append(f"{name}: CLI failed - {result.output.strip()[:200]}")
            continue

        payload = json.loads(result.output)
        analysis = AnalysisResult.model_validate(payload["analysis"])
        oov = find_oov(analysis)
        if oov:
            failures.append(f"{name}: out of vocabulary {oov}")

        report.append(
            f"{name:<26} {analysis.shot_type.value:<14} {analysis.camera_movement.value:<14}"
            f" conf={analysis.confidence:.2f} ${payload['estimated_cost_usd']:.5f}"
            f"  {analysis.caption[:70]}"
        )

    with capsys.disabled():
        print(f"\nM1 acceptance - provider={PROVIDER} model={MODEL or 'default'}")
        for line in report:
            print("  " + line)

    assert not failures, "\n".join(failures)
