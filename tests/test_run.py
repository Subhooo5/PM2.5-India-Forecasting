import pytest

import run


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(run, "STAGES", {name: (lambda cfg, n=name: seen.append(n)) for name in run.STAGES})
    monkeypatch.setattr(run.stress, "write_manifest", lambda cfg: seen.append("manifest"))
    return seen


def test_stage_order():
    assert list(run.STAGES) == ["audit", "preprocess", "features", "tuning", "leakage", "cross_city", "uncertainty",
                                "stress", "figures"]


@pytest.mark.parametrize("argv", [[], ["all"]])
def test_all_runs_every_stage_in_order_then_manifest(calls, argv):
    run.main(argv, cfg={})
    assert calls == list(run.STAGES) + ["manifest"]


def test_named_stages_run_alone_without_manifest(calls):
    assert run.main(["figures", "leakage"], cfg={}) == ["figures", "leakage"]
    assert calls == ["figures", "leakage"]


def test_unknown_stage_is_rejected(calls):
    with pytest.raises(SystemExit, match="unknown stage"):
        run.main(["nope"], cfg={})
    assert calls == []
