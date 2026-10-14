"""Scope changes and reuse guards; these tests do not fit posterior models."""
from pathlib import Path
import importlib.util
import shutil

import numpy as np
import pandas as pd
import pytest

from dwts.data import prepare

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/2026_MCM_Problem_C_Data.csv"
BASE = ROOT / "outputs/report_final"
spec = importlib.util.spec_from_file_location("scope_runner", ROOT / "scripts/run_scope_candidates.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def baseline():
    return prepare(RAW, outcome_policy="label_first", include_final_week_eliminations=False)[0]


def test_default_option_preserves_legacy_panel_and_audit():
    before, old_audit = prepare(RAW)
    after, new_audit = prepare(RAW, exclude_withdrawn_contestants=False)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    pd.testing.assert_frame_equal(old_audit, new_audit, check_exact=True)


def test_zero_first_changes_only_s18_diana_constraint():
    old = baseline()
    new, _ = prepare(RAW, **runner.PROTOCOLS["candidate_zero_first"]["data"])
    changed = [s for s in range(1, 35) if not runner.inputs_equal(old[old.season == s], new[new.season == s])]
    assert changed == [18]
    differences = old.eliminated != new.eliminated
    assert new.loc[differences, ["season", "week", "name", "eliminated"]].values.tolist() == [[18, 1, "Diana Nyad", True]]
    pd.testing.assert_frame_equal(old.drop(columns="eliminated"), new.drop(columns="eliminated"), check_exact=True)


def test_exclusion_removes_ten_contestants_in_nine_seasons_before_normalization():
    old = baseline()
    new, audit = prepare(RAW, **runner.PROTOCOLS["candidate_exclude_withdrawn"]["data"])
    changed = [s for s in range(1, 35) if not runner.inputs_equal(old[old.season == s], new[new.season == s])]
    assert changed == [3, 7, 9, 16, 18, 21, 28, 29, 31]
    assert len(new) == 2738 and len(old) - len(new) == 39
    assert old.contestant_id.nunique() - new.contestant_id.nunique() == 10
    assert not new.result_label.str.contains("withdrew", case=False).any()
    assert (audit.issue == "withdrawn_contestant_excluded_before_normalization").sum() == 10
    groups = new.groupby(["season", "week"])
    np.testing.assert_allclose(groups.judge_share.sum(), 1, rtol=0, atol=1e-14)
    np.testing.assert_allclose(groups.judge_z.mean(), 0, rtol=0, atol=1e-12)
    assert (groups.n_active.first() == groups.size()).all()
    merged = new.merge(old, on=runner.KEYS, suffixes=("_new", "_old"))
    changed_numeric = (merged.judge_z_new != merged.judge_z_old) | (merged.judge_share_new != merged.judge_share_old)
    assert changed_numeric.sum() == 389
    counts = groups.eliminated.any()
    assert [int(counts[[key[0] in region for key in counts.index]].sum())
            for region in [range(1, 3), range(3, 28), range(28, 35)]] == [11, 192, 56]


def test_exact_reuse_comparison_rejects_one_ulp_change():
    frame = baseline().query("season == 1").copy()
    modified = frame.copy()
    ix = modified.index[0]
    modified.loc[ix, "judge_z"] = np.nextafter(modified.loc[ix, "judge_z"], np.inf)
    assert not runner.inputs_equal(frame, modified)
    with pytest.raises(ValueError, match="consistency failed"):
        runner.assert_inputs_match(frame, modified, "one ULP", exact=True)


def test_preflight_accepts_frozen_original_season_inputs():
    _, config, canonical, votes = runner.preflight(RAW, BASE)
    assert len(canonical) == len(votes) == 2777
    assert config["fit"]["draws"] == 4000


def test_preflight_refuses_modified_raw(tmp_path):
    modified = tmp_path / "raw.csv"
    modified.write_bytes(RAW.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Raw input hash"):
        runner.preflight(modified, BASE)


def test_preflight_refuses_changed_original_vote_input(tmp_path):
    copied = tmp_path / "base"
    shutil.copytree(BASE, copied)
    path = copied / "season_01/votes.csv"
    votes = pd.read_csv(path, float_precision="round_trip")
    votes.loc[0, "judge_share"] += 0.01
    votes.to_csv(path, index=False)
    with pytest.raises(ValueError, match="baseline season 1"):
        runner.preflight(RAW, copied)


def test_runner_refuses_existing_output_without_mutation(tmp_path):
    existing = tmp_path / "candidate_zero_first"
    existing.mkdir()
    marker = existing / "manifest.json"
    marker.write_text('{"status":"running","belongs_to":"previous_run"}')
    before = marker.read_bytes()
    with pytest.raises(ValueError, match="Preserve existing results"):
        runner.run(RAW, BASE, tmp_path)
    assert marker.read_bytes() == before
    assert not (tmp_path / "candidate_exclude_withdrawn").exists()
