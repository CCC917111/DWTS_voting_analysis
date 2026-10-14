"""Explicit, provenance-preserving normalization of six season-34 partner labels.

This branch identifies the regular professional dancer for longitudinal history.
It does not allocate week-9 scores between the professional and celebrity guest.
Personal-name aliases are left unchanged unless the separately declared verified
alias stage is explicitly requested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .features import partner_features


POLICY = "explicit_s34_primary_pro_strip_week9_annotation_v1"
CANONICAL_POLICY = "explicit_s34_primary_pro_and_verified_person_aliases_v1"
SOURCE_URL = "https://www.goodmorningamerica.com/amp/culture/story/dancing-stars-season-34-124855780"
VERIFIED_ALIASES = {"Val Chmerkovskiy": "Valentin Chmerkovskiy", "Kym Herjavec": "Kym Johnson"}
ALIAS_SOURCES = {
    "Val Chmerkovskiy": {
        "url": "https://abcnews.com/amp/GMA/Culture/val-chmerkovskiy-talks-dwts-season-31-gabby-windey/story?id=93565359",
        "supporting_fact": "ABC's interview identifies the professional as Valentin (Val) Chmerkovskiy.",
    },
    "Kym Herjavec": {
        "url": "https://abcnews.com/Entertainment/kym-robert-herjavec-announce-expecting-1st-baby/story?id=51726288",
        "supporting_fact": "ABC identifies Kym Herjavec as nee Johnson and describes her season-20 partnership.",
    },
}
# Preserve the source's spelling, including Xoshitl, in the annotation column.
PARTNER_ANNOTATIONS = {
    "Alan Bersten (Rashad Jennings week 9)": ("Alan Bersten", "Rashad Jennings", 9),
    "Daniella Karagach (Rumer Willis week 9)": ("Daniella Karagach", "Rumer Willis", 9),
    "Emma Slater/Kaitlyn Bristowe (week 9)": ("Emma Slater", "Kaitlyn Bristowe", 9),
    "Ezra Sosa (Apolo Anton Ohno week 9)": ("Ezra Sosa", "Apolo Anton Ohno", 9),
    "Val Chmerkovskiy (Joey Graziadei week 9)": ("Val Chmerkovskiy", "Joey Graziadei", 9),
    "Witney Carson (Xoshitl Gomez week 9)": ("Witney Carson", "Xoshitl Gomez", 9),
}


def canonical_primary_partners(rows: pd.DataFrame, include_verified_aliases=False) -> pd.DataFrame:
    """Return an isolated annotated copy; change only six reviewed partner keys.

    An unreviewed annotated string fails closed. Existing provenance columns are
    rejected so a second application cannot erase the original partner label.
    The ordinary Val/Valentin and Kym Johnson/Herjavec aliases remain distinct
    unless include_verified_aliases is explicitly enabled.
    """
    required = {"season", "week", "partner"}
    if required - set(rows):
        raise ValueError(f"Missing partner-identity columns: {sorted(required-set(rows))}")
    new_columns = {"partner_raw", "partner_guest_annotation", "partner_annotation_week",
                   "partner_annotation_this_week", "partner_identity_changed", "partner_id_policy"}
    if new_columns & set(rows):
        raise ValueError("Partner-identity provenance already exists; use the original source")
    if rows.partner.isna().any() or not rows.partner.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise ValueError("Partner identities must be nonempty strings")
    annotated = rows.partner.str.contains(r"[()/]", regex=True)
    unknown = sorted(set(rows.loc[annotated, "partner"]) - set(PARTNER_ANNOTATIONS))
    if unknown:
        raise ValueError(f"Unreviewed partner annotations: {unknown}")
    reviewed = rows.partner.isin(PARTNER_ANNOTATIONS)
    if not rows.loc[reviewed, "season"].eq(34).all():
        raise ValueError("Reviewed week-9 annotation mapping is scoped to season 34")
    d = rows.copy()
    d["partner_raw"] = d.partner
    d["partner_guest_annotation"] = ""
    d["partner_annotation_week"] = pd.Series(pd.NA, index=d.index, dtype="Int64")
    d["partner_identity_changed"] = False
    for raw, (primary, guest, week) in PARTNER_ANNOTATIONS.items():
        mask = d.partner_raw.eq(raw)
        d.loc[mask, "partner"] = primary
        d.loc[mask, "partner_guest_annotation"] = guest
        d.loc[mask, "partner_annotation_week"] = week
        d.loc[mask, "partner_identity_changed"] = True
    d["partner_annotation_this_week"] = d.week.eq(d.partner_annotation_week).fillna(False).astype(bool)
    if include_verified_aliases:
        d["partner"] = d.partner.replace(VERIFIED_ALIASES)
        d["partner_identity_changed"] = d.partner.ne(d.partner_raw)
    d["partner_id_policy"] = CANONICAL_POLICY if include_verified_aliases else POLICY
    return d


def feature_endpoints_fast(rows, reference):
    """Vectorized audit reference for the baseline feature formula, not a model.

    Only for unique pro-week data and complete season folds. Caller compares this
    reference with partner_features before relying on split-difference summaries.
    """
    if rows.duplicated(["season", "week", "partner"]).any():
        raise ValueError("Audit reference requires unique pro-week rows")
    def normalized(d):
        x = d[["season", "week", "partner"]].copy()
        x["pq_judge"] = d.judge_mean / d.groupby(["season", "week"]).judge_mean.transform("mean")
        x["pq_vote"] = d.fan_mean * d.n_active
        return x
    d, ref = normalized(rows), normalized(reference)
    d = d.sort_values(["season", "partner", "week"])
    cols = ["pq_judge", "pq_vote"]
    grouped = d.groupby(["season", "partner"])
    prior_sums = grouped[cols].cumsum() - d[cols]
    prior_counts = grouped.cumcount()
    within = prior_sums.div(prior_counts.replace(0, np.nan), axis=0).fillna(1.)
    history = within.copy()
    for season in d.season.unique():
        old = ref[ref.season < season].groupby("partner")[cols].mean()
        mask = d.season.eq(season) & d.partner.isin(old.index)
        history.loc[mask, cols] = old.reindex(d.loc[mask, "partner"]).to_numpy()
    return within.reindex(rows.index), history.reindex(rows.index)


def split_feature_audit(raw, canonical, folds=5, inner_folds=3):
    """Identify every changed outer/inner design at each predeclared weight."""
    from .attribution import _designs
    outer = list(GroupKFold(min(folds, raw.season.nunique())).split(raw, groups=raw.season))
    specifications = []
    for outer_fold, (tr, te) in enumerate(outer):
        specifications.append((outer_fold, -1, tr, te))
        training = raw.iloc[tr]
        for inner_fold, (itr, ite) in enumerate(GroupKFold(min(inner_folds, training.season.nunique())).split(training, groups=training.season)):
            specifications.append((outer_fold, inner_fold, tr[itr], tr[ite]))
    summary, changed_rows = [], []
    # One full outer and one inner split are checked against production designs,
    # including the preprocessing outputs; no estimator is fitted here.
    validations = []
    for outer_fold, inner_fold, tr, te in specifications:
        endpoints = []
        for split, idx in (("train", tr), ("test", te)):
            a0, a1 = feature_endpoints_fast(raw.iloc[idx], raw.iloc[tr])
            b0, b1 = feature_endpoints_fast(canonical.iloc[idx], canonical.iloc[tr])
            endpoints.append((a0, a1, b0, b1))
            delta0, delta1 = b0-a0, b1-a1
            # Group membership in a season must be unchanged by these mappings.
            if not np.allclose(delta0, 0, rtol=0, atol=1e-12):
                raise ValueError("Unexpected within-season history change")
            for weight in (0., .25, .5, .75, 1.):
                delta = (1-weight)*delta0 + weight*delta1
                changed = delta.abs().gt(1e-12).any(axis=1)
                summary.append({"outer_fold": outer_fold, "inner_fold": inner_fold,
                                "split": split, "history_weight": weight, "rows": len(idx),
                                "changed_rows": int(changed.sum()),
                                "max_abs_change": float(np.abs(delta.to_numpy()).max()),
                                "design_equal_at_1e_12": not bool(changed.any())})
                if weight == 1.:
                    for i in delta.index[changed]:
                        changed_rows.append({"outer_fold": outer_fold, "inner_fold": inner_fold,
                                             "split": split, "source_row": int(i),
                                             "season": int(raw.at[i, "season"]), "week": int(raw.at[i, "week"]),
                                             "contestant_id": raw.at[i, "contestant_id"],
                                             "partner_raw": raw.at[i, "partner"], "partner_canonical": canonical.at[i, "partner"],
                                             "delta_pq_judge_at_weight1": delta.at[i, "pq_judge"],
                                             "delta_pq_vote_at_weight1": delta.at[i, "pq_vote"]})
        if outer_fold == 0 and inner_fold in (-1, 0):
            for dataset, offset in ((raw, 0), (canonical, 2)):
                a, b, _ = _designs(dataset, tr, te, .5, {})
                for split, actual, ep in (("train", a, endpoints[0]), ("test", b, endpoints[1])):
                    expected = .5*ep[offset] + .5*ep[offset+1]
                    error = float(np.abs(actual[:, 1:3]-expected.to_numpy()).max())
                    if error > 1e-12:
                        raise AssertionError(f"Audit reference differs from production features: {error}")
                    validations.append({"outer_fold": outer_fold, "inner_fold": inner_fold,
                                        "dataset": "raw" if offset == 0 else "canonical", "split": split,
                                        "max_abs_error": error, "tolerance": 1e-12})
    return pd.DataFrame(summary), pd.DataFrame(changed_rows), pd.DataFrame(validations)


def _changed_features(raw, canonical, reference_raw, reference_canonical, fold, split):
    """Record feature endpoint changes without fitting any statistical model."""
    records = []
    for weight in (0., 1.):
        a = partner_features(raw, reference=reference_raw, history_weight=weight)
        b = partner_features(canonical, reference=reference_canonical, history_weight=weight)
        delta = b[["pq_judge", "pq_vote"]] - a[["pq_judge", "pq_vote"]]
        changed = delta.abs().gt(1e-12).any(axis=1)
        for idx in a.index[changed]:
            records.append({"source_row": int(idx), "fold": fold, "split": split,
                            "history_weight": weight, "season": int(a.at[idx, "season"]),
                            "week": int(a.at[idx, "week"]), "contestant_id": a.at[idx, "contestant_id"],
                            "partner_raw": a.at[idx, "partner"], "partner_canonical": b.at[idx, "partner"],
                            "pq_judge_raw": a.at[idx, "pq_judge"], "pq_judge_canonical": b.at[idx, "pq_judge"],
                            "pq_vote_raw": a.at[idx, "pq_vote"], "pq_vote_canonical": b.at[idx, "pq_vote"],
                            "delta_pq_judge": delta.at[idx, "pq_judge"], "delta_pq_vote": delta.at[idx, "pq_vote"]})
    return records


def prepare_identity_audit(source, out, include_verified_aliases=False):
    """Create canonical input and auditable feature deltas, never RF predictions."""
    source, out = Path(source), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty identity-audit output directory")
    votes_path = source / "votes.csv" if source.is_dir() else source
    raw = pd.read_csv(votes_path).reset_index(drop=True)
    canonical = canonical_primary_partners(raw, include_verified_aliases=include_verified_aliases)
    keys = ["season", "week", "contestant_id"]
    if raw.duplicated(keys).any() or canonical.duplicated(["season", "week", "partner"]).any():
        raise ValueError("Identity audit requires unique contestant keys and canonical pro-week keys")
    pd.testing.assert_frame_equal(raw.drop(columns="partner"), canonical[raw.columns].drop(columns="partner"))
    out.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(out / "votes_canonical_primary.csv", index=False)
    affected = canonical.loc[canonical.partner_identity_changed, keys + [
        "partner_raw", "partner", "partner_guest_annotation", "partner_annotation_week",
        "partner_annotation_this_week", "partner_id_policy"]]
    affected.to_csv(out / "partner_identity_rows.csv", index=False)
    mapping = affected.groupby(["partner_raw", "partner", "partner_guest_annotation", "partner_annotation_week"], as_index=False, dropna=False).agg(
        rows=("week", "size"), min_week=("week", "min"), max_week=("week", "max"))
    def sources_for(raw_name):
        urls = [SOURCE_URL] if raw_name in PARTNER_ANNOTATIONS else []
        primary = PARTNER_ANNOTATIONS.get(raw_name, (raw_name, None, None))[0]
        if include_verified_aliases and primary in ALIAS_SOURCES:
            urls.append(ALIAS_SOURCES[primary]["url"])
        return " ; ".join(urls)
    mapping["source_url"] = mapping.partner_raw.map(sources_for)
    mapping.to_csv(out / "partner_identity_mapping.csv", index=False)
    summary, changes_frame, validation = split_feature_audit(raw, canonical)
    summary.to_csv(out / "split_design_changes.csv", index=False)
    changes_frame.to_csv(out / "feature_endpoint_changes.csv", index=False)
    validation.to_csv(out / "fast_reference_validation.csv", index=False)
    # Preserve the existing selected weights as a diagnostic only. A downstream
    # nested-CV experiment must select weights again on canonical training data.
    selection_path = votes_path.parent / "history_weight_selection.csv"
    if selection_path.exists():
        selection = pd.read_csv(selection_path)
        frozen = changes_frame[changes_frame.inner_fold.eq(-1)].merge(
            selection[["task", "outer_fold", "history_weight"]].rename(columns={"history_weight": "selected_weight"}),
            on="outer_fold", validate="many_to_many")
        frozen["delta_pq_judge"] = frozen.delta_pq_judge_at_weight1 * frozen.selected_weight
        frozen["delta_pq_vote"] = frozen.delta_pq_vote_at_weight1 * frozen.selected_weight
        frozen = frozen.loc[frozen[["delta_pq_judge", "delta_pq_vote"]].abs().gt(1e-12).any(axis=1)]
        frozen[["task", "outer_fold", "split", "selected_weight", "source_row"] + keys +
               ["partner_raw", "partner_canonical", "delta_pq_judge", "delta_pq_vote"]].to_csv(
            out / "frozen_selection_feature_changes.csv", index=False)
    manifest = {
        "policy": CANONICAL_POLICY if include_verified_aliases else POLICY, "source_votes": str(votes_path),
        "source_sha256": hashlib.sha256(votes_path.read_bytes()).hexdigest(),
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "canonical_votes_sha256": hashlib.sha256((out / "votes_canonical_primary.csv").read_bytes()).hexdigest(),
        "source_rows": len(raw), "changed_identity_rows": len(affected),
        "raw_partner_keys": raw.partner.nunique(), "canonical_partner_keys": canonical.partner.nunique(),
        "week9_annotation_rows": int(canonical.partner_annotation_this_week.sum()),
        "model_fits_performed": 0, "outcomes_and_row_order_unchanged": True,
        "aliases_merged": include_verified_aliases,
        "alias_sources": ALIAS_SOURCES if include_verified_aliases else {},
        "notes": ["Partner refers to the regular primary pro across the season; guest score contributions are not reassigned.",
                  "Six reviewed season-34 annotations are changed; verified aliases are merged only in the explicitly enabled branch.",
                  "Five GroupKFold season splits reproduce the baseline splitter; endpoint deltas scale linearly with history weight.",
                  "Frozen-selection deltas are diagnostic; rerun inner CV before presenting corrected model performance.",
                  "This is a reconstruction identity correction, not evidence of the report authors' preprocessing."],
        "primary_source": SOURCE_URL,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/report_verified")
    parser.add_argument("--out", default="outputs/v5_partner_identity")
    parser.add_argument("--verified-aliases", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_identity_audit(args.source, args.out, args.verified_aliases), indent=2))
