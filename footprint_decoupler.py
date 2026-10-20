#!/usr/bin/env python3
"""
footprint_decoupler.py
======================

Model the environmental "blind spot" in which spending-based (EEIO) carbon
accounting diverges from a quantity/quality-based (process-LCA style) account
of the same household transactions.

Core identity exploited by this tool
------------------------------------
For a transaction with physical quantity q, unit price p and category c:

    EEIO  = I_c * (q * p)                            [kg CO2e]
    LCA   = q * m_c * EF_c * mu_tier * D(theta)      [kg CO2e]

where I_c is the sector carbon intensity (kg CO2e / EUR), m_c the mass per
unit (kg), EF_c the mass-based emission factor (kg CO2e / kg), mu_tier a
material multiplier for the quality tier and D(theta) = 1 / (1 + g_c * theta)
an inverse-quality (durability) coefficient driven by a continuous quality
index theta in [0, 1].

Because quantity q cancels in the ratio,

    ln(EEIO / LCA) = ln(I_c / (m_c * EF_c))          (category constant)
                   + ln(p)                           (price effect)
                   - ln(mu_tier * D(theta))          (quality effect)

so every divergence between the two models is a price-per-physical-unit or a
quality effect -- never a volume effect. The decomposition module estimates
the share of residual variance attributable to each component and locates the
spending threshold at which EEIO switches from under- to over-estimation.

All default parameters are illustrative, internally consistent placeholders
for method development. They are NOT calibrated to EXIOBASE, USEEIO or any
LCA database and must be replaced before any empirical use.

Examples
--------
    python footprint_decoupler.py --records 2000 --plot
    python footprint_decoupler.py --records 1000 --export-csv ledger.csv --seed 7
    python footprint_decoupler.py --input synthetic_schema_seed.csv --plot
    python footprint_decoupler.py --records 5000 --calibrate-eeio --plot
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

TIERS: Tuple[str, str] = ("budget", "premium")


@dataclass(frozen=True)
class CategoryParams:
    """Illustrative physical and economic parameters for one product category.

    Attributes
    ----------
    eeio_intensity : float
        Spending-based carbon intensity, kg CO2e per EUR.
    mass_per_unit : float
        Mass of one budget-tier physical unit, kg.
    mass_emission_factor : float
        Cradle-to-gate emission factor per kg of product, kg CO2e / kg.
    premium_share : float
        Probability that a transaction is a premium-tier purchase.
    budget_unit_price : float
        Median unit price for the budget tier, EUR.
    premium_unit_price : float
        Median unit price for the premium tier, EUR.
    price_sigma : float
        Log-normal dispersion (sigma of ln price) within a tier.
    budget_qty_shape, budget_qty_scale : float
        Gamma parameters for budget-tier quantity per transaction.
    premium_qty_shape, premium_qty_scale : float
        Gamma parameters for premium-tier quantity per transaction.
    premium_material_multiplier : float
        Extra embodied material/processing per premium unit (>= 1).
    durability_gain : float
        Strength g of the inverse-quality coefficient D = 1 / (1 + g * theta).
    """

    eeio_intensity: float
    mass_per_unit: float
    mass_emission_factor: float
    premium_share: float
    budget_unit_price: float
    premium_unit_price: float
    price_sigma: float
    budget_qty_shape: float
    budget_qty_scale: float
    premium_qty_shape: float
    premium_qty_scale: float
    premium_material_multiplier: float
    durability_gain: float


CATEGORY_PARAMS: Dict[str, CategoryParams] = {
    "Apparel": CategoryParams(
        eeio_intensity=0.40, mass_per_unit=0.45, mass_emission_factor=25.0,
        premium_share=0.35, budget_unit_price=12.0, premium_unit_price=95.0,
        price_sigma=0.35, budget_qty_shape=2.5, budget_qty_scale=2.2,
        premium_qty_shape=1.2, premium_qty_scale=0.9,
        premium_material_multiplier=1.20, durability_gain=2.0,
    ),
    "Electronics": CategoryParams(
        eeio_intensity=0.30, mass_per_unit=1.2, mass_emission_factor=60.0,
        premium_share=0.40, budget_unit_price=60.0, premium_unit_price=650.0,
        price_sigma=0.40, budget_qty_shape=1.8, budget_qty_scale=1.5,
        premium_qty_shape=1.0, premium_qty_scale=0.6,
        premium_material_multiplier=1.35, durability_gain=1.5,
    ),
    "Groceries": CategoryParams(
        eeio_intensity=0.90, mass_per_unit=1.0, mass_emission_factor=2.8,
        premium_share=0.30, budget_unit_price=2.2, premium_unit_price=6.5,
        price_sigma=0.30, budget_qty_shape=6.0, budget_qty_scale=3.0,
        premium_qty_shape=3.0, premium_qty_scale=2.5,
        premium_material_multiplier=1.05, durability_gain=0.3,
    ),
    "Furniture": CategoryParams(
        eeio_intensity=0.35, mass_per_unit=18.0, mass_emission_factor=3.5,
        premium_share=0.30, budget_unit_price=80.0, premium_unit_price=700.0,
        price_sigma=0.45, budget_qty_shape=1.5, budget_qty_scale=1.4,
        premium_qty_shape=1.0, premium_qty_scale=0.5,
        premium_material_multiplier=1.40, durability_gain=2.5,
    ),
}

# Beta distributions for the continuous quality index theta within each tier.
QUALITY_BETA: Dict[str, Tuple[float, float]] = {
    "budget": (2.0, 6.0),   # mean 0.25
    "premium": (6.0, 2.0),  # mean 0.75
}

# Quality index assigned when an input ledger carries a tier but no index.
DEFAULT_QUALITY_INDEX: Dict[str, float] = {"budget": 0.25, "premium": 0.75}

# Column aliases accepted on input (e.g. the schema seed file).
INPUT_ALIASES: Dict[str, str] = {
    "implied_item_count": "true_quantity",
    "quantity": "true_quantity",
    "perceived_quality_tier": "true_quality_tier",
    "quality_tier": "true_quality_tier",
    "spend": "amount_spent",
    "amount": "amount_spent",
}

REQUIRED_COLUMNS = ("transaction_id", "category", "amount_spent",
                    "true_quantity", "true_quality_tier")

MIN_RECORDS_FOR_FIT = 3

# Plot colours (validated categorical slots 1-2; tiers also differ by marker).
TIER_STYLE = {
    "budget": {"color": "#2a78d6", "marker": "o", "label": "Budget tier"},
    "premium": {"color": "#eb6834", "marker": "D", "label": "Premium tier"},
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


# --------------------------------------------------------------------------- #
# 1. Synthetic transaction generator
# --------------------------------------------------------------------------- #

def generate_synthetic_ledger(n_records: int = 500,
                              seed: Optional[int] = 42) -> pd.DataFrame:
    """Simulate a household transaction ledger with hidden physical truth.

    Each record draws a category uniformly, a quality tier (Bernoulli on the
    category's premium share), a quantity (Gamma, shifted so at least one unit
    is bought), a continuous quality index theta (Beta by tier) and a unit
    price (log-normal around the tier median, nudged upward by theta).
    Spending is quantity times unit price, so equal spending can arise from a
    large budget basket or a single premium item.

    Parameters
    ----------
    n_records : int
        Number of transactions to generate (must be >= 1).
    seed : int or None
        Seed for numpy's Generator; None gives a non-reproducible draw.

    Returns
    -------
    pandas.DataFrame
        Columns: transaction_id, category, amount_spent, true_quantity,
        true_quality_tier, quality_index, unit_price.
    """
    try:
        n_records = int(n_records)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"n_records must be an integer, got {n_records!r}") from exc
    if n_records < 1:
        raise ValueError(f"n_records must be >= 1, got {n_records}")

    rng = np.random.default_rng(seed)
    categories = np.array(list(CATEGORY_PARAMS.keys()))
    cat_draw = rng.choice(categories, size=n_records)

    tier = np.empty(n_records, dtype=object)
    quantity = np.empty(n_records, dtype=np.int64)
    theta = np.empty(n_records, dtype=np.float64)
    unit_price = np.empty(n_records, dtype=np.float64)

    for cat in categories:
        idx = np.flatnonzero(cat_draw == cat)
        if idx.size == 0:
            continue
        p = CATEGORY_PARAMS[cat]
        is_premium = rng.random(idx.size) < p.premium_share
        tier[idx] = np.where(is_premium, "premium", "budget")

        shape = np.where(is_premium, p.premium_qty_shape, p.budget_qty_shape)
        scale = np.where(is_premium, p.premium_qty_scale, p.budget_qty_scale)
        quantity[idx] = 1 + np.floor(rng.gamma(shape, scale)).astype(np.int64)

        a = np.where(is_premium, QUALITY_BETA["premium"][0], QUALITY_BETA["budget"][0])
        b = np.where(is_premium, QUALITY_BETA["premium"][1], QUALITY_BETA["budget"][1])
        theta[idx] = rng.beta(a, b)

        median = np.where(is_premium, p.premium_unit_price, p.budget_unit_price)
        # Within-tier price rises modestly with quality (centred on tier mean).
        tier_mean_theta = np.where(is_premium, 0.75, 0.25)
        log_price = (np.log(median) + 0.5 * (theta[idx] - tier_mean_theta)
                     + rng.normal(0.0, p.price_sigma, idx.size))
        unit_price[idx] = np.exp(log_price)

    width = max(4, len(str(n_records)))
    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:0{width}d}" for i in range(1, n_records + 1)],
        "category": cat_draw.astype(str),
        "amount_spent": np.round(quantity * unit_price, 2),
        "true_quantity": quantity,
        "true_quality_tier": tier.astype(str),
        "quality_index": np.round(theta, 4),
        "unit_price": np.round(unit_price, 2),
    })
    return df


# --------------------------------------------------------------------------- #
# Input loading and validation
# --------------------------------------------------------------------------- #

def _read_csv_tolerant(path: Path) -> pd.DataFrame:
    """Read a CSV, repairing files where each whole line is wrapped in quotes."""
    df = pd.read_csv(path)
    if df.shape[1] == 1 and "," in str(df.columns[0]):
        lines = path.read_text(encoding="utf-8").splitlines()
        cleaned = []
        for line in lines:
            s = line.strip()
            if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
                s = s[1:-1].replace('""', '"')
            if s:
                cleaned.append(s)
        df = pd.read_csv(io.StringIO("\n".join(cleaned)))
    return df


def load_ledger(path: str) -> pd.DataFrame:
    """Load an observed ledger and map it onto the internal schema.

    Accepts the schema-seed column names (implied_item_count,
    perceived_quality_tier) as aliases, normalises category and tier labels,
    and fills quality_index / unit_price where absent.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input ledger not found: {p}")
    try:
        df = _read_csv_tolerant(p)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Input ledger is empty: {p}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse CSV {p}: {exc}") from exc

    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in INPUT_ALIASES.items() if k in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input ledger is missing required columns: {missing}")

    canonical = {c.lower(): c for c in CATEGORY_PARAMS}
    df["category"] = (df["category"].astype(str).str.strip().str.lower()
                      .map(canonical).fillna(df["category"].astype(str).str.strip()))
    df["true_quality_tier"] = df["true_quality_tier"].astype(str).str.strip().str.lower()

    if "quality_index" not in df.columns:
        df["quality_index"] = df["true_quality_tier"].map(DEFAULT_QUALITY_INDEX)
    return df


def validate_ledger(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a ledger into valid rows and rejected rows with a reason.

    Rejection rules: missing values in required fields, non-numeric or
    non-positive spending, non-positive quantity, unknown category or tier,
    quality index outside [0, 1].

    Returns
    -------
    (clean, rejected) : tuple of DataFrames
    """
    if df is None or len(df) == 0:
        raise ValueError("Ledger contains no rows.")

    work = df.copy()
    work["amount_spent"] = pd.to_numeric(work["amount_spent"], errors="coerce")
    work["true_quantity"] = pd.to_numeric(work["true_quantity"], errors="coerce")
    work["quality_index"] = pd.to_numeric(work.get("quality_index"), errors="coerce")

    reason = pd.Series("", index=work.index, dtype=object)

    def flag(mask: pd.Series, text: str) -> None:
        sel = mask & (reason == "")
        reason.loc[sel] = text

    flag(work[list(REQUIRED_COLUMNS)].isna().any(axis=1), "missing required field")
    flag(work["amount_spent"] <= 0, "non-positive amount_spent")
    flag(work["true_quantity"] <= 0, "non-positive true_quantity")
    flag(~work["category"].isin(CATEGORY_PARAMS.keys()), "unknown category")
    flag(~work["true_quality_tier"].isin(TIERS), "unknown quality tier")
    flag(work["quality_index"].isna() | (work["quality_index"] < 0)
         | (work["quality_index"] > 1), "quality_index outside [0, 1]")

    rejected = work.loc[reason != ""].assign(reject_reason=reason[reason != ""])
    clean = work.loc[reason == ""].copy()
    clean["true_quantity"] = clean["true_quantity"].astype(np.float64)
    if "unit_price" not in clean.columns or clean["unit_price"].isna().any():
        clean["unit_price"] = clean["amount_spent"] / clean["true_quantity"]
    return clean.reset_index(drop=True), rejected.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Coupled (EEIO) vs decoupled (LCA) emissions calculators
# --------------------------------------------------------------------------- #

def _param_vector(df: pd.DataFrame, attr: str,
                  overrides: Optional[Dict[str, float]] = None) -> np.ndarray:
    lookup = {c: getattr(p, attr) for c, p in CATEGORY_PARAMS.items()}
    if overrides:
        lookup.update(overrides)
    return df["category"].map(lookup).to_numpy(dtype=np.float64)


def calculate_standard_eeio(df: pd.DataFrame,
                            intensities: Optional[Dict[str, float]] = None
                            ) -> pd.Series:
    """Spending-based footprint: amount_spent x sector intensity (kg CO2e).

    Parameters
    ----------
    df : DataFrame with 'category' and 'amount_spent'.
    intensities : optional mapping category -> kg CO2e/EUR overriding defaults.
    """
    intensity = _param_vector(df, "eeio_intensity", intensities)
    spend = df["amount_spent"].to_numpy(dtype=np.float64)
    if np.isnan(intensity).any():
        raise ValueError("EEIO intensity missing for one or more categories.")
    return pd.Series(spend * intensity, index=df.index, name="eeio_kgco2e")


def quality_coefficient(df: pd.DataFrame) -> np.ndarray:
    """Inverse-quality modifier mu_tier * 1/(1 + g_c * theta).

    Premium units carry more embodied material (mu > 1) but their durability
    gain spreads that burden over a longer service life, so the net
    life-cycle burden per functional unit falls as theta rises.
    """
    theta = df["quality_index"].to_numpy(dtype=np.float64)
    gain = _param_vector(df, "durability_gain")
    mu = np.where(df["true_quality_tier"].to_numpy() == "premium",
                  _param_vector(df, "premium_material_multiplier"), 1.0)
    return mu / (1.0 + gain * theta)


def calculate_decoupled_lca(df: pd.DataFrame) -> pd.Series:
    """Quantity/quality footprint, independent of spending (kg CO2e).

    LCA = true_quantity x mass_per_unit x mass_emission_factor x
          quality_coefficient.
    """
    qty = df["true_quantity"].to_numpy(dtype=np.float64)
    mass = _param_vector(df, "mass_per_unit")
    ef = _param_vector(df, "mass_emission_factor")
    return pd.Series(qty * mass * ef * quality_coefficient(df),
                     index=df.index, name="lca_kgco2e")


def calibrate_eeio_intensities(df: pd.DataFrame) -> Dict[str, float]:
    """Total-preserving intensities: sum(LCA_c) / sum(spend_c) per category.

    Using these removes any level bias, so the remaining EEIO-LCA divergence
    is purely distributional (who buys what, at what price and quality).
    """
    lca = calculate_decoupled_lca(df)
    grouped = pd.DataFrame({"cat": df["category"], "lca": lca,
                            "spend": df["amount_spent"]}).groupby("cat").sum()
    out: Dict[str, float] = {}
    for cat, row in grouped.iterrows():
        if row["spend"] > 0:
            out[str(cat)] = float(row["lca"] / row["spend"])
    return out


# --------------------------------------------------------------------------- #
# 3. Decomposition error estimator
# --------------------------------------------------------------------------- #

def _threshold_from_fit(intercept: float, slope: float,
                        log_intensity: float) -> float:
    """Spend s* where the fitted LCA curve crosses the EEIO line.

    Fitted: ln LCA = a + b ln s;  EEIO: ln E = ln I + ln s.
    Setting them equal: ln s* = (a - ln I) / (1 - b).
    """
    if not np.isfinite(slope) or abs(1.0 - slope) < 1e-6:
        return float("nan")
    log_s = (intercept - log_intensity) / (1.0 - slope)
    if not np.isfinite(log_s) or abs(log_s) > 50:
        return float("nan")
    return float(np.exp(log_s))


def estimate_decomposition_error(df: pd.DataFrame,
                                 intensities: Optional[Dict[str, float]] = None,
                                 n_bootstrap: int = 500,
                                 seed: Optional[int] = 42) -> pd.DataFrame:
    """Quantify and locate the EEIO-vs-LCA divergence per category.

    For each category:
      * spending elasticity b of the true footprint from an OLS fit of
        ln(LCA) on ln(spend) (scipy.stats.linregress). EEIO assumes b = 1;
        b < 1 means EEIO over-states footprints at high spend and
        under-states them at low spend.
      * the crossover spend threshold s* with a percentile bootstrap CI.
      * bias summaries: mean signed relative error, share of over-estimates,
        and a Mann-Whitney U test that the log-ratio differs between tiers.
      * a covariance attribution of Var[ln(EEIO/LCA)] into a price effect
        (ln unit price) and a quality effect (-ln quality coefficient). The
        two shares sum to 1 because the category constant has no variance.

    Returns
    -------
    DataFrame indexed by category.
    """
    eeio = calculate_standard_eeio(df, intensities).to_numpy()
    lca = calculate_decoupled_lca(df).to_numpy()
    rng = np.random.default_rng(seed)
    rows = []

    for cat in CATEGORY_PARAMS:
        mask = (df["category"] == cat).to_numpy()
        n = int(mask.sum())
        intensity = (intensities or {}).get(cat, CATEGORY_PARAMS[cat].eeio_intensity)
        record = {"category": cat, "n": n, "eeio_intensity": intensity}
        if n == 0:
            rows.append(record)
            continue

        spend = df.loc[mask, "amount_spent"].to_numpy(dtype=np.float64)
        e, lc = eeio[mask], lca[mask]
        log_ratio = np.log(e) - np.log(lc)
        rel_err = (e - lc) / lc

        record.update({
            "total_eeio_kg": float(e.sum()),
            "total_lca_kg": float(lc.sum()),
            "aggregate_bias_pct": float(100 * (e.sum() - lc.sum()) / lc.sum()),
            "mean_signed_rel_error_pct": float(100 * rel_err.mean()),
            "share_overestimated": float((e > lc).mean()),
        })

        tiers = df.loc[mask, "true_quality_tier"].to_numpy()
        lr_b, lr_p = log_ratio[tiers == "budget"], log_ratio[tiers == "premium"]
        record["median_ratio_budget"] = float(np.exp(np.median(lr_b))) if lr_b.size else np.nan
        record["median_ratio_premium"] = float(np.exp(np.median(lr_p))) if lr_p.size else np.nan
        if lr_b.size >= 2 and lr_p.size >= 2:
            u = stats.mannwhitneyu(lr_p, lr_b, alternative="two-sided")
            record["tier_mwu_pvalue"] = float(u.pvalue)
        else:
            record["tier_mwu_pvalue"] = np.nan

        # Variance attribution of the log ratio.
        sub = df.loc[mask]
        c_price = np.log(sub["unit_price"].to_numpy(dtype=np.float64))
        c_quality = -np.log(quality_coefficient(sub))
        var_total = float(np.var(log_ratio))
        if var_total > 1e-12:
            cov_price = np.cov(c_price, log_ratio, bias=True)[0, 1]
            cov_quality = np.cov(c_quality, log_ratio, bias=True)[0, 1]
            record["var_share_price"] = float(cov_price / var_total)
            record["var_share_quality"] = float(cov_quality / var_total)
        else:
            record["var_share_price"] = np.nan
            record["var_share_quality"] = np.nan

        # Elasticity fit and crossover threshold.
        x, y = np.log(spend), np.log(lc)
        if n >= MIN_RECORDS_FOR_FIT and np.ptp(x) > 0:
            fit = stats.linregress(x, y)
            threshold = _threshold_from_fit(fit.intercept, fit.slope, np.log(intensity))
            record.update({
                "spend_elasticity": float(fit.slope),
                "elasticity_stderr": float(fit.stderr),
                "fit_intercept": float(fit.intercept),
                "fit_r2": float(fit.rvalue ** 2),
                "crossover_spend_eur": threshold,
            })
            boots = np.full(max(int(n_bootstrap), 0), np.nan)
            for i in range(boots.size):
                pick = rng.integers(0, n, n)
                if np.ptp(x[pick]) <= 0:
                    continue
                bfit = stats.linregress(x[pick], y[pick])
                boots[i] = _threshold_from_fit(bfit.intercept, bfit.slope, np.log(intensity))
            finite = boots[np.isfinite(boots)]
            if finite.size >= 10:
                lo, hi = np.percentile(finite, [2.5, 97.5])
                record["crossover_ci_low"] = float(lo)
                record["crossover_ci_high"] = float(hi)
            else:
                record["crossover_ci_low"] = np.nan
                record["crossover_ci_high"] = np.nan
            if np.isfinite(threshold):
                below = spend < threshold
                record["share_under_below_threshold"] = (
                    float((e[below] < lc[below]).mean()) if below.any() else np.nan)
                record["share_over_above_threshold"] = (
                    float((e[~below] > lc[~below]).mean()) if (~below).any() else np.nan)
        rows.append(record)

    return pd.DataFrame(rows).set_index("category")


def binned_divergence(df: pd.DataFrame, n_bins: int = 8) -> pd.DataFrame:
    """Median EEIO/LCA ratio by spend quantile bin (pooled across categories)."""
    work = df[["amount_spent", "eeio_kgco2e", "lca_kgco2e"]].copy()
    k = int(min(max(n_bins, 1), work["amount_spent"].nunique()))
    if k < 2:
        return pd.DataFrame()
    work["spend_bin"] = pd.qcut(work["amount_spent"], q=k, duplicates="drop")
    work["ratio"] = work["eeio_kgco2e"] / work["lca_kgco2e"]
    return (work.groupby("spend_bin", observed=True)
            .agg(n=("ratio", "size"), median_ratio=("ratio", "median"),
                 share_over=("ratio", lambda r: float((r > 1).mean())))
            .reset_index())


# --------------------------------------------------------------------------- #
# 4. Visualisation
# --------------------------------------------------------------------------- #

def plot_divergence(df: pd.DataFrame, decomposition: pd.DataFrame,
                    intensities: Optional[Dict[str, float]],
                    out_path: str = "emissions_divergence.png") -> Path:
    """Small multiples of spend vs true footprint, one panel per category.

    Each panel shows transactions coloured and shaped by quality tier, the
    dashed linear spending-based (EEIO) assumption, the fitted elasticity
    curve of the true footprint and, where it exists, the crossover spend.
    Log-log axes are used because spending spans several orders of
    magnitude; on them the linear EEIO assumption is a slope-1 line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cats = [c for c in CATEGORY_PARAMS if (df["category"] == c).any()]
    if not cats:
        raise ValueError("No valid records to plot.")
    ncols = 2 if len(cats) > 1 else 1
    nrows = int(np.ceil(len(cats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows),
                             squeeze=False, facecolor=SURFACE)

    for ax, cat in zip(axes.flat, cats):
        sub = df[df["category"] == cat]
        ax.set_facecolor(SURFACE)
        for tier in TIERS:
            t = sub[sub["true_quality_tier"] == tier]
            if t.empty:
                continue
            st = TIER_STYLE[tier]
            ax.scatter(t["amount_spent"], t["lca_kgco2e"], s=26, alpha=0.75,
                       color=st["color"], marker=st["marker"],
                       edgecolors=SURFACE, linewidths=0.6, zorder=3)

        spend_min = float(sub["amount_spent"].min())
        spend_max = float(sub["amount_spent"].max())
        lo, hi = spend_min * 0.8, spend_max * 1.25
        grid = np.geomspace(lo, hi, 200)
        intensity = (intensities or {}).get(cat, CATEGORY_PARAMS[cat].eeio_intensity)
        ax.plot(grid, intensity * grid, linestyle="--", color=INK_PRIMARY,
                linewidth=1.6, zorder=4)

        row = decomposition.loc[cat] if cat in decomposition.index else None
        if row is not None and np.isfinite(row.get("spend_elasticity", np.nan)):
            ax.plot(grid, np.exp(row["fit_intercept"]) * grid ** row["spend_elasticity"],
                    color=INK_SECONDARY, linewidth=1.6, zorder=4)
            thr = row.get("crossover_spend_eur", np.nan)
            if np.isfinite(thr) and lo <= thr <= hi:
                ax.axvline(thr, color=INK_SECONDARY, linewidth=1.0, linestyle=":",
                           zorder=2)
                ax.annotate(f"crossover ≈ €{thr:,.0f}", xy=(thr, 0.04),
                            xycoords=("data", "axes fraction"), xytext=(4, 0),
                            textcoords="offset points", fontsize=8.5,
                            color=INK_SECONDARY)
            subtitle = (f"elasticity b = {row['spend_elasticity']:.2f}   "
                        f"aggregate EEIO bias {row['aggregate_bias_pct']:+.0f}%")
        else:
            subtitle = "too few records for an elasticity fit"

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_title(f"{cat}  (n = {len(sub)})", loc="left", fontsize=11,
                     color=INK_PRIMARY, fontweight="bold", pad=18)
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=8.5,
                color=INK_SECONDARY, va="bottom")
        ax.set_xlabel("Amount spent (EUR, log)", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel("True footprint (kg CO2e, log)", color=INK_SECONDARY, fontsize=9)
        ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.7, zorder=0)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_COLOR)

    for ax in list(axes.flat)[len(cats):]:
        ax.set_visible(False)

    handles = [
        Line2D([], [], linestyle="none", marker=TIER_STYLE[t]["marker"],
               color=TIER_STYLE[t]["color"], markersize=7, label=TIER_STYLE[t]["label"])
        for t in TIERS
    ] + [
        Line2D([], [], linestyle="--", color=INK_PRIMARY, linewidth=1.6,
               label="Spending-based assumption (EEIO, linear)"),
        Line2D([], [], color=INK_SECONDARY, linewidth=1.6,
               label="Fitted true footprint (power law)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.965), labelcolor=INK_PRIMARY)
    fig.suptitle("Where spending-based carbon accounting diverges from "
                 "quantity/quality reality", x=0.01, ha="left", y=0.995,
                 fontsize=13, color=INK_PRIMARY, fontweight="bold")
    fig.text(0.01, 0.005, "Points above the dashed line: EEIO under-estimates; "
             "below: EEIO over-estimates. Synthetic, illustrative parameters.",
             fontsize=8, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))

    out = Path(out_path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Pipeline and CLI
# --------------------------------------------------------------------------- #

def run_pipeline(df: pd.DataFrame, calibrate: bool, n_bootstrap: int,
                 seed: Optional[int]
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                            Optional[Dict[str, float]]]:
    """Validate, compute both footprints, and run the decomposition."""
    clean, rejected = validate_ledger(df)
    if clean.empty:
        raise ValueError("No valid rows remain after validation.")
    intensities = calibrate_eeio_intensities(clean) if calibrate else None
    clean["eeio_kgco2e"] = calculate_standard_eeio(clean, intensities)
    clean["lca_kgco2e"] = calculate_decoupled_lca(clean)
    clean["residual_kgco2e"] = clean["eeio_kgco2e"] - clean["lca_kgco2e"]
    clean["eeio_to_lca_ratio"] = clean["eeio_kgco2e"] / clean["lca_kgco2e"]
    clean["eeio_direction"] = np.where(clean["eeio_to_lca_ratio"] > 1,
                                       "over-estimate", "under-estimate")
    decomposition = estimate_decomposition_error(clean, intensities, n_bootstrap, seed)
    return clean, rejected, decomposition, intensities


def _fmt(value: float, spec: str) -> str:
    """Format a number, rendering NaN/None as 'n/a'."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return format(v, spec) if np.isfinite(v) else "n/a"


def format_report(clean: pd.DataFrame, rejected: pd.DataFrame,
                  decomposition: pd.DataFrame,
                  intensities: Optional[Dict[str, float]]) -> str:
    """Human-readable console summary."""
    lines = []
    total_e, total_l = clean["eeio_kgco2e"].sum(), clean["lca_kgco2e"].sum()
    lines.append("=" * 78)
    lines.append("FOOTPRINT DECOUPLER - spending-based vs quantity/quality accounting")
    lines.append("=" * 78)
    lines.append(f"Valid records: {len(clean)}   rejected: {len(rejected)}")
    if not rejected.empty:
        for reason, count in rejected["reject_reason"].value_counts().items():
            lines.append(f"  - {count} x {reason}")
    mode = "calibrated (total-preserving)" if intensities else "fixed defaults"
    lines.append(f"EEIO intensities: {mode}")
    lines.append(f"Total EEIO footprint : {total_e:12,.1f} kg CO2e")
    lines.append(f"Total LCA footprint  : {total_l:12,.1f} kg CO2e")
    lines.append(f"Aggregate EEIO bias  : {100 * (total_e - total_l) / total_l:+.1f}%")
    lines.append(f"Transactions over-estimated by EEIO: "
                 f"{100 * (clean['eeio_to_lca_ratio'] > 1).mean():.1f}%")
    lines.append("")

    for cat, row in decomposition.iterrows():
        lines.append(f"[{cat}]  n = {int(row['n'])}   I = {row['eeio_intensity']:.3f} kg/EUR")
        if row["n"] == 0:
            lines.append("  no records")
            lines.append("")
            continue
        lines.append(f"  aggregate bias {row['aggregate_bias_pct']:+.1f}%   "
                     f"over-estimated share {100 * row['share_overestimated']:.0f}%")
        lines.append(f"  median EEIO/LCA ratio  budget {_fmt(row['median_ratio_budget'], '.2f')}   "
                     f"premium {_fmt(row['median_ratio_premium'], '.2f')}   "
                     f"(Mann-Whitney p = {_fmt(row['tier_mwu_pvalue'], '.2g')})")
        if np.isfinite(row.get("var_share_price", np.nan)):
            lines.append(f"  log-ratio variance: price effect {100 * row['var_share_price']:.0f}%, "
                         f"quality effect {100 * row['var_share_quality']:.0f}%")
        b = row.get("spend_elasticity", np.nan)
        if np.isfinite(b):
            lines.append(f"  spend elasticity of true footprint b = {b:.3f} "
                         f"(SE {row['elasticity_stderr']:.3f}, R2 {row['fit_r2']:.2f}); "
                         "EEIO assumes 1")
            thr = row.get("crossover_spend_eur", np.nan)
            if np.isfinite(thr):
                ci = ""
                if np.isfinite(row.get("crossover_ci_low", np.nan)):
                    ci = (f" [95% CI {row['crossover_ci_low']:,.0f} - "
                          f"{row['crossover_ci_high']:,.0f}]")
                direction = ("EEIO under-estimates below, over-estimates above"
                             if b < 1 else "EEIO over-estimates below, under-estimates above")
                lines.append(f"  crossover spend ≈ EUR {thr:,.0f}{ci}: {direction}")
            else:
                lines.append("  no finite crossover within the data "
                             "(elasticity ≈ 1 or curves do not meet)")
        else:
            lines.append("  too few records for an elasticity fit")
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="footprint_decoupler",
        description=("Compare spending-based (EEIO) and quantity/quality-based "
                     "(LCA) carbon footprints on a transaction ledger and locate "
                     "where they diverge."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--records", type=int, default=500,
                        help="Number of synthetic transactions to generate "
                             "(ignored when --input is given).")
    parser.add_argument("--input", type=str, default=None,
                        help="Optional CSV ledger to analyse instead of synthetic "
                             "data. Needs transaction_id, category, amount_spent "
                             "and quantity/tier columns (seed-file aliases "
                             "implied_item_count, perceived_quality_tier accepted).")
    parser.add_argument("--export-csv", type=str, default=None, metavar="PATH",
                        help="Write the enriched ledger (both footprints, residuals) "
                             "to this CSV; a <stem>_decomposition.csv is written "
                             "alongside it.")
    parser.add_argument("--plot", action="store_true",
                        help="Save the divergence figure.")
    parser.add_argument("--plot-path", type=str, default="emissions_divergence.png",
                        help="Output path for the figure when --plot is set.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data generation and bootstrap; "
                             "use -1 for a non-reproducible run.")
    parser.add_argument("--calibrate-eeio", action="store_true",
                        help="Replace default EEIO intensities with total-preserving "
                             "ones so only distributional divergence remains.")
    parser.add_argument("--bootstrap", type=int, default=500,
                        help="Bootstrap resamples for the crossover-threshold CI "
                             "(0 disables).")
    parser.add_argument("--json", action="store_true",
                        help="Print the decomposition table as JSON instead of the "
                             "text report.")
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.records < 1 and args.input is None:
        parser.error("--records must be a positive integer.")
    if args.bootstrap < 0:
        parser.error("--bootstrap must be >= 0.")
    seed = None if args.seed == -1 else args.seed

    try:
        if args.input:
            raw = load_ledger(args.input)
        else:
            raw = generate_synthetic_ledger(args.records, seed=seed)
        clean, rejected, decomposition, intensities = run_pipeline(
            raw, args.calibrate_eeio, args.bootstrap, seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(json.loads(decomposition.reset_index().to_json(orient="records")),
                         indent=2))
    else:
        print(format_report(clean, rejected, decomposition, intensities))
        bins = binned_divergence(clean)
        if not bins.empty:
            print("Pooled spend bins (median EEIO/LCA ratio):")
            for _, r in bins.iterrows():
                print(f"  {str(r['spend_bin']):>28}  n={int(r['n']):4d}  "
                      f"ratio={r['median_ratio']:.2f}  over={100 * r['share_over']:.0f}%")

    try:
        if args.export_csv:
            out = Path(args.export_csv)
            if out.parent and not out.parent.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
            clean.to_csv(out, index=False)
            decomp_path = out.with_name(f"{out.stem}_decomposition.csv")
            decomposition.to_csv(decomp_path)
            if not rejected.empty:
                rejected.to_csv(out.with_name(f"{out.stem}_rejected.csv"), index=False)
            print(f"\nLedger written to {out}\nDecomposition written to {decomp_path}",
                  file=sys.stderr)
        if args.plot:
            path = plot_divergence(clean, decomposition, intensities, args.plot_path)
            print(f"Figure written to {path}", file=sys.stderr)
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
