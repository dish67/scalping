from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "XBTUSDTM_1d_clean.csv"
RESULTS = ROOT / "results"
INITIAL_CAPITAL = 10_000.0
SMA_LENGTH = 200


@dataclass
class Position:
    entry_time: pd.Timestamp
    entry: float
    qty: float
    entry_fee: float


@dataclass(frozen=True)
class Costs:
    name: str
    taker_fee_pct: float
    funding_pct: float
    slippage_pct: float


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().set_index("datetime").sort_index()
    df["sma200"] = df["close"].rolling(SMA_LENGTH).mean()
    return df.dropna(subset=["sma200"]).copy()


def count_missing_daily_candles(raw: pd.DataFrame) -> tuple[int, int]:
    ordered = raw.sort_values("datetime")
    expected = pd.Timedelta(days=1)
    differences = ordered["datetime"].diff()
    gaps = differences[differences > expected]
    missing = int(sum(int(delta / expected) - 1 for delta in gaps))
    return len(gaps), missing


def funding_periods(entry: pd.Timestamp, exit_: pd.Timestamp) -> int:
    eight_hours = 8 * 60 * 60
    return max(0, int(exit_.timestamp() // eight_hours - entry.timestamp() // eight_hours))


def close_position(
    position: Position,
    timestamp: pd.Timestamp,
    exit_price: float,
    equity: float,
    fee_rate: float,
    funding_rate: float,
    result: str,
) -> tuple[float, dict]:
    gross = (exit_price - position.entry) * position.qty
    exit_fee = exit_price * position.qty * fee_rate
    funding = (
        position.entry
        * position.qty
        * funding_rate
        * funding_periods(position.entry_time, timestamp)
    )
    net = gross - position.entry_fee - exit_fee - funding
    equity += gross - exit_fee - funding
    trade = {
        "entry_time": position.entry_time,
        "exit_time": timestamp,
        "entry": position.entry,
        "exit": exit_price,
        "qty": position.qty,
        "gross_pnl": gross,
        "fees": position.entry_fee + exit_fee,
        "funding": funding,
        "net_pnl": net,
        "result": result,
    }
    return equity, trade


def run(df: pd.DataFrame, costs: Costs) -> tuple[pd.DataFrame, pd.DataFrame]:
    equity = INITIAL_CAPITAL
    fee_rate = costs.taker_fee_pct / 100
    funding_rate = costs.funding_pct / 100
    slippage_rate = costs.slippage_pct / 100
    position: Position | None = None
    pending_entry = False
    pending_exit = False
    trades: list[dict] = []
    curve: list[dict] = []

    for timestamp, row in df.iterrows():
        # Un signal produit à la clôture est toujours exécuté à l'ouverture
        # suivante. La SMA du jour n'est donc jamais tradée rétroactivement.
        if pending_exit and position is not None:
            exit_price = float(row.open) * (1 - slippage_rate)
            equity, trade = close_position(
                position,
                timestamp,
                exit_price,
                equity,
                fee_rate,
                funding_rate,
                "sma_exit",
            )
            trades.append(trade)
            position = None
            pending_exit = False

        if pending_entry and position is None:
            entry = float(row.open) * (1 + slippage_rate)
            qty = equity / (entry * (1 + fee_rate))
            entry_fee = entry * qty * fee_rate
            equity -= entry_fee
            position = Position(timestamp, entry, qty, entry_fee)
            pending_entry = False

        if position is None and not pending_entry and row.close > row.sma200:
            pending_entry = True
        elif position is not None and row.close < row.sma200:
            pending_exit = True

        marked_equity = equity
        if position is not None:
            unrealized = (float(row.close) - position.entry) * position.qty
            projected_exit_fee = float(row.close) * position.qty * fee_rate
            accrued_funding = (
                position.entry
                * position.qty
                * funding_rate
                * funding_periods(position.entry_time, timestamp)
            )
            marked_equity += unrealized - projected_exit_fee - accrued_funding
        curve.append(
            {
                "datetime": timestamp,
                "equity": marked_equity,
                "in_position": position is not None,
            }
        )

    if position is not None:
        timestamp = df.index[-1]
        exit_price = float(df.iloc[-1].close) * (1 - slippage_rate)
        equity, trade = close_position(
            position,
            timestamp,
            exit_price,
            equity,
            fee_rate,
            funding_rate,
            "end_of_test",
        )
        trades.append(trade)
        if curve:
            curve[-1]["equity"] = equity
            curve[-1]["in_position"] = False

    return pd.DataFrame(trades), pd.DataFrame(curve)


def buy_hold(df: pd.DataFrame, costs: Costs, include_funding: bool) -> float:
    if df.empty:
        return 0.0
    fee_rate = costs.taker_fee_pct / 100
    funding_rate = costs.funding_pct / 100 if include_funding else 0.0
    slippage_rate = costs.slippage_pct / 100
    entry = float(df.iloc[0].open) * (1 + slippage_rate)
    exit_ = float(df.iloc[-1].close) * (1 - slippage_rate)
    qty = INITIAL_CAPITAL / (entry * (1 + fee_rate))
    entry_fee = entry * qty * fee_rate
    exit_fee = exit_ * qty * fee_rate
    funding = (
        entry
        * qty
        * funding_rate
        * funding_periods(df.index[0], df.index[-1])
    )
    return (exit_ - entry) * qty - entry_fee - exit_fee - funding


def report(
    trades: pd.DataFrame,
    curve: pd.DataFrame,
    frame: pd.DataFrame,
    split: str,
    costs: Costs,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"daily_regime_v6_1_{split}_{costs.name}"
    trades.to_csv(RESULTS / f"trades_{stem}.csv", index=False)
    curve.to_csv(RESULTS / f"equity_{stem}.csv", index=False)

    wins = trades.loc[trades.net_pnl > 0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
    losses = trades.loc[trades.net_pnl < 0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty else np.inf
    pnl = trades.net_pnl.sum() if not trades.empty else 0.0
    drawdown = curve.equity / curve.equity.cummax() - 1
    exposure = curve.in_position.mean() * 100
    durations = (
        (pd.to_datetime(trades.exit_time) - pd.to_datetime(trades.entry_time))
        .dt.total_seconds()
        .div(86_400)
        if not trades.empty
        else pd.Series(dtype=float)
    )
    benchmark_perp = buy_hold(frame, costs, include_funding=True)
    benchmark_spot = buy_hold(frame, costs, include_funding=False)
    counts = trades.result.value_counts() if not trades.empty else pd.Series(dtype=int)
    yearly_lines: list[str] = []
    if split == "ensemble_complet_reference" and not curve.empty:
        dated_curve = curve.copy()
        dated_curve["datetime"] = pd.to_datetime(dated_curve["datetime"])
        previous_equity = INITIAL_CAPITAL
        for year, group in dated_curve.groupby(dated_curve["datetime"].dt.year):
            ending_equity = float(group.iloc[-1].equity)
            annual_pnl = ending_equity - previous_equity
            annual_return = annual_pnl / previous_equity * 100
            path = pd.concat(
                [
                    pd.Series([previous_equity], dtype=float),
                    group.equity.reset_index(drop=True),
                ],
                ignore_index=True,
            )
            annual_dd = (path / path.cummax() - 1).min() * 100
            yearly_lines.append(
                f"  {year}: {annual_pnl:.2f} USDT "
                f"({annual_return:.2f}%) | DD {annual_dd:.2f}%"
            )
            previous_equity = ending_equity
    yearly_summary = (
        "Variation continue de l'equity par année:\n"
        + "\n".join(yearly_lines)
        + "\n"
        if yearly_lines
        else ""
    )

    summary = (
        f"BTC Daily Regime V6.1 — {split} — {costs.name}\n"
        f"Période: {frame.index[0]} → {frame.index[-1]}\n"
        f"Règle: clôture > SMA 200 | Exécution: prochain open | Long/cash 1x\n"
        f"Frais taker: {costs.taker_fee_pct:.3f}% | Funding: "
        f"{costs.funding_pct:.3f}% / 8H | Slippage: {costs.slippage_pct:.3f}% / ordre\n"
        f"Trades: {len(trades)} | PnL net: {pnl:.2f} USDT "
        f"({pnl / INITIAL_CAPITAL * 100:.2f}%)\n"
        f"Profit factor: {profit_factor:.3f} | "
        f"Gagnants: {(trades.net_pnl > 0).mean() * 100 if not trades.empty else 0:.2f}%\n"
        f"Drawdown maximum: {drawdown.min() * 100:.2f}% | "
        f"Exposition: {exposure:.2f}%\n"
        f"Durée moyenne/médiane: "
        f"{durations.mean() if not durations.empty else 0:.1f}/"
        f"{durations.median() if not durations.empty else 0:.1f} jours\n"
        f"Frais: {trades.fees.sum() if not trades.empty else 0:.2f} | "
        f"Funding: {trades.funding.sum() if not trades.empty else 0:.2f} USDT\n"
        f"Sorties SMA/fin: {counts.get('sma_exit', 0)}/{counts.get('end_of_test', 0)}\n"
        f"Buy&Hold perp net funding: {benchmark_perp:.2f} USDT | "
        f"Buy&Hold spot proxy: {benchmark_spot:.2f} USDT\n"
        f"{yearly_summary}"
    )
    print(summary)
    (RESULTS / f"summary_{stem}.txt").write_text(summary, encoding="utf-8")

    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(curve.datetime), curve.equity)
    plt.axhline(INITIAL_CAPITAL, color="grey", linewidth=1, alpha=0.5)
    plt.title(f"BTC Daily Regime V6.1 — {split} — {costs.name}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / f"equity_{stem}.png", dpi=150)
    plt.close()


def report_period(
    trades: pd.DataFrame,
    curve: pd.DataFrame,
    frame: pd.DataFrame,
    split: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    costs: Costs,
) -> None:
    dated_curve = curve.copy()
    dated_curve["datetime"] = pd.to_datetime(dated_curve["datetime"])
    mask = pd.Series(True, index=dated_curve.index)
    if start is not None:
        mask &= dated_curve["datetime"] >= start
    if end is not None:
        mask &= dated_curve["datetime"] < end
    period_curve = dated_curve.loc[mask].copy()
    if period_curve.empty:
        print(f"BTC Daily Regime V6.1 — {split}: période vide")
        return

    before = dated_curve.loc[dated_curve["datetime"] < period_curve.iloc[0].datetime]
    starting_equity = (
        float(before.iloc[-1].equity) if not before.empty else INITIAL_CAPITAL
    )
    ending_equity = float(period_curve.iloc[-1].equity)
    pnl = ending_equity - starting_equity
    period_return = pnl / starting_equity * 100
    path = pd.concat(
        [
            pd.Series([starting_equity], dtype=float),
            period_curve.equity.reset_index(drop=True),
        ],
        ignore_index=True,
    )
    drawdown = (path / path.cummax() - 1).min() * 100
    exposure = period_curve.in_position.mean() * 100

    period_start = pd.Timestamp(period_curve.iloc[0].datetime)
    period_end = pd.Timestamp(period_curve.iloc[-1].datetime)
    entries = 0
    exits = 0
    if not trades.empty:
        entry_times = pd.to_datetime(trades.entry_time)
        exit_times = pd.to_datetime(trades.exit_time)
        entries = int(((entry_times >= period_start) & (entry_times <= period_end)).sum())
        exits = int(((exit_times >= period_start) & (exit_times <= period_end)).sum())

    benchmark_frame = frame.loc[
        (frame.index >= period_start) & (frame.index <= period_end)
    ]
    benchmark_perp = buy_hold(benchmark_frame, costs, include_funding=True)
    benchmark_spot = buy_hold(benchmark_frame, costs, include_funding=False)
    summary = (
        f"BTC Daily Regime V6.1 — {split} — {costs.name}\n"
        f"Période mesurée en continu: {period_start} → {period_end}\n"
        f"Equity début/fin: {starting_equity:.2f} → {ending_equity:.2f} USDT\n"
        f"PnL de période: {pnl:.2f} USDT ({period_return:.2f}%)\n"
        f"Drawdown de période: {drawdown:.2f}% | Exposition: {exposure:.2f}%\n"
        f"Entrées/sorties pendant la période: {entries}/{exits}\n"
        f"Buy&Hold perp de période: {benchmark_perp:.2f} USDT | "
        f"Buy&Hold spot proxy: {benchmark_spot:.2f} USDT\n"
        "Note: aucune position n'est fermée artificiellement à la frontière.\n"
    )
    print(summary)

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"daily_regime_v6_1_{split}_{costs.name}"
    period_curve.to_csv(RESULTS / f"equity_{stem}.csv", index=False)
    (RESULTS / f"summary_{stem}.txt").write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BTC Daily Regime V6.1: filtre journalier SMA 200, long/cash."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--taker-fee", type=float, default=0.035)
    parser.add_argument("--funding", type=float, default=0.010)
    parser.add_argument("--slippage", type=float, default=0.010)
    parser.add_argument(
        "--no-stress",
        action="store_true",
        help="N'exécute que le scénario BTCC de base.",
    )
    parser.add_argument(
        "--allow-gaps",
        action="store_true",
        help="Autorise explicitement un backtest sur une série journalière incomplète.",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.data, parse_dates=["datetime"])
    gap_count, missing_count = count_missing_daily_candles(raw)
    if gap_count and not args.allow_gaps:
        raise RuntimeError(
            f"Données journalières incomplètes: {gap_count} trou(s), environ "
            f"{missing_count} bougie(s) manquante(s). "
            "Retéléchargez les données avant le backtest."
        )

    prepared = prepare(raw)
    if prepared.empty:
        raise RuntimeError(
            f"Il faut au moins {SMA_LENGTH} bougies journalières continues."
        )

    periods = {
        "developpement_jusqua_2024": (
            None,
            pd.Timestamp("2025-01-01", tz="UTC"),
        ),
        "validation_2025": (
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2026-01-01", tz="UTC"),
        ),
        "test_2026": (
            pd.Timestamp("2026-01-01", tz="UTC"),
            None,
        ),
    }
    scenarios = [
        Costs("btcc_vip3_base", args.taker_fee, args.funding, args.slippage)
    ]
    if not args.no_stress:
        scenarios.append(
            Costs(
                "stress",
                max(args.taker_fee, 0.050),
                max(args.funding, 0.015),
                max(args.slippage, 0.020),
            )
        )

    for costs in scenarios:
        # Une seule simulation continue par scénario. Les rapports temporels
        # découpent ensuite l'equity sans réinitialiser le capital ni la position.
        trades, curve = run(prepared, costs)
        for split, (start, end) in periods.items():
            report_period(trades, curve, prepared, split, start, end, costs)
        report(
            trades,
            curve,
            prepared,
            "ensemble_complet_reference",
            costs,
        )


if __name__ == "__main__":
    main()
