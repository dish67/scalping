from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "XBTUSDTM_4h.csv"
RESULTS = ROOT / "results"


@dataclass
class Position:
    entry_time: pd.Timestamp
    entry: float
    qty: float
    entry_fee: float
    stop: float
    highest_close: float


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().set_index("datetime").sort_index()
    previous_close = df.close.shift()
    true_range = pd.concat(
        [
            df.high - df.low,
            (df.high - previous_close).abs(),
            (df.low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    df["ema50"] = df.close.ewm(span=50, adjust=False).mean()
    return df.dropna().copy()


def count_missing_4h_candles(raw: pd.DataFrame) -> tuple[int, int]:
    ordered = raw.sort_values("datetime")
    expected = pd.Timedelta(hours=4)
    differences = ordered["datetime"].diff()
    gaps = differences[differences > expected]
    missing = int(sum(int(delta / expected) - 1 for delta in gaps))
    return len(gaps), missing


def funding_periods(entry: pd.Timestamp, exit_: pd.Timestamp) -> int:
    eight_hours = 8 * 60 * 60
    return max(0, int(exit_.timestamp() // eight_hours - entry.timestamp() // eight_hours))


def run(
    df: pd.DataFrame,
    entry_length: int,
    exit_length: int,
    atr_multiple: float,
    taker_fee_pct: float,
    funding_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    initial_capital = 10_000.0
    equity = initial_capital
    fee_rate = taker_fee_pct / 100
    funding_rate = funding_pct / 100
    slippage = 0.2
    risk_pct = 0.005
    position: Position | None = None
    pending_entry = False
    pending_exit = False
    trades: list[dict] = []
    curve: list[dict] = []

    entry_channel = df.high.shift(1).rolling(entry_length).max()
    exit_channel = df.low.shift(1).rolling(exit_length).min()

    for i in range(max(entry_length, 53), len(df)):
        timestamp = df.index[i]
        row = df.iloc[i]

        # Les signaux à la clôture précédente sont exécutés à l'ouverture
        # suivante afin de ne pas supposer une exécution rétroactive.
        if pending_exit and position is not None:
            exit_price = row.open - slippage
            gross = (exit_price - position.entry) * position.qty
            exit_fee = exit_price * position.qty * fee_rate
            periods = funding_periods(position.entry_time, timestamp)
            funding = position.entry * position.qty * funding_rate * periods
            net = gross - position.entry_fee - exit_fee - funding
            equity += gross - exit_fee - funding
            trades.append(
                {
                    "entry_time": position.entry_time,
                    "exit_time": timestamp,
                    "entry": position.entry,
                    "exit": exit_price,
                    "qty": position.qty,
                    "gross_pnl": gross,
                    "fees": position.entry_fee + exit_fee,
                    "funding": funding,
                    "net_pnl": net,
                    "result": "channel",
                }
            )
            position = None
            pending_exit = False

        if pending_entry and position is None:
            entry = row.open + slippage
            stop_distance = row.atr * atr_multiple
            estimated_cost = entry * fee_rate * 2 + slippage * 2
            estimated_funding = entry * funding_rate * 6
            risk_per_unit = stop_distance + estimated_cost + estimated_funding
            qty = min(equity * risk_pct / risk_per_unit, equity / entry)
            entry_fee = entry * qty * fee_rate
            equity -= entry_fee
            position = Position(
                timestamp, entry, qty, entry_fee,
                entry - stop_distance, float(row.close),
            )
            pending_entry = False

        # Stop suiveur intrabar. Il est mis à jour seulement avec les clôtures
        # déjà observées, jamais avec le futur plus haut de la même bougie.
        if position is not None:
            if row.low <= position.stop:
                exit_price = position.stop - slippage
                gross = (exit_price - position.entry) * position.qty
                exit_fee = exit_price * position.qty * fee_rate
                periods = funding_periods(position.entry_time, timestamp)
                funding = position.entry * position.qty * funding_rate * periods
                net = gross - position.entry_fee - exit_fee - funding
                equity += gross - exit_fee - funding
                trades.append(
                    {
                        "entry_time": position.entry_time,
                        "exit_time": timestamp,
                        "entry": position.entry,
                        "exit": exit_price,
                        "qty": position.qty,
                        "gross_pnl": gross,
                        "fees": position.entry_fee + exit_fee,
                        "funding": funding,
                        "net_pnl": net,
                        "result": "atr_stop",
                    }
                )
                position = None
                pending_exit = False
            else:
                position.highest_close = max(position.highest_close, float(row.close))
                position.stop = max(position.stop, position.highest_close - row.atr * atr_multiple)

        if position is None and not pending_entry:
            trend_ok = row.close > row.ema50 and row.ema50 > df.iloc[i - 3].ema50
            if row.close > entry_channel.iloc[i] and trend_ok:
                pending_entry = True
        elif position is not None and row.close < exit_channel.iloc[i]:
            pending_exit = True

        marked_equity = equity
        if position is not None:
            marked_equity += (row.close - position.entry) * position.qty
        curve.append({"datetime": timestamp, "equity": marked_equity})

    # Fermeture au dernier cours pour ne pas ignorer une position ouverte.
    if position is not None:
        timestamp = df.index[-1]
        exit_price = df.iloc[-1].close - slippage
        gross = (exit_price - position.entry) * position.qty
        exit_fee = exit_price * position.qty * fee_rate
        periods = funding_periods(position.entry_time, timestamp)
        funding = position.entry * position.qty * funding_rate * periods
        net = gross - position.entry_fee - exit_fee - funding
        trades.append(
            {
                "entry_time": position.entry_time,
                "exit_time": timestamp,
                "entry": position.entry,
                "exit": exit_price,
                "qty": position.qty,
                "gross_pnl": gross,
                "fees": position.entry_fee + exit_fee,
                "funding": funding,
                "net_pnl": net,
                "result": "end_of_test",
            }
        )

    return pd.DataFrame(trades), pd.DataFrame(curve)


def buy_hold(df: pd.DataFrame, taker_fee_pct: float, funding_pct: float) -> float:
    if df.empty:
        return 0.0
    capital = 10_000.0
    fee_rate = taker_fee_pct / 100
    funding_rate = funding_pct / 100
    entry, exit_ = float(df.iloc[0].open), float(df.iloc[-1].close)
    qty = capital / entry
    fees = (entry + exit_) * qty * fee_rate
    funding = entry * qty * funding_rate * funding_periods(df.index[0], df.index[-1])
    return (exit_ - entry) * qty - fees - funding


def report(
    trades: pd.DataFrame,
    curve: pd.DataFrame,
    frame: pd.DataFrame,
    split: str,
    model: str,
    taker_fee_pct: float,
    funding_pct: float,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"trend_v5_{split}_{model}"
    trades.to_csv(RESULTS / f"trades_{stem}.csv", index=False)
    curve.to_csv(RESULTS / f"equity_{stem}.csv", index=False)
    benchmark = buy_hold(frame, taker_fee_pct, funding_pct)
    if trades.empty:
        print(f"BTC Trend V5 — {split} — {model}: aucun trade | Buy&Hold {benchmark:.2f} USDT")
        return

    wins = trades.loc[trades.net_pnl > 0, "net_pnl"]
    losses = trades.loc[trades.net_pnl < 0, "net_pnl"]
    pf = wins.sum() / abs(losses.sum()) if not losses.empty else np.inf
    dd = curve.equity / curve.equity.cummax() - 1
    counts = trades.result.value_counts()
    summary = (
        f"BTC Trend V5 — {split} — {model}\n"
        f"Période: {frame.index[0]} → {frame.index[-1]}\n"
        f"Timeframe: 4H | Long/cash | Risque: 0.50%\n"
        f"Trades: {len(trades)} | PnL net: {trades.net_pnl.sum():.2f} USDT\n"
        f"Profit factor: {pf:.3f} | Gagnants: {(trades.net_pnl > 0).mean() * 100:.2f}%\n"
        f"Gain moyen: {wins.mean() if not wins.empty else 0:.2f} USDT\n"
        f"Perte moyenne: {losses.mean() if not losses.empty else 0:.2f} USDT\n"
        f"Frais: {trades.fees.sum():.2f} | Funding: {trades.funding.sum():.2f} USDT\n"
        f"Drawdown maximum: {dd.min() * 100:.2f}%\n"
        f"Sorties canal/ATR/fin: {counts.get('channel', 0)}/{counts.get('atr_stop', 0)}/{counts.get('end_of_test', 0)}\n"
        f"Benchmark Buy&Hold futures net du funding: {benchmark:.2f} USDT\n"
    )
    print(summary)
    (RESULTS / f"summary_{stem}.txt").write_text(summary, encoding="utf-8")
    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(curve.datetime), curve.equity)
    plt.title(f"BTC Trend V5 — {split} — {model}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / f"equity_{stem}.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC Trend V5: Donchian + ATR sur 4H.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--taker-fee", type=float, default=0.035)
    parser.add_argument("--funding", type=float, default=0.010)
    parser.add_argument(
        "--allow-gaps",
        action="store_true",
        help="Autorise explicitement un backtest sur une serie 4H incomplete.",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.data, parse_dates=["datetime"])
    gap_count, missing_count = count_missing_4h_candles(raw)
    if gap_count and not args.allow_gaps:
        raise RuntimeError(
            f"Donnees 4H incompletes: {gap_count} trou(s), environ "
            f"{missing_count} bougie(s) manquante(s). "
            "Retelchargez les donnees avant le backtest."
        )
    prepared = prepare(raw)
    splits = {
        "developpement_avant_2026": prepared.loc[prepared.index < pd.Timestamp("2026-01-01", tz="UTC")],
        "validation_2026": prepared.loc[prepared.index >= pd.Timestamp("2026-01-01", tz="UTC")],
    }
    models = {
        "donchian_20_10": (20, 10, 3.0),
        "donchian_55_20": (55, 20, 3.0),
    }
    for split, frame in splits.items():
        for model, parameters in models.items():
            trades, curve = run(frame, *parameters, args.taker_fee, args.funding)
            report(trades, curve, frame, split, model, args.taker_fee, args.funding)


if __name__ == "__main__":
    main()
