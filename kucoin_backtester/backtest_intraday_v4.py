from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "XBTUSDTM_15m.csv"
RESULTS = ROOT / "results"


@dataclass
class Pending:
    side: str
    created: int
    expires: int
    midpoint: float
    stop: float
    touched: int | None = None


@dataclass
class Position:
    side: str
    entry_time: pd.Timestamp
    entry_index: int
    entry: float
    stop: float
    target: float
    qty: float
    entry_fee: float


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().set_index("datetime").sort_index()
    df["ema20"] = df.close.ewm(span=20, adjust=False).mean()
    df["ema50"] = df.close.ewm(span=50, adjust=False).mean()
    df["rsi"] = rsi(df.close)
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
    df["volume_avg"] = df.volume.rolling(20).mean()

    four_hour = df.resample("4h", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    four_hour["ema20_4h"] = four_hour.close.ewm(span=20, adjust=False).mean()
    four_hour["ema50_4h"] = four_hour.close.ewm(span=50, adjust=False).mean()
    four_hour["ema200_4h"] = four_hour.close.ewm(span=200, adjust=False).mean()
    # Une bougie de décalage: seulement la dernière 4H entièrement clôturée.
    context = four_hour[["close", "ema20_4h", "ema50_4h", "ema200_4h"]].shift(1)
    context.columns = ["close_4h", "ema20_4h", "ema50_4h", "ema200_4h"]
    df = pd.merge_asof(
        df.reset_index().sort_values("datetime"),
        context.reset_index().sort_values("datetime"),
        on="datetime",
        direction="backward",
    ).set_index("datetime")
    return df.dropna().copy()


def funding_periods(entry: pd.Timestamp, exit_: pd.Timestamp) -> int:
    interval_seconds = 8 * 60 * 60
    return max(0, int(exit_.timestamp() // interval_seconds - entry.timestamp() // interval_seconds))


def run(
    df: pd.DataFrame,
    direction: str,
    taker_fee_pct: float,
    funding_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    initial_capital = 10_000.0
    equity = initial_capital
    taker_fee = taker_fee_pct / 100
    funding_rate = funding_pct / 100
    slippage = 0.2
    risk_pct = 0.0025
    max_hold_bars = 64  # 16 heures en 15 minutes.
    pending: Pending | None = None
    position: Position | None = None
    trades: list[dict] = []
    setups: list[dict] = []
    curve: list[dict] = []
    last_exit = -10_000
    consecutive_losses = 0
    trades_today = 0
    current_day = None
    day_start_equity = equity
    bull_sweep_index: int | None = None
    bear_sweep_index: int | None = None
    bull_sweep_low = np.nan
    bear_sweep_high = np.nan

    prior_low = df.low.shift(1).rolling(20).min()
    prior_high = df.high.shift(1).rolling(20).max()
    structure_high = df.high.shift(3).rolling(10).max()
    structure_low = df.low.shift(3).rolling(10).min()

    for i in range(203, len(df)):
        timestamp = df.index[i]
        row = df.iloc[i]
        if current_day != timestamp.date():
            current_day = timestamp.date()
            day_start_equity = equity
            trades_today = 0
            consecutive_losses = 0

        if position is not None and i > position.entry_index:
            if position.side == "long":
                stop_hit, target_hit = row.low <= position.stop, row.high >= position.target
                if stop_hit:
                    exit_price, reason = position.stop - slippage, "stop"
                elif target_hit:
                    exit_price, reason = position.target - slippage, "target"
                elif i - position.entry_index >= max_hold_bars:
                    exit_price, reason = row.close - slippage, "time"
                else:
                    exit_price, reason = None, None
            else:
                stop_hit, target_hit = row.high >= position.stop, row.low <= position.target
                if stop_hit:
                    exit_price, reason = position.stop + slippage, "stop"
                elif target_hit:
                    exit_price, reason = position.target + slippage, "target"
                elif i - position.entry_index >= max_hold_bars:
                    exit_price, reason = row.close + slippage, "time"
                else:
                    exit_price, reason = None, None

            if exit_price is not None:
                gross = (exit_price - position.entry) * position.qty
                if position.side == "short":
                    gross = -gross
                exit_fee = exit_price * position.qty * taker_fee
                periods = funding_periods(position.entry_time, timestamp)
                funding = position.entry * position.qty * funding_rate * periods
                net = gross - position.entry_fee - exit_fee - funding
                equity += gross - exit_fee - funding
                trades.append(
                    {
                        "side": position.side,
                        "entry_time": position.entry_time,
                        "exit_time": timestamp,
                        "entry": position.entry,
                        "exit": exit_price,
                        "stop": position.stop,
                        "target": position.target,
                        "qty": position.qty,
                        "gross_pnl": gross,
                        "fees": position.entry_fee + exit_fee,
                        "funding": funding,
                        "net_pnl": net,
                        "bars": i - position.entry_index,
                        "result": reason,
                    }
                )
                consecutive_losses = consecutive_losses + 1 if net < 0 else 0
                position = None
                last_exit = i

        curve.append({"datetime": timestamp, "equity": equity})
        if position is not None:
            continue

        if pending is not None:
            invalid = row.low <= pending.stop if pending.side == "long" else row.high >= pending.stop
            if i > pending.expires or invalid:
                pending = None
            elif i > pending.created:
                midpoint_touch = row.low <= pending.midpoint if pending.side == "long" else row.high >= pending.midpoint
                if midpoint_touch and pending.touched is None:
                    pending.touched = i
                recent_touch = pending.touched is not None and i - pending.touched <= 4
                bar_range = row.high - row.low
                body = abs(row.close - row.open)
                if pending.side == "long":
                    confirmed = (
                        recent_touch and row.close > pending.midpoint and row.close > row.open
                        and row.close > df.iloc[i - 1].high and body >= row.atr * 0.20
                        and bar_range > 0 and (row.high - row.close) / bar_range <= 0.35
                        and row.close - pending.midpoint <= row.atr * 0.80
                    )
                    entry = row.close + slippage
                else:
                    confirmed = (
                        recent_touch and row.close < pending.midpoint and row.close < row.open
                        and row.close < df.iloc[i - 1].low and body >= row.atr * 0.20
                        and bar_range > 0 and (row.close - row.low) / bar_range <= 0.35
                        and pending.midpoint - row.close <= row.atr * 0.80
                    )
                    entry = row.close - slippage

                if confirmed:
                    stop_distance = abs(entry - pending.stop)
                    entry_cost = entry * taker_fee + slippage
                    exit_cost = entry * taker_fee + slippage
                    estimated_funding = entry * funding_rate * 2
                    risk_per_unit = stop_distance + entry_cost + exit_cost + estimated_funding
                    target_distance = max(
                        risk_per_unit * 2.0 + entry_cost + exit_cost + estimated_funding,
                        entry * 0.008,
                    )
                    target = entry + target_distance if pending.side == "long" else entry - target_distance
                    qty = min(equity * risk_pct / risk_per_unit, equity / entry)
                    entry_fee = entry * qty * taker_fee
                    equity -= entry_fee
                    position = Position(pending.side, timestamp, i, entry, pending.stop, target, qty, entry_fee)
                    trades_today += 1
                    pending = None
                    continue
                if pending is not None:
                    continue

        if row.low < prior_low.iloc[i] and row.close > prior_low.iloc[i]:
            bull_sweep_index, bull_sweep_low = i, float(row.low)
        if row.high > prior_high.iloc[i] and row.close < prior_high.iloc[i]:
            bear_sweep_index, bear_sweep_high = i, float(row.high)

        daily_loss_hit = equity - day_start_equity <= -equity * 0.01
        protections_ok = trades_today < 2 and consecutive_losses < 2 and i - last_exit > 8 and not daily_loss_hit
        if not protections_ok or pending is not None:
            continue

        candle_1, candle_2 = df.iloc[i - 2], df.iloc[i - 1]
        body_2 = abs(candle_2.close - candle_2.open)
        volume_ok = candle_2.volume >= candle_2.volume_avg * 1.1
        bull_gap_low, bull_gap_high = float(candle_1.high), float(row.low)
        bear_gap_low, bear_gap_high = float(row.high), float(candle_1.low)
        bull_fvg = bull_gap_high > bull_gap_low and bull_gap_high - bull_gap_low >= row.atr * 0.10
        bear_fvg = bear_gap_high > bear_gap_low and bear_gap_high - bear_gap_low >= row.atr * 0.10
        bull_displacement = candle_2.close > candle_2.open and body_2 >= candle_2.atr * 0.80
        bear_displacement = candle_2.close < candle_2.open and body_2 >= candle_2.atr * 0.80
        bull_recent = bull_sweep_index is not None and 1 <= i - bull_sweep_index <= 8
        bear_recent = bear_sweep_index is not None and 1 <= i - bear_sweep_index <= 8
        long_context = row.close_4h > row.ema200_4h and row.ema20_4h > row.ema50_4h
        short_context = row.close_4h < row.ema200_4h and row.ema20_4h < row.ema50_4h

        if (
            direction != "short" and bull_recent and bull_displacement and bull_fvg
            and row.close > structure_high.iloc[i] and volume_ok and long_context
        ):
            midpoint = (bull_gap_low + bull_gap_high) / 2
            stop = bull_sweep_low - row.atr * 0.20
            if stop < midpoint:
                pending = Pending("long", i, i + 12, midpoint, stop)
                setups.append({"time": timestamp, "side": "long", "midpoint": midpoint, "stop": stop})
                bull_sweep_index = None
        elif (
            direction != "long" and bear_recent and bear_displacement and bear_fvg
            and row.close < structure_low.iloc[i] and volume_ok and short_context
        ):
            midpoint = (bear_gap_low + bear_gap_high) / 2
            stop = bear_sweep_high + row.atr * 0.20
            if stop > midpoint:
                pending = Pending("short", i, i + 12, midpoint, stop)
                setups.append({"time": timestamp, "side": "short", "midpoint": midpoint, "stop": stop})
                bear_sweep_index = None

    return pd.DataFrame(trades), pd.DataFrame(curve), pd.DataFrame(setups)


def report(
    trades: pd.DataFrame,
    curve: pd.DataFrame,
    setups: pd.DataFrame,
    split: str,
    direction: str,
    taker_fee_pct: float,
    funding_pct: float,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"intraday_v4_{split}_{direction}"
    trades.to_csv(RESULTS / f"trades_{stem}.csv", index=False)
    setups.to_csv(RESULTS / f"setups_{stem}.csv", index=False)
    curve.to_csv(RESULTS / f"equity_{stem}.csv", index=False)
    if trades.empty:
        print(f"V4 {split} {direction}: aucun trade ({len(setups)} configurations)")
        return

    wins = trades.loc[trades.net_pnl > 0, "net_pnl"]
    losses = trades.loc[trades.net_pnl < 0, "net_pnl"]
    pf = wins.sum() / abs(losses.sum()) if not losses.empty else np.inf
    dd = curve.equity / curve.equity.cummax() - 1
    counts = trades.result.value_counts()
    summary = (
        f"BTC Intraday V4 — {split} — {direction}\n"
        f"Timeframe: 15m | Contexte: 4H | Risque: 0.25%\n"
        f"Frais taker: {taker_fee_pct:.3f}% | Funding conservateur: {funding_pct:.3f}% / 8H\n"
        f"Configurations: {len(setups)} | Trades: {len(trades)}\n"
        f"PnL net: {trades.net_pnl.sum():.2f} USDT\n"
        f"Profit factor: {pf:.3f}\n"
        f"Taux gagnant: {(trades.net_pnl > 0).mean() * 100:.2f}%\n"
        f"Gain moyen: {wins.mean() if not wins.empty else 0:.2f} USDT\n"
        f"Perte moyenne: {losses.mean() if not losses.empty else 0:.2f} USDT\n"
        f"Frais: {trades.fees.sum():.2f} USDT | Funding: {trades.funding.sum():.2f} USDT\n"
        f"Drawdown maximum: {dd.min() * 100:.2f}%\n"
        f"Sorties cible/stop/temps: {counts.get('target', 0)}/{counts.get('stop', 0)}/{counts.get('time', 0)}\n"
    )
    print(summary)
    (RESULTS / f"summary_{stem}.txt").write_text(summary, encoding="utf-8")
    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(curve.datetime), curve.equity)
    plt.title(f"BTC Intraday V4 — {split} — {direction}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / f"equity_{stem}.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC Intraday V4 sur chandelles KuCoin 15m.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--taker-fee", type=float, default=0.035)
    parser.add_argument("--funding", type=float, default=0.010)
    args = parser.parse_args()

    raw = pd.read_csv(args.data, parse_dates=["datetime"])
    prepared = prepare(raw)
    splits = {
        "developpement_2025": prepared.loc[prepared.index < pd.Timestamp("2026-01-01", tz="UTC")],
        "validation_2026": prepared.loc[prepared.index >= pd.Timestamp("2026-01-01", tz="UTC")],
    }
    for split, frame in splits.items():
        for direction in ("both", "long", "short"):
            trades, curve, setups = run(frame, direction, args.taker_fee, args.funding)
            report(trades, curve, setups, split, direction, args.taker_fee, args.funding)


if __name__ == "__main__":
    main()
