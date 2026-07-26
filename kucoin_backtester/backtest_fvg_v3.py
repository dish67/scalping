from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import DEFAULT_DATA, RESULTS, prepare


@dataclass
class PendingOrder:
    side: str
    created_index: int
    expires_index: int
    limit: float
    stop: float
    sweep_level: float
    fvg_low: float
    fvg_high: float
    touched_index: int | None = None


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


def overlaps(low_a: float, high_a: float, low_b: float, high_b: float) -> bool:
    return max(low_a, low_b) <= min(high_a, high_b)


def last_opposite_candle(df: pd.DataFrame, start: int, end: int, bullish_setup: bool) -> tuple[float, float] | None:
    for j in range(end, start - 1, -1):
        row = df.iloc[j]
        opposite = row.close < row.open if bullish_setup else row.close > row.open
        if opposite:
            return float(row.low), float(row.high)
    return None


def run(
    df: pd.DataFrame,
    direction: str,
    maker_fee_pct: float,
    taker_fee_pct: float,
    require_order_block: bool,
    entry_mode: str,
    max_hold_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    initial_capital = 10_000.0
    equity = initial_capital
    maker_fee = maker_fee_pct / 100
    taker_fee = taker_fee_pct / 100
    slippage = 0.2
    risk_pct = 0.005
    liquidity_lookback = 20
    structure_lookback = 10
    sweep_valid_bars = 6
    order_expiry_bars = 12
    min_displacement_atr = 1.0
    min_fvg_atr = 0.10
    stop_buffer_atr = 0.20
    reward_risk = 2.0
    min_target_pct = 0.40

    trades: list[dict] = []
    curve: list[dict] = []
    setups: list[dict] = []
    pending: PendingOrder | None = None
    position: Position | None = None
    last_exit_index = -10_000
    current_day = None
    day_start_equity = equity
    trades_today = 0
    consecutive_losses = 0
    last_bull_sweep_index: int | None = None
    last_bear_sweep_index: int | None = None
    last_bull_sweep_low = np.nan
    last_bear_sweep_high = np.nan

    prior_low = df["low"].shift(1).rolling(liquidity_lookback).min()
    prior_high = df["high"].shift(1).rolling(liquidity_lookback).max()
    structure_high = df["high"].shift(3).rolling(structure_lookback).max()
    structure_low = df["low"].shift(3).rolling(structure_lookback).min()

    for i in range(max(liquidity_lookback, 203), len(df)):
        timestamp = df.index[i]
        row = df.iloc[i]
        day = timestamp.date()
        if current_day != day:
            current_day = day
            day_start_equity = equity
            trades_today = 0
            consecutive_losses = 0

        # Sorties. Si stop et cible sont touchés dans la même bougie, le stop gagne.
        if position is not None and i > position.entry_index:
            if position.side == "long":
                stop_hit = row.low <= position.stop
                target_hit = row.high >= position.target
                if stop_hit:
                    exit_price, exit_reason = position.stop - slippage, "stop"
                elif target_hit:
                    exit_price, exit_reason = position.target - slippage, "target"
                elif max_hold_bars > 0 and i - position.entry_index >= max_hold_bars:
                    exit_price, exit_reason = row.close - slippage, "time"
                else:
                    exit_price, exit_reason = None, None
            else:
                stop_hit = row.high >= position.stop
                target_hit = row.low <= position.target
                if stop_hit:
                    exit_price, exit_reason = position.stop + slippage, "stop"
                elif target_hit:
                    exit_price, exit_reason = position.target + slippage, "target"
                elif max_hold_bars > 0 and i - position.entry_index >= max_hold_bars:
                    exit_price, exit_reason = row.close + slippage, "time"
                else:
                    exit_price, exit_reason = None, None
            if exit_price is not None:
                gross = (exit_price - position.entry) * position.qty
                if position.side == "short":
                    gross = -gross
                exit_fee = exit_price * position.qty * taker_fee
                net = gross - position.entry_fee - exit_fee
                equity += gross - exit_fee
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
                        "net_pnl": net,
                        "bars": i - position.entry_index,
                        "result": exit_reason,
                    }
                )
                consecutive_losses = consecutive_losses + 1 if net < 0 else 0
                last_exit_index = i
                position = None

        curve.append({"datetime": timestamp, "equity": equity})
        if position is not None:
            continue

        # V3: ordre limite au milieu du FVG.
        # V3.1: le prix doit toucher/reprendre le milieu du FVG, puis confirmer
        # avec une bougie de rejet qui casse la bougie précédente.
        if pending is not None:
            invalidated = (
                row.low <= pending.stop if pending.side == "long" else row.high >= pending.stop
            )
            if i > pending.expires_index or invalidated:
                pending = None
            elif i > pending.created_index:
                midpoint_touched = (
                    row.low <= pending.limit if pending.side == "long" else row.high >= pending.limit
                )
                if midpoint_touched and pending.touched_index is None:
                    pending.touched_index = i

                if entry_mode == "limit":
                    entry_confirmed = row.low <= pending.limit <= row.high
                    entry = pending.limit
                    entry_fee_rate = maker_fee
                    entry_slippage = 0.0
                else:
                    bar_range = row.high - row.low
                    body = abs(row.close - row.open)
                    recent_touch = (
                        pending.touched_index is not None
                        and i - pending.touched_index <= 3
                    )
                    if pending.side == "long":
                        rejection = (
                            recent_touch
                            and row.close > pending.limit
                            and row.close > row.open
                            and row.close > df.iloc[i - 1].high
                            and body >= row.atr * 0.20
                            and bar_range > 0
                            and (row.high - row.close) / bar_range <= 0.30
                            and row.close - pending.limit <= row.atr * 0.60
                        )
                        entry = row.close + slippage
                    else:
                        rejection = (
                            recent_touch
                            and row.close < pending.limit
                            and row.close < row.open
                            and row.close < df.iloc[i - 1].low
                            and body >= row.atr * 0.20
                            and bar_range > 0
                            and (row.close - row.low) / bar_range <= 0.30
                            and pending.limit - row.close <= row.atr * 0.60
                        )
                        entry = row.close - slippage
                    entry_confirmed = rejection
                    entry_fee_rate = taker_fee
                    entry_slippage = slippage

                if not entry_confirmed:
                    continue

                stop_distance = abs(entry - pending.stop)
                entry_cost = entry * entry_fee_rate + entry_slippage
                estimated_exit_cost = entry * taker_fee + slippage
                risk_per_unit = stop_distance + entry_cost + estimated_exit_cost
                target_distance = max(
                    reward_risk * risk_per_unit + entry_cost + estimated_exit_cost,
                    entry * min_target_pct / 100,
                )
                target = entry + target_distance if pending.side == "long" else entry - target_distance
                qty = min(equity * risk_pct / risk_per_unit, equity / entry)
                entry_fee = entry * qty * entry_fee_rate
                equity -= entry_fee
                position = Position(pending.side, timestamp, i, entry, pending.stop, target, qty, entry_fee)
                trades_today += 1
                pending = None
                continue

        # Un sweep est mémorisé uniquement après réintégration à la clôture.
        if row.low < prior_low.iloc[i] and row.close > prior_low.iloc[i]:
            last_bull_sweep_index = i
            last_bull_sweep_low = float(row.low)
        if row.high > prior_high.iloc[i] and row.close < prior_high.iloc[i]:
            last_bear_sweep_index = i
            last_bear_sweep_high = float(row.high)

        daily_loss_hit = equity - day_start_equity <= -equity * 0.015
        protections_ok = (
            trades_today < 2
            and consecutive_losses < 3
            and i - last_exit_index > 12
            and not daily_loss_hit
        )
        if not protections_ok or pending is not None:
            continue

        candle_1 = df.iloc[i - 2]
        candle_2 = df.iloc[i - 1]
        body_2 = abs(candle_2.close - candle_2.open)
        volume_ok = candle_2.volume >= candle_2.volume_avg * 1.1
        bull_fvg_low, bull_fvg_high = float(candle_1.high), float(row.low)
        bear_fvg_low, bear_fvg_high = float(row.high), float(candle_1.low)
        bull_fvg = bull_fvg_high > bull_fvg_low and bull_fvg_high - bull_fvg_low >= row.atr * min_fvg_atr
        bear_fvg = bear_fvg_high > bear_fvg_low and bear_fvg_high - bear_fvg_low >= row.atr * min_fvg_atr
        bull_displacement = candle_2.close > candle_2.open and body_2 >= candle_2.atr * min_displacement_atr
        bear_displacement = candle_2.close < candle_2.open and body_2 >= candle_2.atr * min_displacement_atr
        bull_bos = row.close > structure_high.iloc[i]
        bear_bos = row.close < structure_low.iloc[i]
        bull_sweep_recent = (
            last_bull_sweep_index is not None
            and 1 <= i - last_bull_sweep_index <= sweep_valid_bars
        )
        bear_sweep_recent = (
            last_bear_sweep_index is not None
            and 1 <= i - last_bear_sweep_index <= sweep_valid_bars
        )

        bull_ob = (
            last_opposite_candle(df, last_bull_sweep_index, i - 1, True)
            if bull_sweep_recent and last_bull_sweep_index is not None else None
        )
        bear_ob = (
            last_opposite_candle(df, last_bear_sweep_index, i - 1, False)
            if bear_sweep_recent and last_bear_sweep_index is not None else None
        )
        bull_ob_ok = not require_order_block or (
            bull_ob is not None and overlaps(bull_fvg_low, bull_fvg_high, bull_ob[0], bull_ob[1])
        )
        bear_ob_ok = not require_order_block or (
            bear_ob is not None and overlaps(bear_fvg_low, bear_fvg_high, bear_ob[0], bear_ob[1])
        )

        if (
            direction != "short"
            and bull_sweep_recent and bull_displacement and bull_bos and bull_fvg
            and bull_ob_ok and volume_ok
        ):
            limit = (bull_fvg_low + bull_fvg_high) / 2
            stop = last_bull_sweep_low - row.atr * stop_buffer_atr
            if stop < limit:
                pending = PendingOrder(
                    "long", i, i + order_expiry_bars, limit, stop, prior_low.iloc[last_bull_sweep_index],
                    bull_fvg_low, bull_fvg_high,
                )
                setups.append(
                    {"time": timestamp, "side": "long", "limit": limit, "stop": stop,
                     "fvg_low": bull_fvg_low, "fvg_high": bull_fvg_high, "order_block": bull_ob}
                )
                last_bull_sweep_index = None
        elif (
            direction != "long"
            and bear_sweep_recent and bear_displacement and bear_bos and bear_fvg
            and bear_ob_ok and volume_ok
        ):
            limit = (bear_fvg_low + bear_fvg_high) / 2
            stop = last_bear_sweep_high + row.atr * stop_buffer_atr
            if stop > limit:
                pending = PendingOrder(
                    "short", i, i + order_expiry_bars, limit, stop, prior_high.iloc[last_bear_sweep_index],
                    bear_fvg_low, bear_fvg_high,
                )
                setups.append(
                    {"time": timestamp, "side": "short", "limit": limit, "stop": stop,
                     "fvg_low": bear_fvg_low, "fvg_high": bear_fvg_high, "order_block": bear_ob}
                )
                last_bear_sweep_index = None

    return pd.DataFrame(trades), pd.DataFrame(curve), pd.DataFrame(setups)


def report(
    trades: pd.DataFrame,
    curve: pd.DataFrame,
    setups: pd.DataFrame,
    direction: str,
    variant: str,
    maker_fee_pct: float,
    taker_fee_pct: float,
    entry_mode: str,
    max_hold_bars: int,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if entry_mode == "confirm" and max_hold_bars > 0:
        version = f"v32_confirm_{max_hold_bars}bars"
    else:
        version = "v31_confirm" if entry_mode == "confirm" else "v3_limit"
    stem = f"fvg_{version}_{variant}_{direction}"
    trades.to_csv(RESULTS / f"trades_{stem}.csv", index=False)
    curve.to_csv(RESULTS / f"equity_{stem}.csv", index=False)
    setups.to_csv(RESULTS / f"setups_{stem}.csv", index=False)
    if trades.empty:
        print(f"{version} {variant} {direction}: aucun trade ({len(setups)} configurations)")
        return

    wins = trades.loc[trades.net_pnl > 0, "net_pnl"]
    losses = trades.loc[trades.net_pnl < 0, "net_pnl"]
    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty else np.inf
    drawdown = curve.equity / curve.equity.cummax() - 1
    exit_counts = trades["result"].value_counts()
    exit_month = pd.to_datetime(trades.exit_time, utc=True).dt.tz_convert(None).dt.to_period("M")
    monthly = trades.assign(month=exit_month).groupby("month").agg(
        trades=("net_pnl", "size"),
        pnl=("net_pnl", "sum"),
        win_rate=("net_pnl", lambda values: (values > 0).mean() * 100),
    )
    monthly.to_csv(RESULTS / f"monthly_{stem}.csv")
    display_version = (
        "V3.2 scalping"
        if entry_mode == "confirm" and max_hold_bars > 0
        else "V3.1 confirmation"
        if entry_mode == "confirm"
        else "V3 limite"
    )
    summary = (
        f"BTC Liquidity FVG {display_version}"
        f" — {variant} — {direction}\n"
        f"Frais entrée {'taker' if entry_mode == 'confirm' else 'maker'}: "
        f"{taker_fee_pct if entry_mode == 'confirm' else maker_fee_pct:.3f}%"
        f" | sortie taker: {taker_fee_pct:.3f}%\n"
        f"Durée maximum: {'illimitée' if max_hold_bars <= 0 else str(max_hold_bars) + ' bougies'}\n"
        f"Configurations: {len(setups)} | Trades exécutés: {len(trades)}\n"
        f"PnL net: {trades.net_pnl.sum():.2f} USDT\n"
        f"Profit factor: {profit_factor:.3f}\n"
        f"Taux gagnant: {(trades.net_pnl > 0).mean() * 100:.2f}%\n"
        f"Gain moyen: {wins.mean() if not wins.empty else 0:.2f} USDT\n"
        f"Perte moyenne: {losses.mean() if not losses.empty else 0:.2f} USDT\n"
        f"Frais: {trades.fees.sum():.2f} USDT\n"
        f"Drawdown maximum: {drawdown.min() * 100:.2f}%\n"
        f"Sorties cible/stop/temps: {exit_counts.get('target', 0)}/"
        f"{exit_counts.get('stop', 0)}/{exit_counts.get('time', 0)}\n"
    )
    (RESULTS / f"summary_{stem}.txt").write_text(summary, encoding="utf-8")
    print(summary)
    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(curve.datetime), curve.equity)
    plt.title(f"BTC Liquidity FVG {version} — {variant} — {direction}")
    plt.ylabel("Capital (USDT)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / f"equity_{stem}.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest V3: sweep de liquidité, BOS, FVG et order block.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--direction", choices=["both", "long", "short", "all"], default="all")
    parser.add_argument("--maker-fee", type=float, default=0.020)
    parser.add_argument("--taker-fee", type=float, default=0.035)
    parser.add_argument("--no-order-block", action="store_true", help="Teste FVG sans exiger le chevauchement OB.")
    parser.add_argument(
        "--entry",
        choices=["limit", "confirm"],
        default="limit",
        help="limit=V3; confirm=V3.1 avec confirmation après retest.",
    )
    parser.add_argument(
        "--max-hold-bars",
        type=int,
        default=0,
        help="Sortie au marché après N bougies; 0 désactive la limite.",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.data, parse_dates=["datetime"])
    prepared = prepare(raw)
    directions = ["both", "long", "short"] if args.direction == "all" else [args.direction]
    variant = "fvg_only" if args.no_order_block else "fvg_ob"
    for direction in directions:
        trades, curve, setups = run(
            prepared, direction, args.maker_fee, args.taker_fee,
            not args.no_order_block, args.entry, args.max_hold_bars,
        )
        report(
            trades, curve, setups, direction, variant,
            args.maker_fee, args.taker_fee, args.entry, args.max_hold_bars,
        )


if __name__ == "__main__":
    main()
