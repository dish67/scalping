from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "XBTUSDTM_5m.csv"
RESULTS = ROOT / "results"


@dataclass
class Position:
    side: str
    entry_time: pd.Timestamp
    entry: float
    stop: float
    target: float
    qty: float
    entry_fee: float
    entry_index: int


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().set_index("datetime").sort_index()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = rsi(df["close"])
    previous_close = df["close"].shift()
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    df["volume_avg"] = df["volume"].rolling(20).mean()
    day = df.index.floor("D")
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical * df["volume"]).groupby(day).cumsum() / df["volume"].groupby(day).cumsum()

    hourly = df.resample("1h", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    hourly["htf20"] = hourly["close"].ewm(span=20, adjust=False).mean()
    hourly["htf50"] = hourly["close"].ewm(span=50, adjust=False).mean()
    hourly["htf200"] = hourly["close"].ewm(span=200, adjust=False).mean()
    # Une heure de décalage: seules les données HTF clôturées sont visibles.
    htf = hourly[["close", "htf20", "htf50", "htf200"]].shift(1)
    htf.columns = ["htf_close", "htf20", "htf50", "htf200"]
    df = pd.merge_asof(
        df.reset_index().sort_values("datetime"),
        htf.reset_index().sort_values("datetime"),
        on="datetime",
        direction="backward",
    ).set_index("datetime")
    df["htf50_past"] = df["htf50"].shift(36)  # 3 heures sur des bougies de 5 minutes.
    return df.dropna().copy()


def run(df: pd.DataFrame, direction: str, fee_pct: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    initial_capital = 10_000.0
    equity = initial_capital
    fee_rate = fee_pct / 100
    slippage = 0.2
    risk_pct = 0.005
    trades: list[dict] = []
    curve: list[dict] = []
    position: Position | None = None
    last_exit_index = -10_000
    trades_today = 0
    consecutive_losses = 0
    current_day = None
    day_start_equity = equity

    ema_touch = (
        (df["low"] <= df["ema20"] * 1.001) & (df["high"] >= df["ema20"] * 0.999)
    ) | ((df["low"] <= df["vwap"] * 1.001) & (df["high"] >= df["vwap"] * 0.999))
    pullback_recent = ema_touch.rolling(5).max().fillna(0).astype(bool)
    breakout_high = df["high"].shift(1).rolling(2).max()
    breakout_low = df["low"].shift(1).rolling(2).min()
    swing_low = df["low"].rolling(7).min()
    swing_high = df["high"].rolling(7).max()

    for i in range(3, len(df)):
        timestamp = df.index[i]
        row = df.iloc[i]
        day = timestamp.date()
        if current_day != day:
            current_day = day
            trades_today = 0
            consecutive_losses = 0
            day_start_equity = equity

        if position is not None and i > position.entry_index:
            if position.side == "long":
                stop_hit, target_hit = row.low <= position.stop, row.high >= position.target
                exit_price = position.stop - slippage if stop_hit else position.target - slippage if target_hit else None
            else:
                stop_hit, target_hit = row.high >= position.stop, row.low <= position.target
                exit_price = position.stop + slippage if stop_hit else position.target + slippage if target_hit else None

            # Si stop et cible sont touchés dans la même bougie, choix conservateur: stop.
            if exit_price is not None:
                gross = (exit_price - position.entry) * position.qty
                if position.side == "short":
                    gross = -gross
                exit_fee = exit_price * position.qty * fee_rate
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
                        "return_pct": net / initial_capital * 100,
                        "bars": i - position.entry_index,
                    }
                )
                consecutive_losses = consecutive_losses + 1 if net < 0 else 0
                position = None
                last_exit_index = i

        curve.append({"datetime": timestamp, "equity": equity})
        if position is not None:
            continue

        daily_loss_hit = equity - day_start_equity <= -equity * 0.015
        protected = trades_today < 3 and consecutive_losses < 3 and i - last_exit_index > 12 and not daily_loss_hit
        if not protected:
            continue

        bar_range = row.high - row.low
        body = abs(row.close - row.open)
        volume_ok = row.volume >= row.volume_avg * 1.1
        volatility_ok = 0.05 <= row.atr / row.close * 100 <= 0.50
        long_trend = row.htf_close > row.htf200 and row.htf20 > row.htf50 > row.htf50_past
        short_trend = row.htf_close < row.htf200 and row.htf20 < row.htf50 < row.htf50_past
        long_local = row.ema20 > row.ema50 and row.ema50 > df.iloc[i - 3].ema50
        short_local = row.ema20 < row.ema50 and row.ema50 < df.iloc[i - 3].ema50
        long_quality = bar_range > 0 and body >= row.atr * 0.20 and (row.high - row.close) / bar_range <= 0.30
        short_quality = bar_range > 0 and body >= row.atr * 0.20 and (row.close - row.low) / bar_range <= 0.30
        long_entry = (
            direction != "short" and long_trend and long_local and pullback_recent.iloc[i]
            and row.close > row.open and row.close > breakout_high.iloc[i]
            and row.close > row.ema20 and row.close > row.vwap and 50 <= row.rsi <= 70
            and volume_ok and volatility_ok and row.close - row.ema20 <= row.atr * 1.2 and long_quality
        )
        short_entry = (
            direction != "long" and short_trend and short_local and pullback_recent.iloc[i]
            and row.close < row.open and row.close < breakout_low.iloc[i]
            and row.close < row.ema20 and row.close < row.vwap and 30 <= row.rsi <= 50
            and volume_ok and volatility_ok and row.ema20 - row.close <= row.atr * 1.2 and short_quality
        )
        if not (long_entry or short_entry):
            continue

        entry = row.close + slippage if long_entry else row.close - slippage
        stop = min(swing_low.iloc[i] - row.atr * 0.30, entry - row.atr * 0.80) if long_entry else max(
            swing_high.iloc[i] + row.atr * 0.30, entry + row.atr * 0.80
        )
        stop_distance = abs(entry - stop)
        round_trip_cost = entry * fee_rate * 2 + slippage * 2
        risk_per_unit = stop_distance + round_trip_cost
        target_distance = max(risk_per_unit * 1.5 + round_trip_cost, entry * 0.004, round_trip_cost * 3)
        if target_distance > row.atr * 8:
            continue
        target = entry + target_distance if long_entry else entry - target_distance
        qty = min(equity * risk_pct / risk_per_unit, equity / entry)
        entry_fee = entry * qty * fee_rate
        equity -= entry_fee
        position = Position("long" if long_entry else "short", timestamp, entry, stop, target, qty, entry_fee, i)
        trades_today += 1

    return pd.DataFrame(trades), pd.DataFrame(curve)


def report(trades: pd.DataFrame, curve: pd.DataFrame, direction: str, fee_pct: float) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    trades.to_csv(RESULTS / f"trades_{direction}.csv", index=False)
    curve.to_csv(RESULTS / f"equity_{direction}.csv", index=False)
    if trades.empty:
        print(f"{direction}: aucun trade")
        return

    wins = trades.loc[trades.net_pnl > 0, "net_pnl"]
    losses = trades.loc[trades.net_pnl < 0, "net_pnl"]
    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty else np.inf
    equity = curve["equity"]
    drawdown = equity / equity.cummax() - 1
    exit_month = pd.to_datetime(trades.exit_time, utc=True).dt.tz_convert(None).dt.to_period("M")
    monthly = trades.assign(month=exit_month).groupby("month").agg(
        trades=("net_pnl", "size"), pnl=("net_pnl", "sum"), win_rate=("net_pnl", lambda x: (x > 0).mean() * 100)
    )
    monthly.to_csv(RESULTS / f"monthly_{direction}.csv")

    summary = (
        f"Direction: {direction}\n"
        f"Commission par ordre: {fee_pct:.3f}%\n"
        f"Trades: {len(trades)}\n"
        f"PnL net: {trades.net_pnl.sum():.2f} USDT\n"
        f"Profit factor: {profit_factor:.3f}\n"
        f"Taux gagnant: {(trades.net_pnl > 0).mean() * 100:.2f}%\n"
        f"Gain moyen: {wins.mean() if not wins.empty else 0:.2f} USDT\n"
        f"Perte moyenne: {losses.mean() if not losses.empty else 0:.2f} USDT\n"
        f"Frais: {trades.fees.sum():.2f} USDT\n"
        f"Drawdown maximum: {drawdown.min() * 100:.2f}%\n"
    )
    (RESULTS / f"summary_{direction}.txt").write_text(summary, encoding="utf-8")
    print(summary)
    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(curve.datetime), curve.equity)
    plt.title(f"BTC MTF V2.2 — {direction}")
    plt.ylabel("Capital (USDT)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS / f"equity_{direction}.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest local de BTC MTF Pullback V2.2.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--direction", choices=["both", "long", "short", "all"], default="all")
    parser.add_argument("--fee", type=float, default=0.06, help="Commission par ordre en pourcentage.")
    args = parser.parse_args()
    raw = pd.read_csv(args.data, parse_dates=["datetime"])
    prepared = prepare(raw)
    directions = ["both", "long", "short"] if args.direction == "all" else [args.direction]
    for direction in directions:
        trades, curve = run(prepared, direction, args.fee)
        report(trades, curve, direction, args.fee)


if __name__ == "__main__":
    main()
