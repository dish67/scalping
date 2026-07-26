from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


API_URL = "https://api-futures.kucoin.com/api/v1/kline/query"
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "XBTUSDTM_5m.csv"
# L'API historique annonce jusqu'a 500 bougies, mais certaines reponses
# Futures sont actuellement tronquees autour de 200. Des fenetres plus petites
# evitent d'avancer le curseur au-dela des bougies effectivement retournees.
MAX_CANDLES_PER_REQUEST = 190


def utc_ms(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def download(symbol: str, granularity: int, start_ms: int, end_ms: int) -> pd.DataFrame:
    interval_ms = granularity * 60_000
    page_span = (MAX_CANDLES_PER_REQUEST - 1) * interval_ms
    rows: dict[int, list[float]] = {}
    cursor = start_ms
    session = requests.Session()

    while cursor <= end_ms:
        page_end = min(cursor + page_span, end_ms)
        response = session.get(
            API_URL,
            params={
                "symbol": symbol,
                "granularity": granularity,
                "from": cursor,
                "to": page_end,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "200000":
            raise RuntimeError(f"KuCoin: {payload}")

        for item in payload.get("data", []):
            timestamp = int(item[0])
            rows[timestamp] = [
                timestamp,
                float(item[1]),
                float(item[2]),
                float(item[3]),
                float(item[4]),
                float(item[5]),
                float(item[6]),
            ]

        cursor = page_end + interval_ms
        print(f"\rTéléchargement: {datetime.fromtimestamp(page_end / 1000, timezone.utc):%Y-%m-%d}", end="")
        time.sleep(0.08)

    print()
    columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    frame = pd.DataFrame(rows.values(), columns=columns).sort_values("timestamp")
    frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame.drop_duplicates("timestamp").reset_index(drop=True)


def continuity_gaps(frame: pd.DataFrame, granularity: int) -> pd.DataFrame:
    if len(frame) < 2:
        return pd.DataFrame()
    interval_ms = granularity * 60_000
    gaps = frame.loc[frame["timestamp"].diff() > interval_ms, ["timestamp", "datetime"]].copy()
    if gaps.empty:
        return gaps
    gaps["previous_datetime"] = frame["datetime"].shift().loc[gaps.index]
    gaps["missing_candles"] = (
        frame["timestamp"].diff().loc[gaps.index].floordiv(interval_ms).astype(int) - 1
    )
    return gaps[["previous_datetime", "datetime", "missing_candles"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Télécharge les bougies KuCoin Futures publiques.")
    parser.add_argument("--symbol", default="XBTUSDTM")
    parser.add_argument("--granularity", type=int, default=5, help="Minutes par bougie.")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-gaps",
        action="store_true",
        help="Enregistre quand meme le fichier si des bougies internes manquent.",
    )
    args = parser.parse_args()

    start_ms = utc_ms(args.start)
    end_ms = utc_ms(args.end) + 86_399_999
    frame = download(args.symbol, args.granularity, start_ms, end_ms)
    gaps = continuity_gaps(frame, args.granularity)
    if not gaps.empty:
        missing = int(gaps["missing_candles"].sum())
        sample = gaps.head(5).to_string(index=False)
        message = (
            f"{len(gaps)} trou(s) detecte(s), soit environ {missing} bougie(s) manquante(s).\n"
            f"{sample}"
        )
        if not args.allow_gaps:
            raise RuntimeError(
                message
                + "\nFichier non enregistre. Relancez seulement avec --allow-gaps "
                  "si ces absences sont normales pour le marche."
            )
        print(f"AVERTISSEMENT: {message}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"{len(frame):,} bougies enregistrées dans {args.output}")
    print(f"Période: {frame['datetime'].iloc[0]} → {frame['datetime'].iloc[-1]}")


if __name__ == "__main__":
    main()
