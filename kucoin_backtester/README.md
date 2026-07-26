# Backtester KuCoin BTC MTF Pullback V2.2

Ce projet utilise uniquement l'API publique KuCoin Futures. Aucune clé API n'est requise.

## Installation WSL

```bash
cd /mnt/d/codex/scalping
source .venv-wsl/bin/activate
python -m pip install -r kucoin_backtester/requirements.txt
```

## Télécharger les bougies

```bash
python kucoin_backtester/download_data.py --start 2025-01-01
```

## Lancer les trois tests

```bash
python kucoin_backtester/backtest.py
```

Les résultats sont créés dans `kucoin_backtester/results/`.

Pour simuler les ordres taker BTCC VIP3 à 0,035 % par ordre :

```bash
python kucoin_backtester/backtest.py --fee 0.035
```

Pour un seul sens :

```bash
python kucoin_backtester/backtest.py --direction long
python kucoin_backtester/backtest.py --direction short
```

## V3 : liquidité, FVG et order block

La V3 utilise par défaut les frais BTCC VIP3 : entrée maker à 0,020 % et
sortie taker à 0,035 %.

```bash
python kucoin_backtester/backtest_fvg_v3.py
```

Pour comparer la même logique sans exiger le chevauchement avec un order block :

```bash
python kucoin_backtester/backtest_fvg_v3.py --no-order-block
```

## V3.1 : confirmation après le retest

La V3.1 attend une bougie de rejet et une cassure de la bougie précédente
après le retour au milieu du FVG. L'entrée et la sortie sont simulées en taker.

```bash
python kucoin_backtester/backtest_fvg_v3.py --entry confirm
```

Comparaison sans order block :

```bash
python kucoin_backtester/backtest_fvg_v3.py --entry confirm --no-order-block
```

## V3.2 : durée maximale de scalping

Test Long seulement, FVG sans order block, maximum 24 bougies (2 heures) :

```bash
python kucoin_backtester/backtest_fvg_v3.py --entry confirm --no-order-block --direction long --max-hold-bars 24
```

Comparaison avec 12 bougies (1 heure) :

```bash
python kucoin_backtester/backtest_fvg_v3.py --entry confirm --no-order-block --direction long --max-hold-bars 12
```

## V4 : intraday 15 minutes avec contexte 4 heures

Télécharger les bougies 15 minutes :

```bash
python kucoin_backtester/download_data.py --granularity 15 --start 2025-01-01 --output kucoin_backtester/data/XBTUSDTM_15m.csv
```

Lancer le test séparé entre développement 2025 et validation 2026 :

```bash
python kucoin_backtester/backtest_intraday_v4.py
```

## V5 : tendance simple Donchian sur 4 heures

Télécharger les bougies 4 heures :

```bash
python kucoin_backtester/download_data.py --granularity 240 --start 2025-01-01 --output kucoin_backtester/data/XBTUSDTM_4h.csv
```

Lancer les modèles Donchian 20/10 et 55/20 :

```bash
python kucoin_backtester/backtest_trend_v5.py
```

## V6 : régime journalier SMA 200

Télécharger une série journalière continue dans un nouveau fichier :

```bash
python kucoin_backtester/download_data.py \
  --granularity 1440 \
  --start 2020-01-01 \
  --output kucoin_backtester/data/XBTUSDTM_1d_clean.csv
```

Le téléchargement et le backtest refusent par défaut les séries contenant
des trous. Ne pas utiliser `--allow-gaps` pour un test de validation.

Lancer le scénario BTCC VIP3 et le scénario de coûts stressés :

```bash
python kucoin_backtester/backtest_daily_regime_v6.py \
  --data kucoin_backtester/data/XBTUSDTM_1d_clean.csv
```

V6.1 utilise une seule règle préétablie : clôture journalière au-dessus de la
SMA 200 pour être long, sinon cash. Tout signal est exécuté au prochain open.
La simulation reste continue aux frontières annuelles : le capital et une
position éventuellement ouverte ne sont jamais réinitialisés le 1er janvier.
