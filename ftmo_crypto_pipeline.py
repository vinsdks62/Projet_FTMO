import os
import gzip
import json
import time
import math
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import ccxt


# =========================
# CONFIG / SPECS
# =========================

@dataclass
class MarketSpec:
    symbol: str = "BTC/USDT"
    exchange: str = "binance"
    timeframe: str = "1m"
    # frais/slippage en fraction (0.0005 = 5 bps)
    taker_fee: float = 0.0004      # approximatif, à stresser
    slippage: float = 0.0002       # approximatif, à stresser


@dataclass
class StrategySpec:
    # stratégie (version crypto)
    ema_fast: int = 20
    ema_slow: int = 50
    sl_pct: float = 0.0020         # 0.20% stop (à calibrer)
    tp_pct: float = 0.0030         # 0.30% TP
    max_trades_per_day: int = 2
    max_consecutive_losses: int = 2
    daily_stop_r: float = -1.0     # stop si pnl_jour <= -1R


@dataclass
class FTMOConfig:
    starting_balance: float = 10_000.0
    max_daily_loss_pct: float = 0.05   # 5%
    max_total_loss_pct: float = 0.10   # 10%
    # règle conservative: contrôle intrabar sur low/high
    conservative_intrabar: bool = True


@dataclass
class Trade:
    day: str
    side: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    pnl_usd: float
    pnl_pct: float
    r: float
    reason: str
    mfe_pct: float
    mae_pct: float
    equity_after: float


# =========================
# UTILS
# =========================

def iso(d: date) -> str:
    return d.isoformat()

def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def utc_day_start(d: date) -> pd.Timestamp:
    return pd.Timestamp(d.isoformat() + " 00:00:00", tz="UTC")

def utc_day_end(d: date) -> pd.Timestamp:
    return pd.Timestamp(d.isoformat() + " 23:59:00", tz="UTC")


# =========================
# DATA DOWNLOAD (CCXT)
# =========================

def make_exchange(name: str):
    ex_cls = getattr(ccxt, name)
    ex = ex_cls({"enableRateLimit": True})
    return ex

def fetch_ohlcv_1m_day(ex, symbol: str, day: date, logger_print=True) -> pd.DataFrame:
    """
    Télécharge 1 journée UTC en 1m via pagination since.
    CCXT ohlcv: [ms, open, high, low, close, volume]
    """
    start = int(utc_day_start(day).timestamp() * 1000)
    end = int((utc_day_end(day) + pd.Timedelta(minutes=1)).timestamp() * 1000)

    all_rows = []
    since = start
    limit = 1000  # binance: 1000
    loops = 0

    while since < end:
        loops += 1
        batch = ex.fetch_ohlcv(symbol, timeframe="1m", since=since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)

        last_ms = batch[-1][0]
        # avance d'1 minute pour éviter doublon
        since = last_ms + 60_000

        # sécurité anti-boucle
        if loops > 2000:
            break

        # rate limit
        if ex.rateLimit:
            time.sleep(ex.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()

    # filtre strict sur la journée
    df = df[(df.index >= utc_day_start(day)) & (df.index <= utc_day_end(day))]
    return df

def save_day_csv_gz(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    with gzip.open(path, "wt", encoding="utf-8") as f:
        df.to_csv(f)

def load_day_csv_gz(path: str) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    return df

def qa_day(df: pd.DataFrame, day: date) -> Dict:
    if df.empty:
        return {"ok": False, "reason": "empty"}

    full = pd.date_range(utc_day_start(day), utc_day_end(day), freq="1min", tz="UTC")
    missing = full.difference(df.index.floor("min"))
    miss_ratio = len(missing) / len(full)

    # returns sanity
    closes = df["close"].astype(float).values
    rets = np.diff(np.log(np.maximum(closes, 1e-12)))
    # outlier metric
    if len(rets) > 10:
        z = np.nanmax(np.abs((rets - np.nanmean(rets)) / (np.nanstd(rets) + 1e-12)))
    else:
        z = 0.0

    return {
        "ok": True,
        "bars": int(len(df)),
        "missing_minutes": int(len(missing)),
        "missing_ratio": float(miss_ratio),
        "max_abs_ret_z": float(z),
    }


# =========================
# INDICATORS / RESAMPLE
# =========================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def resample_2m(df_1m: pd.DataFrame) -> pd.DataFrame:
    df = df_1m.resample("2min", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    return df

def session_vwap_5m(df_1m: pd.DataFrame, day: date) -> pd.Series:
    df5 = df_1m.resample("5min", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    # session = jour UTC
    sess = df5[(df5.index >= utc_day_start(day)) & (df5.index <= utc_day_end(day))]
    if sess.empty:
        return pd.Series(dtype=float)
    pv = (sess["close"] * sess["volume"]).cumsum()
    vv = sess["volume"].cumsum().replace(0, np.nan)
    return pv / vv


# =========================
# FTMO RISK SIM (STRICT)
# =========================

@dataclass
class FTMOState:
    equity: float
    start_equity: float
    daily_start_equity: float
    failed: bool = False
    fail_reason: Optional[str] = None

    @staticmethod
    def init(cfg: FTMOConfig) -> "FTMOState":
        return FTMOState(
            equity=cfg.starting_balance,
            start_equity=cfg.starting_balance,
            daily_start_equity=cfg.starting_balance
        )

def ftmo_limits(cfg: FTMOConfig, st: FTMOState) -> Tuple[float, float]:
    """
    Renvoie les floors:
    - daily_floor = daily_start_equity * (1 - max_daily_loss_pct)
    - total_floor = start_equity * (1 - max_total_loss_pct)
    """
    daily_floor = st.daily_start_equity * (1.0 - cfg.max_daily_loss_pct)
    total_floor = st.start_equity * (1.0 - cfg.max_total_loss_pct)
    return daily_floor, total_floor

def ftmo_check_fail(cfg: FTMOConfig, st: FTMOState, equity_worst: float, when: str):
    if st.failed:
        return
    daily_floor, total_floor = ftmo_limits(cfg, st)
    if equity_worst <= daily_floor:
        st.failed = True
        st.fail_reason = f"DAILY_LOSS_TOUCH@{when}"
        return
    if equity_worst <= total_floor:
        st.failed = True
        st.fail_reason = f"MAX_LOSS_TOUCH@{when}"
        return


# =========================
# EXECUTION MODEL (SL/TP %)
# =========================

def trade_pnl_usd(entry: float, exit: float, side: str, notional_usd: float) -> float:
    """
    1x notionnel (pas de levier ici): pnl = notional * %move
    """
    if side == "LONG":
        pct = (exit - entry) / entry
    else:
        pct = (entry - exit) / entry
    return notional_usd * pct

def apply_costs(pnl_usd: float, entry: float, exit: float, mkt: MarketSpec, notional_usd: float) -> float:
    """
    Coûts simplifiés:
    - taker fee sur entrée + sortie: notional * fee * 2
    - slippage: défavorable sur entrée et sortie -> approx notional * slippage * 2
    """
    fees = notional_usd * mkt.taker_fee * 2.0
    slip = notional_usd * mkt.slippage * 2.0
    return pnl_usd - fees - slip


# =========================
# STRATEGY BACKTEST (2m signal + 1m management)
# =========================

def backtest_day(
    day: date,
    df_1m: pd.DataFrame,
    mkt: MarketSpec,
    strat: StrategySpec,
    ftmo_cfg: FTMOConfig,
    st: FTMOState,
    verbose=True
) -> List[Trade]:

    # reset daily start equity
    st.daily_start_equity = st.equity

    trades: List[Trade] = []
    consec_losses = 0
    day_pnl = 0.0
    trades_taken = 0

    df_2m = resample_2m(df_1m)
    df_2m["EMA_FAST"] = ema(df_2m["close"], strat.ema_fast)
    df_2m["EMA_SLOW"] = ema(df_2m["close"], strat.ema_slow)

    vwap5 = session_vwap_5m(df_1m, day)
    df_2m["VWAP"] = vwap5.reindex(df_2m.index).ffill()

    # boucle sur barres 2m (entrées à la clôture)
    idx = df_2m.index.to_list()
    if len(idx) < 3:
        return trades

    # notionnel = equity (1x). C’est volontairement conservateur / simple.
    # On ajustera ensuite (risk per trade fixe, etc.).
    def current_notional():
        return st.equity

    for i in range(2, len(idx)):
        if st.failed:
            break

        # règles journalières
        R_usd = current_notional() * strat.sl_pct  # 1R en $
        if trades_taken >= strat.max_trades_per_day:
            break
        if consec_losses >= strat.max_consecutive_losses:
            break
        if day_pnl <= strat.daily_stop_r * R_usd:
            break

        row = df_2m.loc[idx[i]]
        prev = df_2m.loc[idx[i-1]]

        if np.isnan(row["VWAP"]):
            continue

        close = float(row["close"])
        vwap = float(row["VWAP"])
        emaf = float(row["EMA_FAST"])
        emas = float(row["EMA_SLOW"])

        long_trend = (close > vwap) and (emaf > emas)
        short_trend = (close < vwap) and (emaf < emas)

        side = None
        # pullback vers EMA: on utilise la bougie précédente
        if long_trend and (float(prev["low"]) <= float(prev["EMA_FAST"])) and (close > float(prev["high"])):
            side = "LONG"
        elif short_trend and (float(prev["high"]) >= float(prev["EMA_FAST"])) and (close < float(prev["low"])):
            side = "SHORT"
        else:
            continue

        entry_time = idx[i]
        entry = close

        # SL/TP en prix
        if side == "LONG":
            sl = entry * (1.0 - strat.sl_pct)
            tp = entry * (1.0 + strat.tp_pct)
        else:
            sl = entry * (1.0 + strat.sl_pct)
            tp = entry * (1.0 - strat.tp_pct)

        if verbose:
            print(f"[ENTRY] {iso(day)} {side} t={entry_time} entry={entry:.2f} SL={sl:.2f} TP={tp:.2f} equity={st.equity:.2f}")

        # gestion en 1m après entry_time, jusqu’à fin de journée
        fwd = df_1m[df_1m.index > entry_time]
        if fwd.empty:
            break

        mfe = 0.0
        mae = 0.0
        exit_reason = "EOD"
        exit_time = fwd.index[-1]
        exit_price = float(fwd.iloc[-1]["close"])

        notional = current_notional()

        for ts, bar in fwd.iterrows():
            hi = float(bar["high"]); lo = float(bar["low"])

            # MFE/MAE en %
            if side == "LONG":
                mfe = max(mfe, (hi - entry) / entry)
                mae = min(mae, (lo - entry) / entry)
                hit_sl = lo <= sl
                hit_tp = hi >= tp
                # equity worst-case intrabar (conservateur)
                worst_exit = sl if hit_sl else lo
                equity_worst = st.equity + apply_costs(trade_pnl_usd(entry, worst_exit, side, notional), entry, worst_exit, mkt, notional)
            else:
                mfe = max(mfe, (entry - lo) / entry)
                mae = min(mae, (entry - hi) / entry)
                hit_sl = hi >= sl
                hit_tp = lo <= tp
                worst_exit = sl if hit_sl else hi
                equity_worst = st.equity + apply_costs(trade_pnl_usd(entry, worst_exit, side, notional), entry, worst_exit, mkt, notional)

            if ftmo_cfg.conservative_intrabar:
                ftmo_check_fail(ftmo_cfg, st, equity_worst, when=str(ts))
                if st.failed:
                    if verbose:
                        print(f"[FTMO FAIL] {st.fail_reason} | equity_worst={equity_worst:.2f}")
                    return trades  # stop immédiat

            # sortie SL/TP (conservateur: si SL & TP touchés => SL)
            if hit_sl and hit_tp:
                exit_reason = "SL"
                exit_time = ts
                exit_price = sl
                break
            elif hit_sl:
                exit_reason = "SL"
                exit_time = ts
                exit_price = sl
                break
            elif hit_tp:
                exit_reason = "TP"
                exit_time = ts
                exit_price = tp
                break

        pnl_gross = trade_pnl_usd(entry, exit_price, side, notional)
        pnl_net = apply_costs(pnl_gross, entry, exit_price, mkt, notional)
        pnl_pct = pnl_net / max(1e-12, notional)

        # applique
        st.equity += pnl_net
        day_pnl += pnl_net

        # update streak
        if pnl_net < 0:
            consec_losses += 1
        else:
            consec_losses = 0

        trades_taken += 1

        # check FTMO sur equity post-trade (réalisée)
        ftmo_check_fail(ftmo_cfg, st, st.equity, when=f"POST_TRADE@{exit_time}")
        if st.failed:
            if verbose:
                print(f"[FTMO FAIL] {st.fail_reason} | equity={st.equity:.2f}")
            return trades

        # R multiple
        r = pnl_net / max(1e-12, (notional * strat.sl_pct))

        tr = Trade(
            day=iso(day),
            side=side,
            entry_time=str(entry_time),
            exit_time=str(exit_time),
            entry=float(entry),
            exit=float(exit_price),
            pnl_usd=float(pnl_net),
            pnl_pct=float(pnl_pct),
            r=float(r),
            reason=exit_reason,
            mfe_pct=float(mfe),
            mae_pct=float(mae),
            equity_after=float(st.equity),
        )
        trades.append(tr)

        if verbose:
            print(f"[EXIT ] {iso(day)} {side} reason={exit_reason} pnl_net={pnl_net:.2f} r={r:.3f} equity={st.equity:.2f}")

    return trades


# =========================
# METRICS + MONTE CARLO (bootstrap jours)
# =========================

def compute_metrics(trades: List[Trade]) -> Dict:
    if not trades:
        return {"n_trades": 0}

    pnl = np.array([t.pnl_usd for t in trades], dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    out = {
        "n_trades": int(len(trades)),
        "net_pnl": float(pnl.sum()),
        "winrate": float((pnl > 0).mean()),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(np.median(pnl)),
        "std_pnl": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.nan,
        "skew": float(stats.skew(pnl)) if len(pnl) > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(pnl)) if len(pnl) > 3 else 0.0,
    }
    return out

def mc_survival_by_day(trades: List[Trade], ftmo_cfg: FTMOConfig, n_paths=2000, horizon_days=20, seed=42) -> Dict:
    """
    Bootstrap par jour: on regroupe les trades par jour, on tire des jours aléatoirement,
    on rejoue uniquement les PnL (approx). Utile pour survivabilité, pas pour remplacer la data.
    """
    if not trades:
        return {"survival_rate": np.nan}

    rng = np.random.default_rng(seed)
    df = pd.DataFrame([t.__dict__ for t in trades])
    groups = [g["pnl_usd"].values.astype(float) for _, g in df.groupby("day")]
    if not groups:
        return {"survival_rate": np.nan}

    survives = 0
    for _ in range(n_paths):
        st = FTMOState.init(ftmo_cfg)
        for _d in range(horizon_days):
            st.daily_start_equity = st.equity
            day_pnls = groups[rng.integers(0, len(groups))]
            # replay trades du jour
            for pnl in day_pnls:
                st.equity += float(pnl)
                ftmo_check_fail(ftmo_cfg, st, st.equity, when="MC")
                if st.failed:
                    break
            if st.failed:
                break
        if not st.failed:
            survives += 1

    return {"n_paths": int(n_paths), "horizon_days": int(horizon_days), "survival_rate": float(survives / n_paths)}


# =========================
# MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--data-dir", default="./data_1m")
    ap.add_argument("--out", default="./results")

    # costs stress
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--slip", type=float, default=0.0002)

    # strategy
    ap.add_argument("--sl", type=float, default=0.0020)
    ap.add_argument("--tp", type=float, default=0.0030)

    # FTMO rules (paramétrables)
    ap.add_argument("--balance", type=float, default=10000.0)
    ap.add_argument("--daily-loss", type=float, default=0.05)
    ap.add_argument("--max-loss", type=float, default=0.10)

    # monte carlo
    ap.add_argument("--mc-paths", type=int, default=2000)
    ap.add_argument("--mc-days", type=int, default=20)

    args = ap.parse_args()

    ensure_dir(args.data_dir)
    ensure_dir(args.out)

    mkt = MarketSpec(symbol=args.symbol, exchange=args.exchange, taker_fee=args.fee, slippage=args.slip)
    strat = StrategySpec(sl_pct=args.sl, tp_pct=args.tp)
    ftmo_cfg = FTMOConfig(starting_balance=args.balance, max_daily_loss_pct=args.daily_loss, max_total_loss_pct=args.max_loss)

    ex = make_exchange(mkt.exchange)

    d0 = date.fromisoformat(args.start)
    d1 = date.fromisoformat(args.end)

    st = FTMOState.init(ftmo_cfg)
    all_trades: List[Trade] = []
    equity_daily = []

    print("\n========== CONFIG ==========")
    print(json.dumps({
        "symbol": mkt.symbol,
        "exchange": mkt.exchange,
        "range": [args.start, args.end],
        "fee": mkt.taker_fee,
        "slippage": mkt.slippage,
        "sl_pct": strat.sl_pct,
        "tp_pct": strat.tp_pct,
        "ftmo_balance": ftmo_cfg.starting_balance,
        "ftmo_max_daily_loss_pct": ftmo_cfg.max_daily_loss_pct,
        "ftmo_max_total_loss_pct": ftmo_cfg.max_total_loss_pct,
        "mode_intrabar_conservative": ftmo_cfg.conservative_intrabar,
    }, indent=2))
    print("============================\n")

    for d in daterange(d0, d1):
        if st.failed:
            print(f"[STOP] FTMO FAIL -> {st.fail_reason}")
            break

        # download or load cache
        day_path = os.path.join(args.data_dir, mkt.symbol.replace("/", "_"), f"{iso(d)}.csv.gz")
        if os.path.exists(day_path):
            df_1m = load_day_csv_gz(day_path)
        else:
            df_1m = fetch_ohlcv_1m_day(ex, mkt.symbol, d)
            if df_1m.empty:
                print(f"[SKIP] {iso(d)} no data")
                continue
            df_1m = df_1m.reset_index().rename(columns={"index": "ts"}).set_index("ts")
            save_day_csv_gz(df_1m, day_path)

        qa = qa_day(df_1m, d)
        print(f"\n=== DAY {iso(d)} | equity_start={st.equity:.2f} | QA={qa} ===")

        trades = backtest_day(d, df_1m, mkt, strat, ftmo_cfg, st, verbose=True)
        all_trades.extend(trades)

        equity_daily.append({"day": iso(d), "equity": st.equity, "failed": st.failed, "fail_reason": st.fail_reason})
        day_pnl = sum(t.pnl_usd for t in trades) if trades else 0.0
        print(f"=== END {iso(d)} | trades={len(trades)} | day_pnl={day_pnl:.2f} | equity_end={st.equity:.2f} | failed={st.failed} ===\n")

    # Save results
    trades_df = pd.DataFrame([t.__dict__ for t in all_trades])
    eq_df = pd.DataFrame(equity_daily)

    trades_path = os.path.join(args.out, "trades.csv")
    equity_path = os.path.join(args.out, "equity.csv")
    metrics_path = os.path.join(args.out, "metrics.json")
    mc_path = os.path.join(args.out, "mc.json")
    plot_path = os.path.join(args.out, "equity.png")

    trades_df.to_csv(trades_path, index=False)
    eq_df.to_csv(equity_path, index=False)

    metrics = compute_metrics(all_trades)
    metrics["final_equity"] = float(st.equity)
    metrics["failed"] = bool(st.failed)
    metrics["fail_reason"] = st.fail_reason
    metrics["days"] = int(len(eq_df))
    metrics["config"] = {
        "symbol": mkt.symbol,
        "exchange": mkt.exchange,
        "fee": mkt.taker_fee,
        "slippage": mkt.slippage,
        "sl_pct": strat.sl_pct,
        "tp_pct": strat.tp_pct,
        "ftmo_balance": ftmo_cfg.starting_balance,
        "ftmo_daily_loss_pct": ftmo_cfg.max_daily_loss_pct,
        "ftmo_max_loss_pct": ftmo_cfg.max_total_loss_pct,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    mc = mc_survival_by_day(all_trades, ftmo_cfg, n_paths=args.mc_paths, horizon_days=args.mc_days)
    with open(mc_path, "w") as f:
        json.dump(mc, f, indent=2)

    # Plot equity
    if not eq_df.empty:
        plt.figure()
        plt.plot(pd.to_datetime(eq_df["day"]), eq_df["equity"])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)

    # PRINT FINAL (copiable)
    print("\n================ FINAL METRICS (copiable) ================\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("\n================ MONTE CARLO (copiable) ================\n")
    print(json.dumps(mc, indent=2, sort_keys=True))
    print("\n================ FILES ================\n")
    print(trades_path)
    print(equity_path)
    print(metrics_path)
    print(mc_path)
    print(plot_path)


if __name__ == "__main__":
    main()
