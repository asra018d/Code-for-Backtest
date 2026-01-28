# st with only one trades
# pl change if position is None and daily_trade_count[d] < MAX_TRADES_PER_DAY and t == ENTRY_TIME:
# to if position is None and daily_trade_count[d] < MAX_TRADES_PER_DAY and t >= ENTRY_TIME and t < EXIT_TIME:

import pandas as pd
import pandas_ta as pta
from kiteconnect import KiteConnect
from datetime import datetime, timedelta, time
from collections import defaultdict
import numpy as np

# ================= USER CONFIG =================
INDEX        = "NIFTY"
TIMEFRAME    = 3
BACK_PERIOD  = 10
ENTRY_TIME   = time(11, 18)
EXIT_TIME    = time(15, 15)
LOTS         = 1

EMA_FAST     = 9
EMA_SLOW     = 21
SUP_LEN      = 10
SUP_MUL      = 3

SL_POINTS    = 30
TGT_POINTS   = 60

MAX_TRADES_PER_DAY = 2
LOT_SIZE     = 65
QTY          = LOTS * LOT_SIZE
START_CAPITAL = 100000

# ================= USER CONFIG =================
comparative     = 1   # 1=Comparative Summary (all strategies), 0=Single strategy
result_type     = 2   # 1=Summary, 2=Daily, 3=Trades, 4=Daily+Trades, 5=Monthly, 6=Monthly+Trades
INDICATOR_TYPE  = 4   # 1=ST, 2=EMA CROSSOVER, 3=EMA+Supertrend, 4=ST+PDHL, 5=PDHL, 6=920

END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=BACK_PERIOD)
USE_ZERODHA = True
CSV_FILE = "ohlc.csv"
# ================= ZERODHA LOGIN =================
api_key = open("zerodha_api_key.txt").read().strip()
access_token = open("zerodha_access_token.txt").read().strip()
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)
print("✅ Zerodha Login Success")
INSTRUMENT_TOKEN = 256265

INDICATOR_MAP = {1: "SUPER TREND", 2: "EMA 9 / 21 ", 3: "EMA + SUPER TREND", 4: "SUPER TREND + PDHL", 5: "PDHL", 6: "920"}
indicator_name = INDICATOR_MAP.get(INDICATOR_TYPE, "UNKNOWN STRATEGY")

# ================= FETCH OHLC =================
def fetch_ohlc():
    if not USE_ZERODHA:
        df = pd.read_csv(CSV_FILE, parse_dates=["datetime"])
        return df.sort_values("datetime").reset_index(drop=True)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=BACK_PERIOD)

    interval_map = {
        1: "minute", 3: "3minute", 5: "5minute",
        10: "10minute", 15: "15minute",
        30: "30minute", 60: "60minute"
    }

    all_data = []
    while start_date < end_date:
        chunk_end = min(start_date + timedelta(days=100), end_date)
        print(f"📥 Fetching {start_date.date()} → {chunk_end.date()}")
        data = kite.historical_data(INSTRUMENT_TOKEN, start_date, chunk_end, interval_map[TIMEFRAME])
        if data:
            all_data.extend(data)
        start_date = chunk_end + timedelta(days=1)

    df = pd.DataFrame(all_data)
    df.rename(columns={"date": "datetime"}, inplace=True)
    return df.sort_values("datetime").drop_duplicates().reset_index(drop=True)

# ================= BACKTEST FUNCTIONS =================
def run_st_backtest(df):
    st = pta.supertrend(df["high"], df["low"], df["close"], SUP_LEN, SUP_MUL)
    df["ST"] = st.iloc[:, 0]

    trades, position = [], None
    daily_count = defaultdict(int)

    for i, row in df.iterrows():
        ts, t, d = row["datetime"], row["datetime"].time(), row["datetime"].date()
        c, h, l, stv = row["close"], row["high"], row["low"], row["ST"]

        if position is None and daily_count[d] < MAX_TRADES_PER_DAY and ENTRY_TIME <= t < EXIT_TIME:
            strike = round(c / 50) * 50
            if c > stv:
                position = {"entry_ts": ts, "entry": c, "trade_type": "BUY_CE",
                            "sl": c - SL_POINTS, "tgt": c + TGT_POINTS,
                            "symbol": f"ATM_CE_{strike}"}
                daily_count[d] += 1
            elif c < stv:
                position = {"entry_ts": ts, "entry": c, "trade_type": "BUY_PE",
                            "sl": c + SL_POINTS, "tgt": c - TGT_POINTS,
                            "symbol": f"ATM_PE_{strike}"}
                daily_count[d] += 1

        if position:
            exit_price, reason = None, None
            if position["trade_type"] == "BUY_CE":
                if l <= position["sl"]: exit_price, reason = position["sl"], "SL"
                elif h >= position["tgt"]: exit_price, reason = position["tgt"], "TARGET"
                elif c < stv: exit_price, reason = c, "REVERSAL"
            else:
                if h >= position["sl"]: exit_price, reason = position["sl"], "SL"
                elif l <= position["tgt"]: exit_price, reason = position["tgt"], "TARGET"
                elif c > stv: exit_price, reason = c, "REVERSAL"

            if t >= EXIT_TIME and exit_price is None: exit_price, reason = c, "TIME_EXIT"
            if exit_price is not None:
                book_trade(trades, position, exit_price, reason, ts)
                position = None

    return pd.DataFrame(trades)

def run_ema_backtest(df):
    df["EMA9"] = pta.ema(df["close"], EMA_FAST)
    df["EMA21"] = pta.ema(df["close"], EMA_SLOW)

    trades, position = [], None
    daily_count = defaultdict(int)

    for i in range(1, len(df)):
        ts, t, d = df["datetime"].iloc[i], df["datetime"].iloc[i].time(), df["datetime"].iloc[i].date()
        c = df["close"].iloc[i]
        e9, e21 = df["EMA9"].iloc[i], df["EMA21"].iloc[i]
        e9_prev, e21_prev = df["EMA9"].iloc[i-1], df["EMA21"].iloc[i-1]

        if pd.isna(e9) or pd.isna(e21): continue
        if position is None and daily_count[d] < MAX_TRADES_PER_DAY and ENTRY_TIME <= t < EXIT_TIME:
            strike = round(c / 50) * 50
            if e9_prev <= e21_prev and e9 > e21:
                position = {"entry_ts": ts, "entry": c, "trade_type": "BUY",
                            "sl": c - SL_POINTS, "tgt": c + TGT_POINTS,
                            "symbol": f"NIFTY_EMA_BUY_{strike}"}
                daily_count[d] += 1
            elif e9_prev >= e21_prev and e9 < e21:
                position = {"entry_ts": ts, "entry": c, "trade_type": "SELL",
                            "sl": c + SL_POINTS, "tgt": c - TGT_POINTS,
                            "symbol": f"NIFTY_EMA_SELL_{strike}"}
                daily_count[d] += 1

        if position:
            exit_price, reason = None, None
            if position["trade_type"] == "BUY":
                if c <= position["sl"]: exit_price, reason = position["sl"], "SL"
                elif c >= position["tgt"]: exit_price, reason = position["tgt"], "TARGET"
                elif e9_prev >= e21_prev and e9 < e21: exit_price, reason = c, "EMA_CROSS_REV"
            else:
                if c >= position["sl"]: exit_price, reason = position["sl"], "SL"
                elif c <= position["tgt"]: exit_price, reason = position["tgt"], "TARGET"
                elif e9_prev <= e21_prev and e9 > e21: exit_price, reason = c, "EMA_CROSS_REV"

            if t >= EXIT_TIME and exit_price is None: exit_price, reason = c, "TIME_EXIT"
            if exit_price is not None:
                book_trade(trades, position, exit_price, reason, ts)
                position = None

    return pd.DataFrame(trades)

def run_st_ema_backtest(df):
    df["EMA9"] = pta.ema(df["close"], EMA_FAST)
    df["EMA21"] = pta.ema(df["close"], EMA_SLOW)
    df["ST"] = pta.supertrend(df["high"], df["low"], df["close"], SUP_LEN, SUP_MUL).iloc[:, 0]

    trades, position = [], None
    daily_count = defaultdict(int)

    for i, row in df.iterrows():
        ts, t, d = row["datetime"], row["datetime"].time(), row["datetime"].date()
        c, e9, e21, stv = row["close"], row["EMA9"], row["EMA21"], row["ST"]
        if pd.isna(e9) or pd.isna(e21): continue

        if position is None and daily_count[d] < MAX_TRADES_PER_DAY and ENTRY_TIME <= t < EXIT_TIME:
            strike = round(c / 50) * 50
            if e9 > e21 and c > stv:
                position = {"entry_ts": ts, "entry": c, "trade_type": "BUY_CE",
                            "sl": c - SL_POINTS, "tgt": c + TGT_POINTS, "symbol": f"ATM_CE_{strike}"}
                daily_count[d] += 1
            elif e9 < e21 and c < stv:
                position = {"entry_ts": ts, "entry": c, "trade_type": "BUY_PE",
                            "sl": c + SL_POINTS, "tgt": c - TGT_POINTS, "symbol": f"ATM_PE_{strike}"}
                daily_count[d] += 1

        if position:
            exit_price, reason = None, None
            if position["trade_type"] == "BUY_CE":
                if row["low"] <= position["sl"]: exit_price, reason = position["sl"], "SL"
                elif row["high"] >= position["tgt"]: exit_price, reason = position["tgt"], "TARGET"
                elif c < stv: exit_price, reason = c, "REVERSAL"
            else:
                if row["high"] >= position["sl"]: exit_price, reason = position["sl"], "SL"
                elif row["low"] <= position["tgt"]: exit_price, reason = position["tgt"], "TARGET"
                elif c > stv: exit_price, reason = c, "REVERSAL"

            if t >= EXIT_TIME and exit_price is None: exit_price, reason = c, "TIME_EXIT"
            if exit_price is not None:
                book_trade(trades, position, exit_price, reason, ts)
                position = None

    return pd.DataFrame(trades)

def run_st_daily_breakout_backtest(df):
    trades, position = [], None
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date

    # ---------- PREVIOUS DAY HIGH / LOW ----------
    daily_levels = df.groupby("date").agg(day_high=("high", "max"),
                                          day_low=("low", "min")).shift(1)
    df = df.merge(daily_levels, left_on="date", right_index=True, how="left")

    # ---------- SUPERTREND ----------
    st = pta.supertrend(high=df["high"], low=df["low"], close=df["close"],
                        length=SUP_LEN, multiplier=SUP_MUL)

    # 🔒 Dynamic column detection
    st_col = [c for c in st.columns if c.startswith("SUPERT_") and not c.startswith("SUPERTd")][0]
    trend_col = [c for c in st.columns if c.startswith("SUPERTd_")][0]

    df["supertrend"] = st[st_col]
    df["trend"] = st[trend_col]  # 1 = bullish, -1 = bearish

    daily_count = defaultdict(int)

    # ---------- BACKTEST LOOP ----------
    for i, row in df.iterrows():
        ts, t, d = row["datetime"], row["datetime"].time(), row["date"]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        pdh, pdl = row["day_high"], row["day_low"]
        st_val, trend = row["supertrend"], row["trend"]

        if pd.isna(pdh) or pd.isna(pdl) or pd.isna(st_val): continue
        strike = round(c / 50) * 50

        # ---------------- ENTRY ----------------
        if position is None and ENTRY_TIME <= t < EXIT_TIME and daily_count[d] < MAX_TRADES_PER_DAY:
            # BUY CE
            if row["close"] > pdh and df["close"].iloc[i-1] <= pdh and c > st_val and trend == 1:
                position = {"entry_ts": ts, "entry": o, "trade_type": "BUY_CE",
                            "sl": o - SL_POINTS, "tgt": o + TGT_POINTS,
                            "symbol": f"ATM_CE_{strike}"}
                daily_count[d] += 1
            # BUY PE
            elif row["close"] < pdl and df["close"].iloc[i-1] >= pdl and c < st_val and trend == -1:
                position = {"entry_ts": ts, "entry": o, "trade_type": "BUY_PE",
                            "sl": o + SL_POINTS, "tgt": o - TGT_POINTS,
                            "symbol": f"ATM_PE_{strike}"}
                daily_count[d] += 1

        # ---------------- EXIT / REVERSAL ----------------
        if position:
            exit_price, reason = None, None
            reverse_type = None

            if position["trade_type"] == "BUY_CE":
                if l <= position["sl"]: exit_price, reason = position["sl"], "SL"
                elif h >= position["tgt"]: exit_price, reason = position["tgt"], "TARGET"
                elif c < pdl and c < st_val and trend == -1:
                    exit_price, reason, reverse_type = c, "REVERSAL", "BUY_PE"

            elif position["trade_type"] == "BUY_PE":
                if h >= position["sl"]: exit_price, reason = position["sl"], "SL"
                elif l <= position["tgt"]: exit_price, reason = position["tgt"], "TARGET"
                elif c > pdh and c > st_val and trend == 1:
                    exit_price, reason, reverse_type = c, "REVERSAL", "BUY_CE"

            if t >= EXIT_TIME and exit_price is None:
                exit_price, reason = c, "TIME_EXIT"

            if exit_price is not None:
                book_trade(trades, position, exit_price, reason, ts)
                position = None

                # ---------------- REVERSE ENTRY ----------------
                if reverse_type and ENTRY_TIME <= t < EXIT_TIME:
                    position = {
                        "entry_ts": ts,
                        "entry": c,
                        "trade_type": reverse_type,
                        "sl": c - SL_POINTS if reverse_type.endswith("CE") else c + SL_POINTS,
                        "tgt": c + TGT_POINTS if reverse_type.endswith("CE") else c - TGT_POINTS,
                        "symbol": f"ATM_{reverse_type[-2:]}_{strike}"
                    }

    return pd.DataFrame(trades)

def run_daily_breakout_backtest(df):
    trades, position = [], None
    df = df.copy()
    df["date"] = df["datetime"].dt.date

    # Previous day high/low
    daily_levels = df.groupby("date").agg(day_high=("high", "max"),
                                          day_low=("low", "min")).shift(1)
    df = df.merge(daily_levels, left_on="date", right_index=True, how="left")
    daily_count = defaultdict(int)

    for i, row in df.iterrows():
        ts, t, d = row["datetime"], row["datetime"].time(), row["date"]
        c, o, h, l = row["close"], row["open"], row["high"], row["low"]
        pdh, pdl = row["day_high"], row["day_low"]

        if pd.isna(pdh) or pd.isna(pdl): continue
        strike = round(c / 50) * 50

        # ---------------- ENTRY ----------------
        if position is None and ENTRY_TIME <= t < EXIT_TIME and daily_count[d] < MAX_TRADES_PER_DAY:
            if row["close"] > pdh and df["close"].iloc[i-1] <= pdh:  # LONG
                position = {"entry_ts": ts, "entry": o, "trade_type": "BUY_CE",
                            "sl": o - SL_POINTS, "tgt": o + TGT_POINTS,
                            "symbol": f"ATM_CE_{strike}"}
                daily_count[d] += 1
            elif row["close"] < pdl and df["close"].iloc[i-1] >= pdl:  # SHORT
                position = {"entry_ts": ts, "entry": o, "trade_type": "BUY_PE",
                            "sl": o + SL_POINTS, "tgt": o - TGT_POINTS,
                            "symbol": f"ATM_PE_{strike}"}
                daily_count[d] += 1

        # ---------------- EXIT / REVERSAL ----------------
        if position:
            exit_price, reason = None, None
            reverse_type = None

            if position["trade_type"] == "BUY_CE":
                if l <= position["sl"]: exit_price, reason = position["sl"], "SL"
                elif h >= position["tgt"]: exit_price, reason = position["tgt"], "TARGET"
                elif c < pdl: exit_price, reason, reverse_type = c, "REVERSAL", "BUY_PE"
            else:
                if h >= position["sl"]: exit_price, reason = position["sl"], "SL"
                elif l <= position["tgt"]: exit_price, reason = position["tgt"], "TARGET"
                elif c > pdh: exit_price, reason, reverse_type = c, "REVERSAL", "BUY_CE"

            if t >= EXIT_TIME and exit_price is None: exit_price, reason = c, "TIME_EXIT"

            if exit_price is not None:
                book_trade(trades, position, exit_price, reason, ts)
                position = None

                # Reverse entry
                if reverse_type and ENTRY_TIME <= t < EXIT_TIME:
                    position = {
                        "entry_ts": ts,
                        "entry": c,
                        "trade_type": reverse_type,
                        "sl": c - SL_POINTS if reverse_type.endswith("CE") else c + SL_POINTS,
                        "tgt": c + TGT_POINTS if reverse_type.endswith("CE") else c - TGT_POINTS,
                        "symbol": f"ATM_{reverse_type[-2:]}_{strike}"
                    }

    return pd.DataFrame(trades)

def run_breakout_920_backtest(df):
    trades, position = [], None
    df = df.copy()
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time
    daily_count = defaultdict(int)

    RANGE_START, RANGE_END = time(9, 18), time(9, 21)
    daily_range = {}

    # Build 9:18-9:21 range
    for d, day_df in df.groupby("date"):
        range_df = day_df[(day_df["time"] >= RANGE_START) & (day_df["time"] < RANGE_END)]
        if not range_df.empty:
            daily_range[d] = {"high": range_df["high"].max(), "low": range_df["low"].min()}

    for i, row in df.iterrows():
        ts, t, d = row["datetime"], row["time"], row["date"]
        if d not in daily_range: continue
        c, h, l = row["close"], row["high"], row["low"]
        high_range, low_range = daily_range[d]["high"], daily_range[d]["low"]
        strike = round(c / 50) * 50

        # ---------------- ENTRY ----------------
        if position is None and ENTRY_TIME <= t < EXIT_TIME and daily_count[d] < 1:
            if c > high_range:
                position = {"entry_ts": ts, "entry": c, "trade_type": "BUY_CE",
                            "sl": c - SL_POINTS, "tgt": c + TGT_POINTS,
                            "symbol": f"ATM_CE_{strike}"}
                daily_count[d] += 1
            elif c < low_range:
                position = {"entry_ts": ts, "entry": c, "trade_type": "BUY_PE",
                            "sl": c + SL_POINTS, "tgt": c - TGT_POINTS,
                            "symbol": f"ATM_PE_{strike}"}
                daily_count[d] += 1

        # ---------------- EXIT ----------------
        if position:
            exit_price, reason = None, None

            if position["trade_type"] == "BUY_CE":
                if l <= position["sl"]: exit_price, reason = position["sl"], "SL"
                elif h >= position["tgt"]: exit_price, reason = position["tgt"], "TARGET"
                elif c < low_range: exit_price, reason = c, "REVERSAL"
            else:
                if h >= position["sl"]: exit_price, reason = position["sl"], "SL"
                elif l <= position["tgt"]: exit_price, reason = position["tgt"], "TARGET"
                elif c > high_range: exit_price, reason = c, "REVERSAL"

            if t >= EXIT_TIME and exit_price is None: exit_price, reason = c, "TIME_EXIT"

            if exit_price is not None:
                book_trade(trades, position, exit_price, reason, ts)
                position = None

    return pd.DataFrame(trades)

# ================= HELPER =================
def book_trade(trades, position, exit_price, reason, ts):
    pnl = (exit_price - position["entry"]) * QTY
    if position["trade_type"].endswith("PE") or position["trade_type"] == "SELL":
        pnl = -pnl

    trades.append({
        "date": ts.date(),
        "entry_ts": position["entry_ts"],
        "exit_ts": ts,
        "trade_type": position["trade_type"],
        "entry": round(position["entry"], 2),
        "sl": round(position["sl"], 2),
        "tgt": round(position["tgt"], 2),
        "exit": round(exit_price, 2),
        "pnl": round(pnl, 0),
        "symbol": position["symbol"],
        "exit_reason": reason
    })

# ==================== PERFORMANCE SUMMARY ====================
def performance_summary(trades, INDEX, indicator_name, result_type):
    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print("❌ No trades generated")
        return None

    # ================= REQUIRED COLUMNS =================
    required_cols = ["entry_ts", "exit_ts", "pnl"]
    for c in required_cols:
        if c not in trades_df.columns:
            raise ValueError(f"Missing column: {c}")

    # ================= PREP =================
    trades_df["entry_ts"] = pd.to_datetime(trades_df["entry_ts"], errors="coerce").dt.tz_localize(None)
    trades_df["exit_ts"]  = pd.to_datetime(trades_df["exit_ts"], errors="coerce").dt.tz_localize(None)

    trades_df = trades_df.sort_values("entry_ts").reset_index(drop=True)

    trades_df["date"]  = trades_df["entry_ts"].dt.date
    trades_df["month"] = trades_df["entry_ts"].dt.to_period("M")
    trades_df["cum_pnl"] = trades_df["pnl"].cumsum()

    daily_pnl = trades_df.groupby("date")["pnl"].sum()

    # ================= CORE STATS =================
    total_trades = len(trades_df)
    total_pnl = trades_df["pnl"].sum()

    wins   = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]

    win_rate = (len(wins) / total_trades) * 100 if total_trades else 0
    avg_win  = wins["pnl"].mean() if not wins.empty else 0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0

    gross_profit = wins["pnl"].sum() if not wins.empty else 0
    gross_loss   = abs(losses["pnl"].sum()) if not losses.empty else 0
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

    capital = calculate_capital_metrics(trades_df, START_CAPITAL)

    max_dd       = capital["max_dd"]
    max_dd_pct   = capital["max_dd_pct"]
    cagr         = capital["cagr"]
    exp          = capital["expectancy"]
    calmar       = cagr / max_dd_pct if max_dd_pct > 0 else 0



    # ================= EXTREMES =================
    max_profit_val = daily_pnl.max()
    max_profit_day = daily_pnl.idxmax()
    max_loss_val   = daily_pnl.min()
    max_loss_day   = daily_pnl.idxmin()

    # ================= EXIT COUNTS =================
    exit_counts = trades_df.get("exit_reason", pd.Series()).value_counts().to_dict()
    trading_days = trades_df["date"].nunique()

    # ================= SUMMARY PRINT =================
    print(f"\n📊 {indicator_name} BACKTEST SUMMARY FOR {INDEX}")
    print("═" * 88)
    print(f"📅 Trading Days                  : {trading_days}")
    print(f"📈 Trades                        : {total_trades}")
    print(f"💰 Total P&L                     : ₹{total_pnl:,.0f}")
    print(f"🏆 Win Rate                      : {win_rate:.2f}%")
    print(f"📊 Avg Win                       : ₹{avg_win:,.0f}")
    print(f"📉 Avg Loss                      : ₹{avg_loss:,.0f}")
    print(f"📊 Profit Factor                 : {profit_factor:.2f}")
    print(f"⚠️ Max Drawdown                  : ₹-{max_dd:,.0f}")
    print(f"📏 Max DD %                      : {max_dd_pct:.2f}%")
    print(f"🚀 CAGR                          : {cagr:.2f}%")
    print(f"📐 Calmar Ratio                  : {calmar:.2f}")
    print(f"🎯 Expectancy                    : {exp:.2f}")
    print(f"🚀 Max Profit Day                : ₹{max_profit_val:,.0f} → {max_profit_day}")
    print(f"🔻 Max Loss Day                  : ₹{max_loss_val:,.0f} → {max_loss_day}")

    print("─" * 88)
    print(f"🎯 Target Hits                   : {exit_counts.get('TARGET',0)}")
    print(f"🛑 Stoploss Hits                 : {exit_counts.get('SL',0)}")
    print(f"🔄 Reversals                     : {exit_counts.get('REVERSAL',0)}")
    print(f"🌙 EOD Closes                    : {exit_counts.get('TIME_EXIT',0)}")

    # ================= DAILY =================
    if result_type in [2, 4]:
        print("\n📆 DAILY PERFORMANCE")
        print("Date       | Trades |   P&L | Cum")
        print("-" * 36)
        running = 0
        for d, g in trades_df.groupby("date"):
            pnl = g["pnl"].sum()
            running += pnl
            print(f"{d} | {len(g):>6} | {pnl:>7.0f} | {running:>7.0f}")

    # ================= TRADES =================
    if result_type in [3, 4, 6]:
        print("\n📆 ALL TRADES")
        print("-" * 120)
        for i, t in trades_df.iterrows():
            print(
                f"{i+1:03d} | {t.entry_ts} | {t.exit_ts} | "
                f"P&L: {t.pnl:>7.0f} | Cum: {t.cum_pnl:>7.0f}"
            )

    # ================= MONTHLY =================
    if result_type in [5, 6]:
        print("\n📆 MONTHLY PERFORMANCE")
        print("Month | Days | Trades |   P&L")
        print("-" * 40)
        for m, g in trades_df.groupby("month"):
            print(f"{m} | {g['date'].nunique():>4} | {len(g):>6} | {g['pnl'].sum():>7.0f}")

    # ================= RETURN FOR COMPARATIVE =================
    return {
        "name": indicator_name,
        "trading_days": trading_days,
        "trades": total_trades,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "cagr": cagr,
        "expectancy": exp,
        "calmar": calmar,
        "exit_counts": exit_counts,
        "equity_df": trades_df
    }

def print_comparative_summary(INDEX, results):
    """
    Print side-by-side comparison of all strategies.
    results = list of flattened stats dicts
    """

    print(f"\n📊 STRATEGY COMPARISON FOR {INDEX} FROM {START_DATE.date()} TO {END_DATE.date()}")
    print("═" * 120)

    # ================= HEADER =================
    print(f"{'Metric':<32} | " + " | ".join(f"{r['name'][:10]:>10}" for r in results))
    print("-" * 120)

    rows = [
        ("📅 Trading Days", "trading_days"),
        ("📈 Total Trades", "trades"),
        ("💰 Total P&L", "total_pnl", lambda x: f"{x:,.0f}"),
        ("🏆 Win Rate %", "win_rate", lambda x: f"{x:.2f}"),
        ("📊 Avg Win", "avg_win", lambda x: f"{x:,.0f}"),
        ("📉 Avg Loss", "avg_loss", lambda x: f"{x:,.0f}"),
        ("⚠️ Max DD", "max_dd", lambda x: f"-{x:,.0f}"),
        ("📉 Max DD %", "max_dd_pct", lambda x: f"{x:.2f}%"),
        ("🚀 CAGR %", "cagr", lambda x: f"{x:.2f}%"),
        ("📏 Calmar", "calmar", lambda x: f"{x:.2f}"),
        ("🎯 Expectancy", "expectancy", lambda x: f"{x:.2f}"),
    ]

    for row in rows:
        label, key, *fmt = row
        fmt = fmt[0] if fmt else None

        values = []
        for r in results:
            val = r.get(key, 0)
            values.append(fmt(val) if fmt else val)

        print(f"{label:<32} | " + " | ".join(f"{str(v):>10}" for v in values))

    # ================= EXIT BREAKDOWN =================
    print("\n📤 EXIT BREAKDOWN")
    print("═" * 120)

    exit_rows = [
        ("🎯 Target Hits", "TARGET"),
        ("🛑 Stoploss Hits", "SL"),
        ("🔄 Reversals", "REVERSAL"),
        ("🌙 EOD Closes", "TIME_EXIT"),
    ]

    for label, key in exit_rows:
        counts = []
        for r in results:
            exit_counts = r.get("exit_counts", {})
            counts.append(exit_counts.get(key, 0))

        print(f"{label:<32} | " + " | ".join(f"{c:>10}" for c in counts))

def calculate_capital_metrics(trades_df, start_capital):
    """
    Capital based metrics for BOTH:
    - performance_summary
    - comparative summary

    Assumes columns:
    entry_ts, exit_ts, pnl
    """

    # ================= SAFETY =================
    if trades_df is None or trades_df.empty:
        return {
            "max_dd": 0,
            "max_dd_pct": 0,
            "cagr": 0,
            "expectancy": 0,
        }

    df = trades_df.copy()

    # ================= CUMULATIVE EQUITY =================
    df["cum_pnl"] = df["pnl"].cumsum()
    equity = start_capital + df["cum_pnl"]

    # ================= MAX DRAWDOWN =================
    peak = equity.cummax()
    dd = equity - peak

    max_dd = abs(dd.min())
    max_dd_pct = (max_dd / peak.max()) * 100 if peak.max() > 0 else 0

    # ================= CAGR =================
    start_date = df["entry_ts"].iloc[0].date()
    end_date   = df["exit_ts"].iloc[-1].date()
    days = (end_date - start_date).days

    end_capital = equity.iloc[-1]

    if start_capital > 0 and days > 0:
        years = days / 252
        cagr = ((end_capital / start_capital) ** (1 / years) - 1) * 100
    else:
        cagr = 0

    # ================= EXPECTANCY =================
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]

    win_rate = len(wins) / len(df)
    loss_rate = 1 - win_rate

    avg_win = wins["pnl"].mean() if not wins.empty else 0
    avg_loss = abs(losses["pnl"].mean()) if not losses.empty else 0

    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    return {
        "max_dd": round(max_dd, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "cagr": round(cagr, 2),
        "expectancy": round(expectancy, 2),
    }

# ================= PERFORMANCE SUMMARY =================
def weekly_performance(trades_df):
    if trades_df.empty:
        return pd.DataFrame()

    trades_df = trades_df.copy()
    trades_df["week"] = trades_df["entry_ts"].dt.to_period("W")

    weekly = trades_df.groupby("week").agg(
        trading_days=("date", "nunique"),
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
        losses=("pnl", lambda x: (x < 0).sum())
    ).reset_index()

    weekly["win_rate"] = (weekly["wins"] / weekly["trades"]) * 100
    weekly["avg_per_day"] = weekly["pnl"] / weekly["trading_days"]

    return weekly

def weekday_performance(trades_df):
    if trades_df.empty:
        return pd.DataFrame()

    trades_df = trades_df.copy()
    trades_df["weekday"] = trades_df["entry_ts"].dt.day_name()

    weekday = trades_df.groupby("weekday").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
        losses=("pnl", lambda x: (x < 0).sum())
    ).reset_index()

    weekday["win_rate"] = (weekday["wins"] / weekday["trades"]) * 100

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekday["weekday"] = pd.Categorical(weekday["weekday"], order)
    weekday = weekday.sort_values("weekday")

    return weekday

def monthly_performance(trades_df):
    if trades_df.empty:
        return pd.DataFrame()

    trades_df = trades_df.copy()
    trades_df["month"] = trades_df["entry_ts"].dt.to_period("M")

    monthly = trades_df.groupby("month").agg(
        trading_days=("date", "nunique"),
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
        losses=("pnl", lambda x: (x < 0).sum())
    ).reset_index()

    monthly["win_rate"] = (monthly["wins"] / monthly["trades"]) * 100
    monthly["avg_per_day"] = monthly["pnl"] / monthly["trading_days"]

    # Monthly drawdown
    max_dds = []
    for m, grp in trades_df.groupby("month"):
        daily = grp.groupby("date")["pnl"].sum().cumsum()
        dd = daily - daily.cummax()
        max_dds.append(abs(dd.min()))

    monthly["max_dd"] = max_dds

    return monthly

def print_weekly_summary(weekly_df):
    print("\n📆 WEEKLY PERFORMANCE")
    print("Week        | Days | Trades |   P&L | Avg/Day | Win %")
    print("-" * 55)

    for r in weekly_df.itertuples():
        print(
            f"{r.week} | {r.trading_days:>4} | {r.trades:>6} | "
            f"{r.pnl:>6.0f} | {r.avg_per_day:>7.0f} | {r.win_rate:>5.1f}%"
        )

def print_weekday_summary(weekday_df):
    print("\n📅 WEEKDAY PERFORMANCE")
    print("Day        | Trades |   P&L | Win %")
    print("-" * 40)

    for r in weekday_df.itertuples():
        print(
            f"{r.weekday:<10} | {r.trades:>6} | {r.pnl:>6.0f} | {r.win_rate:>5.1f}%"
        )

def print_monthly_summary(monthly_df):
    print("\n🗓 MONTHLY PERFORMANCE")
    print("Month     | Days | Trades |   P&L | Avg/Day | Win % | Max DD")
    print("-" * 70)

    for r in monthly_df.itertuples():
        print(
            f"{r.month} | {r.trading_days:>4} | {r.trades:>6} | "
            f"{r.pnl:>6.0f} | {r.avg_per_day:>7.0f} | "
            f"{r.win_rate:>5.1f}% | {r.max_dd:>6.0f}"
        )

# ================= PLOT EQUITY CURVE =================
def plot_equity_curve(equity_map, best_strategy=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    for name, df in equity_map.items():
        if df is None or df.empty:
            continue

        # ✅ SAFETY: compute cum_pnl if missing
        if "cum_pnl" not in df.columns:
            df = df.copy()
            df["cum_pnl"] = df["pnl"].cumsum()

        if name == best_strategy:
            plt.plot(df["cum_pnl"], linewidth=3, label=f"{name} 👑")
        else:
            plt.plot(df["cum_pnl"], alpha=0.6, label=name)

    plt.title("Equity Curve Comparison")
    plt.xlabel("Trades")
    plt.ylabel("Cumulative P&L")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ================= PERFORMANCE SUMMARY =================
def calculate_costs(buy_price, sell_price, qty):

    turnover = (buy_price + sell_price) * qty

    brokerage = BROKERAGE_PER_ORDER * 2
    stt = sell_price * qty * STT
    exchange = turnover * EXCHANGE
    sebi = turnover * SEBI
    stamp = buy_price * qty * STAMP

    gst = GST * (brokerage + exchange)

    total_cost = brokerage + stt + exchange + sebi + stamp + gst
    return round(total_cost, 2)

# ================= BEST STRATEGY =================
def auto_select_best_strategy(results):
    """
    Dynamically pick best strategy based on capital-based metrics.
    Priority:
    1. Highest Calmar Ratio
    2. Highest CAGR
    3. Lowest Max Drawdown
    """

    if not results:
        print("❌ No results to evaluate")
        return None

    df = pd.DataFrame(results)

    required_cols = ["name", "calmar", "cagr", "max_dd"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Missing column: {col}")
            return None

    # 🔥 Normalize scores
    df["score"] = (
            df["calmar"].fillna(0) * 0.5 +
            df["cagr"].fillna(0) * 0.4 -
            df["max_dd"].abs().fillna(0) * 0.1
    )

    best = df.sort_values("score", ascending=False).iloc[0]

    print("\n🏆 BEST STRATEGY SELECTED (AUTO)")
    print("════════════════════════════════")
    print(f"📌 Strategy     : {best['name']}")
    print(f"🚀 CAGR %       : {best['cagr']:.2f}")
    print(f"📉 Max DD       : {best['max_dd']:,.0f}")
    print(f"📏 Calmar Ratio : {best['calmar']:.2f}")
    print(f"🎯 Score        : {best['score']:.2f}")

    return best["name"]

def run_strategy(name, df, backtest_fn):
    trades = backtest_fn(df)
    trades_df = pd.DataFrame(trades)

    stats = performance_summary(
        trades,
        INDEX,
        indicator_name=name,
        result_type=0   # silent mode
    )

    stats["trades_df"] = trades_df

    return {
        "name": name,
        "trades_df": trades_df,
        "stats": stats
    }

def stats_by_name(results):
    return {r["name"]: r["stats"] for r in results}

# ================= RUN ALL BACKTEST =================
def run_all_backtests():
    """
    Run backtests for either a single strategy or comparative of all strategies.
    Collect stats correctly for comparative table.
    """
    global comparative, result_type, INDICATOR_TYPE

    print("\n🚀 RUNNING BACKTEST(S)")
    print("════════════════════════════════")

    df = fetch_ohlc()
    if df is None or df.empty:
        print("❌ No data fetched")
        return

    # ================= STRATEGY MAP =================
    strategy_map = {
        1: ("SUPER TREND", run_st_backtest),
        2: ("EMA", run_ema_backtest),
        3: ("ST+EMA", run_st_ema_backtest),
        4: ("ST+PDHL", run_st_daily_breakout_backtest),
        5: ("PDHL", run_daily_breakout_backtest),
        6: ("920", run_breakout_920_backtest),
    }

    results = []
    equity_map = {}

    # =================================================
    # 🔁 COMPARATIVE MODE
    # =================================================
    if comparative == 1:
        for ind_type, (name, fn) in strategy_map.items():
            try:
                print(f"\n📊 {name} BACKTEST SUMMARY FOR {INDEX} FROM {START_DATE.date()} TO {END_DATE.date()}")
                print("═" * 88)

                trades = fn(df)

                stats = performance_summary(
                    trades=trades,
                    INDEX=INDEX,
                    indicator_name=name,
                    result_type=result_type
                )

                # ❗ VERY IMPORTANT: flatten stats
                stats["name"] = name
                results.append(stats)

            except Exception as e:
                print(f"❌ ERROR IN {name}: {e}")

        # ================= COMPARATIVE OUTPUT =================
        if results:
            print_comparative_summary(INDEX, results)
            best = auto_select_best_strategy(results)

            if equity_map:
                plot_equity_curve(equity_map)

    # =================================================
    # 🎯 SINGLE STRATEGY MODE
    # =================================================
    else:
        if INDICATOR_TYPE not in strategy_map:
            print(f"❌ Invalid INDICATOR_TYPE: {INDICATOR_TYPE}")
            return

        name, fn = strategy_map[INDICATOR_TYPE]

        try:
            print(f"\n📊 {name} BACKTEST SUMMARY FOR {INDEX} FROM {START_DATE.date()} TO {END_DATE.date()}")
            print("═" * 88)

            trades = fn(df)

            performance_summary(
                trades=trades,
                INDEX=INDEX,
                indicator_name=name,
                result_type=result_type
            )

        except Exception as e:
            print(f"❌ ERROR OCCURRED DURING BACKTEST: {e}")

    print("\n✅ ALL BACKTESTS COMPLETED")

# ================= MAIN =================
if __name__ == "__main__":
    try:
        print("\n🚀 STARTING ALL BACKTESTS")
        run_all_backtests()
        print("✅ BACKTESTS COMPLETED SUCCESSFULLY")
    except Exception as e:
        print(f"❌ ERROR OCCURRED DURING BACKTEST: {e}")