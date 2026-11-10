import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, time
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Intraday KO Scanner", page_icon="📈", layout="wide")
st.title("📈 15-Minuten Intraday KO Scanner")
st.caption("Technische Entscheidungshilfe. KO-Produkte, Spreads, Emittentenpreis und Knock-out-Level immer manuell beim Broker prüfen.")

WATCHLIST = {
    "Microsoft": "MSFT",
    "Apple": "AAPL",
    "Nvidia": "NVDA",
    "Amazon": "AMZN",
    "DAX": "^GDAXI",
    "Nasdaq 100": "^NDX",
    "S&P 500": "^GSPC",
    "Gold": "GC=F",
    "Öl WTI": "CL=F",
    "EUR/USD": "EURUSD=X",
}

MIN_RR = 1.8
INTERVAL = "15m"
PERIOD = "30d"


def as_series(frame: pd.DataFrame, column: str) -> pd.Series:
    series = frame[column]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce")


@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker: str) -> pd.DataFrame:
    frame = yf.download(
        ticker,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
        prepost=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        return frame

    result = pd.DataFrame(index=frame.index)
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        result[column] = as_series(frame, column)
    return result.dropna(subset=["High", "Low", "Close"])


def market_state(ticker: str) -> str:
    if ticker == "^GDAXI":
        now = datetime.now(ZoneInfo("Europe/Berlin"))
        opened = now.weekday() < 5 and time(9, 0) <= now.time() < time(17, 30)
        return "🟢 Offen" if opened else "🔴 Geschlossen"

    if ticker in {"MSFT", "AAPL", "NVDA", "AMZN", "^NDX", "^GSPC"}:
        now = datetime.now(ZoneInfo("America/New_York"))
        opened = now.weekday() < 5 and time(9, 30) <= now.time() < time(16, 0)
        return "🟢 Offen" if opened else "🔴 Geschlossen"

    return "🟡 Handelszeiten prüfen"


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]
    volume = df["Volume"].fillna(0)

    df["EMA9"] = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df["EMA20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["EMA200"] = ta.trend.EMAIndicator(close, window=200).ema_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["ADX"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_HIST"] = macd.macd_diff()

    # VWAP resets for every trading date. For assets without useful volume,
    # EMA20 is used as a transparent fallback.
    typical = (high + low + close) / 3
    session = pd.Series(df.index.date, index=df.index)
    cum_volume = volume.groupby(session).cumsum()
    cum_value = (typical * volume).groupby(session).cumsum()
    df["VWAP"] = (cum_value / cum_volume.replace(0, np.nan)).fillna(df["EMA20"])

    df["VOL_MA20"] = volume.rolling(20).mean()
    df["VOL_RATIO"] = volume / df["VOL_MA20"].replace(0, np.nan)
    df["VOL_RATIO"] = df["VOL_RATIO"].replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=["EMA200", "RSI", "ATR", "ADX", "MACD_HIST"])


def analyse(ticker: str, name: str):
    raw = load_data(ticker)
    if len(raw) < 230:
        return None, raw

    df = add_indicators(raw)
    if len(df) < 3:
        return None, df

    last = df.iloc[-1]
    previous = df.iloc[-2]
    price = float(last["Close"])
    atr = float(last["ATR"])
    atr_pct = atr / price * 100 if price else 0
    volume_available = bool(df["Volume"].tail(30).sum() > 0)
    vol_ok = (not volume_available) or pd.isna(last["VOL_RATIO"]) or last["VOL_RATIO"] >= 0.8

    long_conditions = {
        "Trend": price > last["EMA20"] > last["EMA50"] > last["EMA200"],
        "Momentum": 52 <= last["RSI"] <= 68 and last["MACD_HIST"] > 0,
        "Trendstärke": last["ADX"] >= 20,
        "VWAP": price > last["VWAP"],
        "Volumen": vol_ok,
        "2 Kerzen bestätigt": previous["Close"] > previous["EMA20"] and price > last["EMA20"],
        "Nicht überdehnt": (price - last["EMA20"]) <= 1.25 * atr,
    }
    short_conditions = {
        "Trend": price < last["EMA20"] < last["EMA50"] < last["EMA200"],
        "Momentum": 32 <= last["RSI"] <= 48 and last["MACD_HIST"] < 0,
        "Trendstärke": last["ADX"] >= 20,
        "VWAP": price < last["VWAP"],
        "Volumen": vol_ok,
        "2 Kerzen bestätigt": previous["Close"] < previous["EMA20"] and price < last["EMA20"],
        "Nicht überdehnt": (last["EMA20"] - price) <= 1.25 * atr,
    }

    weights = {
        "Trend": 25,
        "Momentum": 15,
        "Trendstärke": 15,
        "VWAP": 10,
        "Volumen": 10,
        "2 Kerzen bestätigt": 15,
        "Nicht überdehnt": 10,
    }
    long_score = sum(weights[key] for key, value in long_conditions.items() if value)
    short_score = sum(weights[key] for key, value in short_conditions.items() if value)

    if long_score >= 75 and long_score > short_score:
        signal, direction, score = "🟢 LONG", "LONG", long_score
    elif short_score >= 75 and short_score > long_score:
        signal, direction, score = "🔴 SHORT", "SHORT", short_score
    else:
        signal, direction, score = "🟡 WARTEN", "WAIT", max(long_score, short_score)

    # Provisional levels based on the underlying, not the KO product quote.
    if direction == "LONG":
        entry = max(price, float(last["EMA9"]))
        stop = entry - 1.20 * atr
        target1 = entry + MIN_RR * (entry - stop)
        target2 = entry + 2.50 * (entry - stop)
        ko_reference = stop - 0.50 * atr
        checks = long_conditions
    elif direction == "SHORT":
        entry = min(price, float(last["EMA9"]))
        stop = entry + 1.20 * atr
        target1 = entry - MIN_RR * (stop - entry)
        target2 = entry - 2.50 * (stop - entry)
        ko_reference = stop + 0.50 * atr
        checks = short_conditions
    else:
        entry = stop = target1 = target2 = ko_reference = np.nan
        checks = long_conditions if long_score >= short_score else short_conditions

    ko_distance_pct = abs(price - ko_reference) / price * 100 if direction != "WAIT" else np.nan
    risk_label = "🟢" if ko_distance_pct >= 3.0 else "🟡" if ko_distance_pct >= 2.0 else "🔴"

    result = {
        "Asset": name,
        "Ticker": ticker,
        "Kurs": round(price, 4),
        "Signal": signal,
        "Score": int(score),
        "Long": int(long_score),
        "Short": int(short_score),
        "RSI": round(float(last["RSI"]), 1),
        "ADX": round(float(last["ADX"]), 1),
        "ATR %": round(atr_pct, 2),
        "Markt": market_state(ticker),
        "Entry": round(entry, 4) if np.isfinite(entry) else None,
        "Stop": round(stop, 4) if np.isfinite(stop) else None,
        "Ziel 1": round(target1, 4) if np.isfinite(target1) else None,
        "Ziel 2": round(target2, 4) if np.isfinite(target2) else None,
        "KO max./min.": round(ko_reference, 4) if np.isfinite(ko_reference) else None,
        "KO Abstand %": round(ko_distance_pct, 2) if np.isfinite(ko_distance_pct) else None,
        "KO Puffer": risk_label if direction != "WAIT" else "–",
        "Checks": checks,
        "Direction": direction,
        "Timestamp": df.index[-1],
    }
    return result, df



def backtest_signals(df: pd.DataFrame, score_threshold=75, atr_stop=1.2, rr=1.8, max_bars=32, costs_bps=8):
    """Backtest ohne Look-ahead: Signal am Kerzenschluss, Einstieg am nächsten Open.
    Falls Stop und Ziel in derselben Kerze berührt werden, wird konservativ der Stop angenommen.
    Ergebnisse beziehen sich auf den Basiswert, nicht auf die Preisbildung eines KO-Produkts.
    """
    if df is None or len(df) < 230:
        return pd.DataFrame(), pd.DataFrame()

    bt = df.copy().dropna(subset=["EMA200", "RSI", "ATR", "ADX", "MACD_HIST", "VWAP"])
    trades = []
    i = 2
    while i < len(bt) - 1:
        row, prev = bt.iloc[i], bt.iloc[i - 1]
        price, atr = float(row["Close"]), float(row["ATR"])
        volume_available = bool(bt["Volume"].iloc[max(0, i-30):i+1].sum() > 0)
        vol_ratio = row.get("VOL_RATIO", np.nan)
        vol_ok = (not volume_available) or pd.isna(vol_ratio) or vol_ratio >= 0.8

        long_checks = [
            price > row["EMA20"] > row["EMA50"] > row["EMA200"],
            52 <= row["RSI"] <= 68 and row["MACD_HIST"] > 0,
            row["ADX"] >= 20, price > row["VWAP"], vol_ok,
            prev["Close"] > prev["EMA20"] and price > row["EMA20"],
            (price - row["EMA20"]) <= 1.25 * atr,
        ]
        short_checks = [
            price < row["EMA20"] < row["EMA50"] < row["EMA200"],
            32 <= row["RSI"] <= 48 and row["MACD_HIST"] < 0,
            row["ADX"] >= 20, price < row["VWAP"], vol_ok,
            prev["Close"] < prev["EMA20"] and price < row["EMA20"],
            (row["EMA20"] - price) <= 1.25 * atr,
        ]
        weights = [25, 15, 15, 10, 10, 15, 10]
        long_score = sum(w for w, ok in zip(weights, long_checks) if ok)
        short_score = sum(w for w, ok in zip(weights, short_checks) if ok)
        direction = "LONG" if long_score >= score_threshold and long_score > short_score else (
            "SHORT" if short_score >= score_threshold and short_score > long_score else None
        )
        if not direction:
            i += 1
            continue

        entry_i = i + 1
        entry = float(bt.iloc[entry_i]["Open"])
        risk = atr_stop * atr
        stop = entry - risk if direction == "LONG" else entry + risk
        target = entry + rr * risk if direction == "LONG" else entry - rr * risk
        exit_i = min(entry_i + max_bars, len(bt) - 1)
        exit_price = float(bt.iloc[exit_i]["Close"])
        reason = "TIME"

        for j in range(entry_i, exit_i + 1):
            high, low = float(bt.iloc[j]["High"]), float(bt.iloc[j]["Low"])
            stop_hit = low <= stop if direction == "LONG" else high >= stop
            target_hit = high >= target if direction == "LONG" else low <= target
            if stop_hit:  # konservative Reihenfolge bei gleichzeitiger Berührung
                exit_i, exit_price, reason = j, stop, "STOP"
                break
            if target_hit:
                exit_i, exit_price, reason = j, target, "TARGET"
                break

        gross_r = (exit_price - entry) / risk if direction == "LONG" else (entry - exit_price) / risk
        cost_r = (2 * costs_bps / 10000 * entry) / risk
        net_r = gross_r - cost_r
        trades.append({
            "Signalzeit": bt.index[i], "Einstieg": bt.index[entry_i], "Ausstieg": bt.index[exit_i],
            "Richtung": direction, "Score": max(long_score, short_score), "Entry": entry,
            "Stop": stop, "Ziel": target, "Exit": exit_price, "Grund": reason,
            "Brutto R": gross_r, "Netto R": net_r, "Gewinn": net_r > 0,
        })
        i = exit_i + 1

    trades = pd.DataFrame(trades)
    if trades.empty:
        return trades, pd.DataFrame()
    trades["Equity R"] = trades["Netto R"].cumsum()
    equity = trades[["Ausstieg", "Equity R"]].set_index("Ausstieg")
    return trades, equity


def backtest_summary(trades: pd.DataFrame, direction=None):
    sample = trades if direction is None else trades[trades["Richtung"] == direction]
    if sample.empty:
        return {"Trades": 0, "Trefferquote": 0.0, "Netto R": 0.0, "Ø R": 0.0, "Profit Factor": 0.0, "Max Drawdown R": 0.0}
    positive = sample.loc[sample["Netto R"] > 0, "Netto R"].sum()
    negative = abs(sample.loc[sample["Netto R"] < 0, "Netto R"].sum())
    curve = sample["Netto R"].cumsum()
    drawdown = curve - curve.cummax()
    return {
        "Trades": len(sample), "Trefferquote": 100 * sample["Gewinn"].mean(),
        "Netto R": sample["Netto R"].sum(), "Ø R": sample["Netto R"].mean(),
        "Profit Factor": positive / negative if negative > 0 else np.inf,
        "Max Drawdown R": abs(drawdown.min()),
    }

st.sidebar.header("Scanner-Einstellungen")
minimum_score = st.sidebar.slider("Mindestscore", 60, 100, 75, 5)
only_open = st.sidebar.checkbox("Nur offene Märkte", value=False)
st.sidebar.subheader("Backtest")
bt_score = st.sidebar.slider("Backtest Mindestscore", 60, 100, 75, 5)
bt_rr = st.sidebar.number_input("Ziel in R", 1.0, 4.0, 1.8, 0.1)
bt_stop_atr = st.sidebar.number_input("Stop in ATR", 0.5, 3.0, 1.2, 0.1)
bt_max_bars = st.sidebar.number_input("Max. Haltedauer in 15m-Kerzen", 4, 96, 32, 4)
bt_costs = st.sidebar.number_input("Kosten/Slippage je Seite (Basispunkte)", 0, 100, 8, 1)
st.sidebar.info("Entry, Stop, Ziele und KO-Referenz beziehen sich auf den Basiswert. Das konkrete Zertifikat kann wegen Bezugsverhältnis, Spread, Aufgeld und Finanzierung abweichen.")

st.header("🕒 Marktstatus")
status_cols = st.columns(3)
status_cols[0].metric("DAX", market_state("^GDAXI"))
status_cols[1].metric("US-Börsen", market_state("MSFT"))
status_cols[2].metric("Datenintervall", "15 Minuten")

rows, datasets = [], {}
with st.spinner("15-Minuten-Daten und Indikatoren werden geladen …"):
    for asset_name, symbol in WATCHLIST.items():
        try:
            result, calculated = analyse(symbol, asset_name)
            if result:
                rows.append(result)
                datasets[asset_name] = calculated
        except Exception as exc:
            st.warning(f"{asset_name}: {exc}")

if not rows:
    st.error("Keine ausreichenden 15-Minuten-Daten verfügbar.")
    st.stop()

scan = pd.DataFrame(rows)
if only_open:
    scan = scan[scan["Markt"].str.contains("Offen", na=False)]

st.header("🌍 15-Minuten-Ranking")
display_columns = [
    "Asset", "Signal", "Score", "Long", "Short", "Kurs", "RSI", "ADX",
    "ATR %", "KO Abstand %", "KO Puffer", "Markt"
]
st.dataframe(
    scan.sort_values(["Score", "ADX"], ascending=False)[display_columns],
    use_container_width=True,
    hide_index=True,
)

trade_ready = scan[(scan["Score"] >= minimum_score) & (scan["Signal"] != "🟡 WARTEN")]
st.header("🔥 Bestätigte Kandidaten")
if trade_ready.empty:
    st.info("Kein Setup erfüllt aktuell Mindestscore und Zwei-Kerzen-Bestätigung.")
else:
    for _, row in trade_ready.sort_values("Score", ascending=False).iterrows():
        st.success(
            f"{row['Signal']} {row['Asset']} | Score {row['Score']} | "
            f"Entry {row['Entry']} | Stop {row['Stop']} | Ziel 1 {row['Ziel 1']} | "
            f"KO-Referenz {row['KO max./min.']} ({row['KO Abstand %']} %)"
        )


st.header("🧪 Backtest LONG und SHORT")
st.caption("Signal am Schluss einer bestätigten 15-Minuten-Kerze, Einstieg am nächsten Open. Keine überlappenden Positionen. Stop vor Ziel, falls beide innerhalb derselben Kerze berührt werden.")
bt_asset = st.selectbox("Asset für Backtest", scan["Asset"].tolist(), key="bt_asset")
bt_data = datasets[bt_asset]
trades, equity = backtest_signals(
    bt_data, score_threshold=bt_score, atr_stop=bt_stop_atr,
    rr=bt_rr, max_bars=int(bt_max_bars), costs_bps=int(bt_costs)
)

if trades.empty:
    st.info("Für die gewählten Parameter wurden im verfügbaren Datenfenster keine abgeschlossenen Trades gefunden.")
else:
    total = backtest_summary(trades)
    long_stats = backtest_summary(trades, "LONG")
    short_stats = backtest_summary(trades, "SHORT")
    summary_df = pd.DataFrame([total, long_stats, short_stats], index=["Gesamt", "LONG", "SHORT"])
    st.dataframe(summary_df.style.format({
        "Trefferquote": "{:.1f}%", "Netto R": "{:.2f}", "Ø R": "{:.2f}",
        "Profit Factor": "{:.2f}", "Max Drawdown R": "{:.2f}"
    }), use_container_width=True)
    st.line_chart(equity, use_container_width=True)
    st.dataframe(
        trades.sort_values("Einstieg", ascending=False).style.format({
            "Entry": "{:.4f}", "Stop": "{:.4f}", "Ziel": "{:.4f}", "Exit": "{:.4f}",
            "Brutto R": "{:.2f}", "Netto R": "{:.2f}", "Equity R": "{:.2f}"
        }), use_container_width=True, hide_index=True
    )
    st.download_button(
        "Backtest-Trades als CSV herunterladen",
        trades.to_csv(index=False).encode("utf-8"),
        file_name=f"backtest_{WATCHLIST[bt_asset].replace('^','')}_15m.csv",
        mime="text/csv",
    )
st.warning("Der Backtest verwendet Yahoo-Basiswertdaten und modelliert weder KO-Preisbildung noch realen Spread, Aufgeld, Finanzierung, Emittentenrisiko oder Slippage exakt. Ergebnisse sind keine Renditeprognose.")

st.header("🔍 Detailanalyse")
asset = st.selectbox("Asset auswählen", scan["Asset"].tolist())
selected = next(row for row in rows if row["Asset"] == asset)
detail = datasets[asset]

metric_cols = st.columns(6)
metric_cols[0].metric("Signal", selected["Signal"])
metric_cols[1].metric("Score", selected["Score"])
metric_cols[2].metric("Kurs", selected["Kurs"])
metric_cols[3].metric("RSI 14", selected["RSI"])
metric_cols[4].metric("ADX 14", selected["ADX"])
metric_cols[5].metric("ATR %", selected["ATR %"])

if selected["Direction"] != "WAIT":
    level_cols = st.columns(5)
    level_cols[0].metric("Entry Basiswert", selected["Entry"])
    level_cols[1].metric("Stop Basiswert", selected["Stop"])
    level_cols[2].metric("Ziel 1 (RR 1,8)", selected["Ziel 1"])
    level_cols[3].metric("Ziel 2", selected["Ziel 2"])
    level_cols[4].metric("KO-Referenz", selected["KO max./min."])

checks_df = pd.DataFrame(
    [{"Kriterium": key, "Erfüllt": "✅" if value else "❌"} for key, value in selected["Checks"].items()]
)
st.dataframe(checks_df, use_container_width=True, hide_index=True)

chart_data = detail[["Close", "EMA9", "EMA20", "EMA50", "EMA200", "VWAP"]].tail(160)
st.line_chart(chart_data, use_container_width=True)

st.caption(f"Letzte Datenkerze: {selected['Timestamp']} | Signale nur nach abgeschlossener 15-Minuten-Kerze bewerten.")
st.warning("KO-Handel ist hochriskant. Keine Position eröffnen, wenn der Markt geschlossen ist, Daten veraltet sind, der Emittenten-Spread auffällig ist oder der reale Chance-Risiko-Faktor unter 1,8 liegt. Niemals den KO-Level als Stop-Loss verwenden.")
