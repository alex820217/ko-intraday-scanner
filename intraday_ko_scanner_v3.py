import time as time_module
from datetime import datetime, time
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st
import ta
import yfinance as yf
from decision_logic import SignalInput, decide_signal
from market_universe import MARKET_GROUPS, build_universe

st.set_page_config(page_title="KO Market Radar V3", page_icon="📡", layout="wide")
st.title("📡 KO Market Radar V3")
st.caption("Breiter Markt-Scan: 5-Minuten Entry-Timing plus 15-Minuten Trendbestätigung")

MIN_15M_BARS = 210
MIN_5M_BARS = 60


def series(frame, column):
    value = frame[column]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return pd.to_numeric(value, errors="coerce")


@st.cache_data(ttl=300, show_spinner=False)
def download_one(ticker, interval, period):
    data = yf.download(ticker, interval=interval, period=period, auto_adjust=True,
                       prepost=False, progress=False, threads=False)
    if data.empty:
        return data
    out = pd.DataFrame(index=data.index)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        out[col] = series(data, col)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def market_open(ticker):
    if ticker.endswith(".DE") or ticker in {"^GDAXI", "^STOXX50E"}:
        now = datetime.now(ZoneInfo("Europe/Berlin"))
        return now.weekday() < 5 and time(9, 0) <= now.time() < time(17, 30)
    if ticker.startswith("^") or ticker.isalpha():
        now = datetime.now(ZoneInfo("America/New_York"))
        return now.weekday() < 5 and time(9, 30) <= now.time() < time(16, 0)
    return True


def add_indicators(df):
    d = df.copy()
    c, h, l, v = d.Close, d.High, d.Low, d.Volume.fillna(0)
    for n in [9, 20, 50, 200]:
        d[f"EMA{n}"] = ta.trend.EMAIndicator(c, window=n).ema_indicator()
    d["RSI"] = ta.momentum.RSIIndicator(c, window=14).rsi()
    d["ATR"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    d["ADX"] = ta.trend.ADXIndicator(h, l, c, window=14).adx()
    d["MACD_HIST"] = ta.trend.MACD(c).macd_diff()
    typical = (h + l + c) / 3
    dates = pd.Series(d.index.date, index=d.index)
    cumulative_volume = v.groupby(dates).cumsum()
    d["VWAP"] = ((typical * v).groupby(dates).cumsum() / cumulative_volume.replace(0, np.nan)).fillna(d.EMA20)
    d["VOL_RATIO"] = v / v.rolling(20).mean().replace(0, np.nan)
    return d


def score_15m(d):
    r, p = d.iloc[-1], d.iloc[-2]
    price, atr = float(r.Close), float(r.ATR)
    vol_present = d.Volume.tail(30).sum() > 0
    vol_ok = (not vol_present) or pd.isna(r.VOL_RATIO) or r.VOL_RATIO >= 0.8
    lc = {
        "15m Trend": price > r.EMA20 > r.EMA50 > r.EMA200,
        "Momentum": 52 <= r.RSI <= 68 and r.MACD_HIST > 0,
        "ADX": r.ADX >= 20, "VWAP": price > r.VWAP, "Volumen": vol_ok,
        "2-Bar": p.Close > p.EMA20 and price > r.EMA20,
        "Nicht überdehnt": (price-r.EMA20) <= 1.25*atr,
    }
    sc = {
        "15m Trend": price < r.EMA20 < r.EMA50 < r.EMA200,
        "Momentum": 32 <= r.RSI <= 48 and r.MACD_HIST < 0,
        "ADX": r.ADX >= 20, "VWAP": price < r.VWAP, "Volumen": vol_ok,
        "2-Bar": p.Close < p.EMA20 and price < r.EMA20,
        "Nicht überdehnt": (r.EMA20-price) <= 1.25*atr,
    }
    weights = {"15m Trend":25,"Momentum":15,"ADX":15,"VWAP":10,"Volumen":10,"2-Bar":15,"Nicht überdehnt":10}
    return sum(weights[k] for k,v in lc.items() if v), sum(weights[k] for k,v in sc.items() if v), lc, sc


def timing_5m(d, direction):
    r, p = d.iloc[-1], d.iloc[-2]
    if direction == "LONG":
        checks = {
            "5m Kurs > EMA9": r.Close > r.EMA9,
            "5m EMA9 > EMA20": r.EMA9 > r.EMA20,
            "5m MACD positiv": r.MACD_HIST > 0,
            "5m Trigger bestätigt": r.Close > p.High,
        }
    else:
        checks = {
            "5m Kurs < EMA9": r.Close < r.EMA9,
            "5m EMA9 < EMA20": r.EMA9 < r.EMA20,
            "5m MACD negativ": r.MACD_HIST < 0,
            "5m Trigger bestätigt": r.Close < p.Low,
        }
    return checks, sum(checks.values())


def scan_asset(meta, spread_ok, event_risk, require_open):
    d15 = add_indicators(download_one(meta["ticker"], "15m", "30d")).dropna()
    d5 = add_indicators(download_one(meta["ticker"], "5m", "5d")).dropna()
    if len(d15) < MIN_15M_BARS or len(d5) < MIN_5M_BARS:
        raise ValueError(f"zu wenige Daten (15m={len(d15)}, 5m={len(d5)})")

    ls, ss, lc, sc = score_15m(d15)
    preferred = "LONG" if ls >= ss else "SHORT"
    t5, timing_count = timing_5m(d5, preferred)
    # Entry Timing: bis zu 10 zusätzliche Punkte, READY benötigt mindestens 3/4 Timing-Bedingungen.
    adjusted_long = min(100, ls + (timing_count * 2 if preferred == "LONG" else 0))
    adjusted_short = min(100, ss + (timing_count * 2 if preferred == "SHORT" else 0))

    r, p = d15.iloc[-1], d15.iloc[-2]
    price, atr = float(r.Close), float(r.ATR)
    stop = price - 1.2*atr if preferred == "LONG" else price + 1.2*atr
    target1 = price + 1.8*abs(price-stop) if preferred == "LONG" else price - 1.8*abs(price-stop)
    rr = abs(target1-price)/abs(price-stop)
    is_open = market_open(meta["ticker"])
    decision = decide_signal(SignalInput(
        price=price, prev_close=float(p.Close), ema20=float(r.EMA20), ema50=float(r.EMA50),
        ema200=float(r.EMA200), prev_ema20=float(p.EMA20), rsi=float(r.RSI),
        macd_hist=float(r.MACD_HIST), adx=float(r.ADX), vwap=float(r.VWAP), atr=atr,
        long_score=adjusted_long, short_score=adjusted_short, rr=rr,
        market_open=(is_open or not require_open), data_fresh=True, confirmed_bar=True,
        spread_ok=spread_ok, event_risk=event_risk,
        volatility_ok=0.05 <= atr/price*100 <= 5.0,
    ))

    # READY wird auf WATCH zurückgestuft, falls das 5m Timing noch nicht bestätigt ist.
    if decision["signal"] in {"🟢 READY LONG", "🔴 READY SHORT"} and timing_count < 3:
        decision["signal"] = "🟡 WATCH " + decision["direction"]
        decision["quality"] = "WATCH"
        decision["reason"] = "15m Setup gültig, 5m Entry-Timing noch nicht bestätigt"

    direction = decision["direction"]
    if direction == "LONG":
        stop, t1, t2, ko = price-1.2*atr, price+2.16*atr, price+3.36*atr, price-1.7*atr
    elif direction == "SHORT":
        stop, t1, t2, ko = price+1.2*atr, price-2.16*atr, price-3.36*atr, price+1.7*atr
    else:
        stop=t1=t2=ko=np.nan

    checks = lc if direction == "LONG" else sc if direction == "SHORT" else (lc if ls >= ss else sc)
    checks = {**checks, **t5}
    return {
        "Gruppe":meta["group"], "Asset":meta["name"], "Ticker":meta["ticker"],
        "Signal":decision["signal"], "Qualität":decision["quality"], "Score":decision["score"],
        "Long":adjusted_long, "Short":adjusted_short, "5m Timing":f"{timing_count}/4",
        "Kurs":price, "RSI":float(r.RSI), "ADX":float(r.ADX), "ATR %":atr/price*100,
        "Markt":"🟢 Offen" if is_open else "🔴 Geschlossen", "Grund":decision["reason"],
        "Entry":price if direction in {"LONG","SHORT"} else np.nan, "Stop":stop,
        "Ziel 1":t1, "Ziel 2":t2, "CRV":rr if direction in {"LONG","SHORT"} else np.nan,
        "KO Referenz":ko, "KO Abstand %":abs(price-ko)/price*100 if np.isfinite(ko) else np.nan,
        "Checks":checks, "Timestamp 15m":d15.index[-1], "Timestamp 5m":d5.index[-1],
    }, d15, d5


st.sidebar.header("Marktuniversum")
default_groups = ["Deutschland", "US Tech", "Indizes", "Metalle", "Energie", "Forex"]
selected_groups = st.sidebar.multiselect("Marktgruppen", list(MARKET_GROUPS), default=default_groups)
max_assets = st.sidebar.slider("Maximale Assets pro Scan", 10, 60, 54, 2)
require_open = st.sidebar.checkbox("Geschlossene Märkte als NO TRADE", True)
spread_ok = st.sidebar.checkbox("KO-Spread manuell akzeptabel", True)
event_risk = st.sidebar.checkbox("Event-Risiko aktiv", False)
top_n = st.sidebar.slider("Top-Ergebnisse anzeigen", 5, 30, 15)

universe = build_universe(selected_groups)
all_items = list(universe.values())[:max_assets]
# Vorfilter V3.1: geschlossene Cash-Märkte werden VOR jedem yfinance-Aufruf entfernt.
# Futures/FX bleiben aktiv, da deren Handelszeiten komplexer als eine einzelne Cash-Session sind.
if require_open:
    items = [meta for meta in all_items if market_open(meta["ticker"])]
    skipped_closed = [meta for meta in all_items if not market_open(meta["ticker"])]
else:
    items = all_items
    skipped_closed = []
if not items:
    st.warning("Mindestens eine Marktgruppe auswählen."); st.stop()

if st.sidebar.button("Cache leeren und neu scannen"):
    st.cache_data.clear(); st.rerun()

progress = st.progress(0, text="Scan wird vorbereitet …")
status = st.empty()
rows, data15, data5, errors = [], {}, {}, []
start = time_module.time()
for n, meta in enumerate(items, start=1):
    status.info(f"Prüfe {n}/{len(items)}: {meta['name']} ({meta['ticker']})")
    try:
        row, d15, d5 = scan_asset(meta, spread_ok, event_risk, require_open)
        rows.append(row); data15[meta["ticker"]]=d15; data5[meta["ticker"]]=d5
    except Exception as exc:
        errors.append({"Asset":meta["name"], "Ticker":meta["ticker"], "Fehler":str(exc)})
    progress.progress(n/len(items), text=f"{n}/{len(items)} Assets geprüft")
status.empty(); progress.empty()

if not rows:
    st.error("Keine Assets konnten ausgewertet werden.")
    if errors: st.dataframe(pd.DataFrame(errors), hide_index=True)
    st.stop()

scan = pd.DataFrame(rows)
priority = {"🟢 READY LONG":0,"🔴 READY SHORT":0,"🟡 WATCH LONG":1,"🟡 WATCH SHORT":1,"⚪ WAIT":2,"⛔ NO TRADE":3}
scan["Prio"] = scan.Signal.map(priority).fillna(9)
scan = scan.sort_values(["Prio","Score","ADX"], ascending=[True,False,False])

m = st.columns(6)
m[0].metric("Scan aktiv",len(items)); m[1].metric("Analysiert",len(rows)); m[2].metric("Übersprungen",len(skipped_closed))
m[3].metric("READY",int(scan.Signal.str.contains("READY").sum()))
m[4].metric("WATCH",int(scan.Signal.str.contains("WATCH").sum()))
m[5].metric("Laufzeit",f"{time_module.time()-start:.1f}s")

st.header("🔥 Top Market Radar")
show = scan.head(top_n).copy()
st.dataframe(show[["Gruppe","Asset","Ticker","Signal","Qualität","Score","Long","Short","5m Timing","RSI","ADX","ATR %","CRV","Markt","Grund"]]
    .style.format({"RSI":"{:.1f}","ADX":"{:.1f}","ATR %":"{:.2f}","CRV":"{:.2f}"}),
    use_container_width=True, hide_index=True)

with st.expander("Alle analysierten Assets"):
    st.dataframe(scan.drop(columns=["Prio","Checks"]), use_container_width=True, hide_index=True)
with st.expander(f"Geschlossene Märkte übersprungen ({len(skipped_closed)})"):
    if skipped_closed:
        st.dataframe(pd.DataFrame(skipped_closed), use_container_width=True, hide_index=True)
    else:
        st.success("Keine Cash-Märkte übersprungen.")
with st.expander(f"Nicht auswertbare Assets ({len(errors)})"):

    if errors: st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)
    else: st.success("Keine Datenfehler.")

st.header("🔍 Setup-Details")
selected = st.selectbox("Asset", scan.Asset.tolist())
r = scan[scan.Asset == selected].iloc[0]
c=st.columns(6)
c[0].metric("Signal",r.Signal); c[1].metric("Qualität",r.Qualität); c[2].metric("Score",int(r.Score))
c[3].metric("5m Timing",r["5m Timing"]); c[4].metric("Long",int(r.Long)); c[5].metric("Short",int(r.Short))
st.info(r.Grund)
st.dataframe(pd.DataFrame([{"Kriterium":k,"Status":"✅" if v else "❌"} for k,v in r.Checks.items()]), hide_index=True, use_container_width=True)
if np.isfinite(r.Entry):
    levels=st.columns(6)
    for col,label in zip(levels,["Entry","Stop","Ziel 1","Ziel 2","CRV","KO Abstand %"]):
        col.metric(label,f"{r[label]:.4f}" if label not in {"CRV","KO Abstand %"} else f"{r[label]:.2f}")

ticker=r.Ticker
left,right=st.columns(2)
with left:
    st.subheader("5-Minuten Entry-Timing")
    st.line_chart(data5[ticker][["Close","EMA9","EMA20","VWAP"]].tail(160))
with right:
    st.subheader("15-Minuten Trendbestätigung")
    st.line_chart(data15[ticker][["Close","EMA20","EMA50","EMA200","VWAP"]].tail(160))

st.caption(f"Letzte 5m-Kerze: {r['Timestamp 5m']} | Letzte 15m-Kerze: {r['Timestamp 15m']}")
st.warning("V3 scannt ein breites, aber bewusst liquides Universum und nicht jede weltweit notierte Aktie. READY ist nur ein Kandidat. KO-Level, Spread, Emittent, Bezugsverhältnis, Handelszeit und Event-Risiko vor einer manuellen Order prüfen.")
