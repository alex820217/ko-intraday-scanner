import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, time
from zoneinfo import ZoneInfo
from decision_logic import SignalInput, decide_signal

st.set_page_config(page_title="KO Intraday Scanner", page_icon="📈", layout="wide")
st.title("📈 KO Intraday Scanner")
st.caption("15-Minuten-Signale mit READY / WATCH / WAIT / NO TRADE und LONG-/SHORT-Backtest")

WATCHLIST = {
    "Microsoft": "MSFT", "Apple": "AAPL", "Nvidia": "NVDA", "Amazon": "AMZN",
    "DAX": "^GDAXI", "Nasdaq 100": "^NDX", "S&P 500": "^GSPC",
    "Gold": "GC=F", "Öl WTI": "CL=F", "EUR/USD": "EURUSD=X",
}

INTERVAL, PERIOD = "15m", "30d"
MIN_RR = 1.8


def ser(df, col):
    x = df[col]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return pd.to_numeric(x, errors="coerce")


@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker):
    x = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=True,
                    prepost=False, progress=False, threads=False)
    if x.empty:
        return x
    out = pd.DataFrame(index=x.index)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = ser(x, c)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def market_is_open(ticker):
    if ticker == "^GDAXI":
        now = datetime.now(ZoneInfo("Europe/Berlin"))
        return now.weekday() < 5 and time(9, 0) <= now.time() < time(17, 30)
    if ticker in {"MSFT", "AAPL", "NVDA", "AMZN", "^NDX", "^GSPC"}:
        now = datetime.now(ZoneInfo("America/New_York"))
        return now.weekday() < 5 and time(9, 30) <= now.time() < time(16, 0)
    return True  # FX/Futures: keine einfache Cash-Session-Sperre


def add_indicators(df):
    d = df.copy()
    c, h, l, v = d.Close, d.High, d.Low, d.Volume.fillna(0)
    for n in [9, 20, 50, 200]:
        d[f"EMA{n}"] = ta.trend.EMAIndicator(c, window=n).ema_indicator()
    d["RSI"] = ta.momentum.RSIIndicator(c, window=14).rsi()
    d["ATR"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    d["ADX"] = ta.trend.ADXIndicator(h, l, c, window=14).adx()
    macd = ta.trend.MACD(c)
    d["MACD_HIST"] = macd.macd_diff()
    typical = (h + l + c) / 3
    session = pd.Series(d.index.date, index=d.index)
    cv = v.groupby(session).cumsum()
    d["VWAP"] = ((typical * v).groupby(session).cumsum() / cv.replace(0, np.nan)).fillna(d.EMA20)
    d["VOL_MA20"] = v.rolling(20).mean()
    d["VOL_RATIO"] = v / d.VOL_MA20.replace(0, np.nan)
    return d.dropna(subset=["EMA200", "RSI", "ATR", "ADX", "MACD_HIST", "VWAP"])


def score_row(d, i):
    r, p = d.iloc[i], d.iloc[i-1]
    price, atr = float(r.Close), float(r.ATR)
    vol_present = d.Volume.iloc[max(0, i-30):i+1].sum() > 0
    vol_ok = (not vol_present) or pd.isna(r.VOL_RATIO) or r.VOL_RATIO >= 0.8
    long_checks = {
        "Trend": price > r.EMA20 > r.EMA50 > r.EMA200,
        "Momentum": 52 <= r.RSI <= 68 and r.MACD_HIST > 0,
        "ADX": r.ADX >= 20, "VWAP": price > r.VWAP, "Volumen": vol_ok,
        "2-Bar": p.Close > p.EMA20 and price > r.EMA20,
        "Nicht überdehnt": (price-r.EMA20) <= 1.25*atr,
    }
    short_checks = {
        "Trend": price < r.EMA20 < r.EMA50 < r.EMA200,
        "Momentum": 32 <= r.RSI <= 48 and r.MACD_HIST < 0,
        "ADX": r.ADX >= 20, "VWAP": price < r.VWAP, "Volumen": vol_ok,
        "2-Bar": p.Close < p.EMA20 and price < r.EMA20,
        "Nicht überdehnt": (r.EMA20-price) <= 1.25*atr,
    }
    weights = {"Trend":25,"Momentum":15,"ADX":15,"VWAP":10,"Volumen":10,"2-Bar":15,"Nicht überdehnt":10}
    ls = sum(weights[k] for k, ok in long_checks.items() if ok)
    ss = sum(weights[k] for k, ok in short_checks.items() if ok)
    return ls, ss


def analyse(name, ticker, spread_ok=True, event_risk=False):
    raw = load_data(ticker)
    if len(raw) < 230:
        return None, raw
    d = add_indicators(raw)
    if len(d) < 3:
        return None, d
    i = len(d)-1
    r, p = d.iloc[i], d.iloc[i-1]
    ls, ss = score_row(d, i)
    price, atr = float(r.Close), float(r.ATR)
    preferred = "LONG" if ls >= ss else "SHORT"
    entry = price
    stop = entry - 1.2*atr if preferred == "LONG" else entry + 1.2*atr
    t1 = entry + 1.8*(entry-stop) if preferred == "LONG" else entry - 1.8*(stop-entry)
    t2 = entry + 2.8*(entry-stop) if preferred == "LONG" else entry - 2.8*(stop-entry)
    rr = abs(t1-entry)/abs(entry-stop) if entry != stop else 0
    decision = decide_signal(SignalInput(
        price=price, prev_close=float(p.Close), ema20=float(r.EMA20), ema50=float(r.EMA50),
        ema200=float(r.EMA200), prev_ema20=float(p.EMA20), rsi=float(r.RSI),
        macd_hist=float(r.MACD_HIST), adx=float(r.ADX), vwap=float(r.VWAP), atr=atr,
        long_score=ls, short_score=ss, rr=rr, market_open=market_is_open(ticker),
        data_fresh=True, confirmed_bar=True, spread_ok=spread_ok, event_risk=event_risk,
        volatility_ok=(0.05 <= atr/price*100 <= 5.0)
    ))
    direction = decision["direction"]
    if direction == "LONG":
        stop, t1, t2, ko = price-1.2*atr, price+1.8*1.2*atr, price+2.8*1.2*atr, price-1.7*atr
    elif direction == "SHORT":
        stop, t1, t2, ko = price+1.2*atr, price-1.8*1.2*atr, price-2.8*1.2*atr, price+1.7*atr
    else:
        stop=t1=t2=ko=np.nan
    ko_pct = abs(price-ko)/price*100 if np.isfinite(ko) else np.nan
    return {
        "Asset":name, "Ticker":ticker, "Signal":decision["signal"], "Qualität":decision["quality"],
        "Score":decision["score"], "Long":ls, "Short":ss, "Grund":decision["reason"],
        "Kurs":price, "RSI":float(r.RSI), "ADX":float(r.ADX), "ATR %":atr/price*100,
        "Entry": price if direction in {"LONG","SHORT"} else np.nan, "Stop":stop,
        "Ziel 1":t1, "Ziel 2":t2, "CRV":rr if direction in {"LONG","SHORT"} else np.nan,
        "KO Referenz":ko, "KO Abstand %":ko_pct, "Checks":decision.get("checks", {}),
        "Timestamp":d.index[-1], "Markt":"🟢 Offen" if market_is_open(ticker) else "🔴 Geschlossen",
    }, d


def backtest(d, ready_score=75, max_bars=32, costs_bps=8):
    trades=[]
    i=2
    while i < len(d)-1:
        r,p=d.iloc[i],d.iloc[i-1]
        ls,ss=score_row(d,i)
        price,atr=float(r.Close),float(r.ATR)
        preferred="LONG" if ls>=ss else "SHORT"
        stop0=price-1.2*atr if preferred=="LONG" else price+1.2*atr
        target0=price+1.8*(price-stop0) if preferred=="LONG" else price-1.8*(stop0-price)
        rr=abs(target0-price)/abs(price-stop0)
        dec=decide_signal(SignalInput(price,float(p.Close),float(r.EMA20),float(r.EMA50),float(r.EMA200),
            float(p.EMA20),float(r.RSI),float(r.MACD_HIST),float(r.ADX),float(r.VWAP),atr,ls,ss,
            rr=rr,market_open=True,data_fresh=True,confirmed_bar=True,spread_ok=True,event_risk=False,
            volatility_ok=(0.05 <= atr/price*100 <= 5.0)))
        if dec["signal"] not in {"🟢 READY LONG","🔴 READY SHORT"} or dec["score"] < ready_score:
            i+=1; continue
        direction=dec["direction"]
        ei=i+1; entry=float(d.iloc[ei].Open); risk=1.2*atr
        stop=entry-risk if direction=="LONG" else entry+risk
        target=entry+1.8*risk if direction=="LONG" else entry-1.8*risk
        end=min(ei+max_bars,len(d)-1); exit_price=float(d.iloc[end].Close); reason="TIME"; xi=end
        for j in range(ei,end+1):
            hi,lo=float(d.iloc[j].High),float(d.iloc[j].Low)
            sh=lo<=stop if direction=="LONG" else hi>=stop
            th=hi>=target if direction=="LONG" else lo<=target
            if sh: xi,exit_price,reason=j,stop,"STOP"; break
            if th: xi,exit_price,reason=j,target,"TARGET"; break
        gross=(exit_price-entry)/risk if direction=="LONG" else (entry-exit_price)/risk
        net=gross-(2*costs_bps/10000*entry)/risk
        trades.append({"Einstieg":d.index[ei],"Ausstieg":d.index[xi],"Richtung":direction,
                       "Score":dec["score"],"Entry":entry,"Stop":stop,"Ziel":target,
                       "Exit":exit_price,"Grund":reason,"Netto R":net,"Gewinn":net>0})
        i=xi+1
    t=pd.DataFrame(trades)
    if not t.empty: t["Equity R"]=t["Netto R"].cumsum()
    return t


def summary(t, direction=None):
    s=t if direction is None else t[t.Richtung==direction]
    if s.empty: return {"Trades":0,"Trefferquote %":0,"Netto R":0,"Ø R":0,"Profit Factor":0,"Max DD R":0}
    pos=s.loc[s["Netto R"]>0,"Netto R"].sum(); neg=abs(s.loc[s["Netto R"]<0,"Netto R"].sum())
    curve=s["Netto R"].cumsum(); dd=curve-curve.cummax()
    return {"Trades":len(s),"Trefferquote %":100*s.Gewinn.mean(),"Netto R":s["Netto R"].sum(),
            "Ø R":s["Netto R"].mean(),"Profit Factor":pos/neg if neg else np.inf,"Max DD R":abs(dd.min())}

st.sidebar.header("Risiko-Overrides")
st.sidebar.caption("Yahoo liefert keinen echten KO-Produktspread oder Eventkalender. Diese beiden Sperren deshalb vor einem Trade manuell setzen.")
spread_ok=st.sidebar.checkbox("Spread des konkreten KO-Produkts akzeptabel", True)
event_risk=st.sidebar.checkbox("Relevantes Event unmittelbar bevorstehend", False)
bt_score=st.sidebar.slider("Backtest READY-Mindestscore",75,100,75,5)
bt_bars=st.sidebar.number_input("Max. Haltedauer (15m-Kerzen)",4,96,32,4)
bt_costs=st.sidebar.number_input("Kosten/Slippage je Seite (bps)",0,100,8,1)

rows=[]; datasets={}
with st.spinner("Marktdaten werden analysiert …"):
    for name,ticker in WATCHLIST.items():
        try:
            row,d=analyse(name,ticker,spread_ok,event_risk)
            if row: rows.append(row); datasets[name]=d
        except Exception as e:
            st.warning(f"{name}: {e}")
if not rows:
    st.error("Keine ausreichenden 15-Minuten-Daten verfügbar."); st.stop()

scan=pd.DataFrame(rows).sort_values(["Score","ADX"],ascending=False)
st.header("🌍 Handelsradar")
st.dataframe(scan[["Asset","Signal","Qualität","Score","Long","Short","RSI","ADX","ATR %","CRV","KO Abstand %","Markt","Grund"]]
             .style.format({"RSI":"{:.1f}","ADX":"{:.1f}","ATR %":"{:.2f}","CRV":"{:.2f}","KO Abstand %":"{:.2f}"}),
             use_container_width=True,hide_index=True)

st.header("🔥 READY Setups")
ready=scan[scan.Signal.isin(["🟢 READY LONG","🔴 READY SHORT"])]
if ready.empty: st.info("Aktuell kein vollständig bestätigtes READY-Setup.")
else:
    for _,r in ready.iterrows():
        st.success(f"{r.Signal} | {r.Asset} | {r.Qualität} | Score {r.Score} | Entry {r.Entry:.4f} | Stop {r.Stop:.4f} | Ziel 1 {r['Ziel 1']:.4f}")

st.header("🔍 Detailanalyse")
asset=st.selectbox("Asset auswählen",scan.Asset.tolist(),key="detail")
r=next(x for x in rows if x["Asset"]==asset); d=datasets[asset]
c=st.columns(6)
c[0].metric("Entscheidung",r["Signal"]); c[1].metric("Qualität",r["Qualität"]); c[2].metric("Score",r["Score"])
c[3].metric("Long",r["Long"]); c[4].metric("Short",r["Short"]); c[5].metric("Markt",r["Markt"])
st.info(f"Begründung: {r['Grund']}")
if r["Checks"]:
    st.dataframe(pd.DataFrame([{"Kriterium":k,"Status":"✅" if v else "❌"} for k,v in r["Checks"].items()]),hide_index=True,use_container_width=True)
if np.isfinite(r["Entry"]):
    x=st.columns(5); x[0].metric("Entry",f"{r['Entry']:.4f}"); x[1].metric("Stop",f"{r['Stop']:.4f}")
    x[2].metric("Ziel 1",f"{r['Ziel 1']:.4f}"); x[3].metric("Ziel 2",f"{r['Ziel 2']:.4f}"); x[4].metric("CRV",f"{r['CRV']:.2f}")
st.line_chart(d[["Close","EMA20","EMA50","EMA200","VWAP"]].tail(160),use_container_width=True)

st.header("🧪 Backtest derselben READY-Regeln")
ba=st.selectbox("Backtest Asset",scan.Asset.tolist(),key="backtest")
tr=backtest(datasets[ba],ready_score=bt_score,max_bars=int(bt_bars),costs_bps=int(bt_costs))
if tr.empty: st.info("Keine READY-Trades im verfügbaren Datenfenster.")
else:
    stats=pd.DataFrame([summary(tr),summary(tr,"LONG"),summary(tr,"SHORT")],index=["Gesamt","LONG","SHORT"])
    st.dataframe(stats.style.format({"Trefferquote %":"{:.1f}","Netto R":"{:.2f}","Ø R":"{:.2f}","Profit Factor":"{:.2f}","Max DD R":"{:.2f}"}),use_container_width=True)
    st.line_chart(tr.set_index("Ausstieg")[["Equity R"]],use_container_width=True)
    st.dataframe(tr.sort_values("Einstieg",ascending=False),use_container_width=True,hide_index=True)
    st.download_button("Backtest CSV herunterladen",tr.to_csv(index=False).encode(),f"backtest_{WATCHLIST[ba].replace('^','')}_15m.csv","text/csv")

st.warning("Entscheidungshilfe, keine Gewinngarantie. Bei KO-Produkten zusätzlich tatsächlichen KO-Level, Geld-Brief-Spread, Bezugsverhältnis, Emittent, Handelszeit und Finanzierung prüfen. Orders bleiben manuell.")
