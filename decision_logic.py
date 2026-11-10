from dataclasses import dataclass
from typing import Optional

READY_SCORE = 75
WATCH_SCORE = 60
MIN_RR = 1.8
MIN_ADX = 20.0
MAX_EMA20_EXTENSION_ATR = 1.25

@dataclass
class SignalInput:
    price: float
    prev_close: float
    ema20: float
    ema50: float
    ema200: float
    prev_ema20: float
    rsi: float
    macd_hist: float
    adx: float
    vwap: float
    atr: float
    long_score: int
    short_score: int
    rr: Optional[float] = None
    market_open: bool = True
    data_fresh: bool = True
    confirmed_bar: bool = True
    spread_ok: bool = True
    event_risk: bool = False
    volatility_ok: bool = True


def decide_signal(x: SignalInput) -> dict:
    """Return NO TRADE / READY LONG / READY SHORT / WATCH LONG / WATCH SHORT / WAIT.

    The hard risk filters have priority over technical scores.
    """
    blockers = []
    if not x.market_open:
        blockers.append("Markt geschlossen")
    if not x.data_fresh:
        blockers.append("Daten veraltet/fehlen")
    if not x.confirmed_bar:
        blockers.append("15m-Kerze nicht bestätigt")
    if x.rr is None or x.rr < MIN_RR:
        blockers.append(f"CRV unter {MIN_RR}" if x.rr is not None else "CRV fehlt")
    if not x.spread_ok:
        blockers.append("Spread zu hoch")
    if x.event_risk:
        blockers.append("Event-Risiko")
    if not x.volatility_ok:
        blockers.append("Volatilität ungeeignet")
    if x.atr <= 0:
        blockers.append("ATR ungültig")

    if blockers:
        return {
            "signal": "⛔ NO TRADE",
            "direction": "NONE",
            "quality": "BLOCKED",
            "score": max(x.long_score, x.short_score),
            "reason": "; ".join(blockers),
        }

    long_checks = {
        "Trend": x.price > x.ema20 > x.ema50 > x.ema200,
        "Momentum": 52 <= x.rsi <= 68 and x.macd_hist > 0,
        "ADX": x.adx >= MIN_ADX,
        "VWAP": x.price > x.vwap,
        "2-Bar": x.prev_close > x.prev_ema20 and x.price > x.ema20,
        "Nicht überdehnt": (x.price - x.ema20) <= MAX_EMA20_EXTENSION_ATR * x.atr,
    }
    short_checks = {
        "Trend": x.price < x.ema20 < x.ema50 < x.ema200,
        "Momentum": 32 <= x.rsi <= 48 and x.macd_hist < 0,
        "ADX": x.adx >= MIN_ADX,
        "VWAP": x.price < x.vwap,
        "2-Bar": x.prev_close < x.prev_ema20 and x.price < x.ema20,
        "Nicht überdehnt": (x.ema20 - x.price) <= MAX_EMA20_EXTENSION_ATR * x.atr,
    }

    long_ready = x.long_score >= READY_SCORE and all(long_checks.values())
    short_ready = x.short_score >= READY_SCORE and all(short_checks.values())

    def quality(score: int) -> str:
        if score >= 90:
            return "A+"
        if score >= 80:
            return "A"
        if score >= 75:
            return "B"
        return "WATCH"

    if long_ready and not short_ready:
        return {"signal": "🟢 READY LONG", "direction": "LONG", "quality": quality(x.long_score),
                "score": x.long_score, "checks": long_checks, "reason": "Alle LONG-Kernbedingungen erfüllt"}

    if short_ready and not long_ready:
        return {"signal": "🔴 READY SHORT", "direction": "SHORT", "quality": quality(x.short_score),
                "score": x.short_score, "checks": short_checks, "reason": "Alle SHORT-Kernbedingungen erfüllt"}

    # WATCH: klare technische Tendenz, aber mindestens eine Kernbedingung fehlt.
    if x.long_score >= WATCH_SCORE and x.long_score > x.short_score:
        missing = [name for name, ok in long_checks.items() if not ok]
        return {"signal": "🟡 WATCH LONG", "direction": "LONG", "quality": "WATCH",
                "score": x.long_score, "checks": long_checks,
                "reason": "Fehlt: " + ", ".join(missing) if missing else "Score noch unter READY"}

    if x.short_score >= WATCH_SCORE and x.short_score > x.long_score:
        missing = [name for name, ok in short_checks.items() if not ok]
        return {"signal": "🟡 WATCH SHORT", "direction": "SHORT", "quality": "WATCH",
                "score": x.short_score, "checks": short_checks,
                "reason": "Fehlt: " + ", ".join(missing) if missing else "Score noch unter READY"}

    return {
        "signal": "⚪ WAIT",
        "direction": "NONE",
        "quality": "NEUTRAL",
        "score": max(x.long_score, x.short_score),
        "checks": long_checks if x.long_score >= x.short_score else short_checks,
        "reason": "Keine ausreichend klare LONG- oder SHORT-Struktur",
    }
