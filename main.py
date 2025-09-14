# main.py - Binance-only Crypto Bot with OpenAI + keyword alerts + local candlestick pattern checks
# - Fetches spot, candles, OI from Binance
# - Sends snapshot + OpenAI analysis to Telegram
# - If OpenAI analysis contains signal keywords -> immediate alert
# - Also computes simple candlestick-based patterns locally (doji, hammer, engulfing, shooting star)
# - Dedupes Telegram messages via cooldown file

import os, asyncio, time, json
from datetime import datetime
import aiohttp
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ENV
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", 300))
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
MSG_COOLDOWN = int(os.environ.get("MSG_COOLDOWN", 70))

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
BINANCE_CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m&limit=50"
BINANCE_OI_URL = "https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/{method}"

STARTUP_FLAG_PATH = "/tmp/bot_startup_sent"
LAST_MSG_PATH = "/tmp/last_msg_sent"
LAST_SNAPSHOT_PATH = "/tmp/last_snapshot.json"

# Init OpenAI client
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("[WARN] OpenAI init failed:", e)
        client = None
else:
    print("[WARN] OPENAI_API_KEY not set; OpenAI analysis disabled.")

# -----------------------
# Telegram helpers
# -----------------------
async def _really_send_telegram(session: aiohttp.ClientSession, text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram credentials missing.")
        return False
    url = TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN, method="sendMessage")
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        async with session.post(url, json=payload, timeout=15) as resp:
            txt = await resp.text()
            if resp.status != 200:
                print(f"[ERROR] Telegram send failed {resp.status}: {txt}")
                return False
            return True
    except Exception as e:
        print("[ERROR] Telegram request exception:", e)
        return False

def _recent_message_sent() -> bool:
    try:
        if os.path.exists(LAST_MSG_PATH):
            ts = int(open(LAST_MSG_PATH).read().strip())
            if time.time() - ts < MSG_COOLDOWN:
                return True
    except Exception:
        pass
    return False

def _mark_message_sent():
    try:
        with open(LAST_MSG_PATH, "w") as f:
            f.write(str(int(time.time())))
    except Exception as e:
        print("[WARN] cannot write last_msg file:", e)

# -----------------------
# Binance fetch helpers
# -----------------------
async def fetch_ticker(session: aiohttp.ClientSession, symbol: str):
    url = BINANCE_TICKER_URL.format(symbol=symbol)
    try:
        async with session.get(url, timeout=15) as r:
            if r.status != 200:
                text = await r.text()
                raise RuntimeError(f"Ticker HTTP {r.status}: {text[:200]}")
            d = await r.json()
            return {
                "price": float(d.get("lastPrice") or 0),
                "volume": float(d.get("volume") or 0),
                "high": float(d.get("highPrice") or 0),
                "low": float(d.get("lowPrice") or 0),
                "pct": float(d.get("priceChangePercent") or 0),
            }
    except Exception as e:
        return e

async def fetch_candles(session: aiohttp.ClientSession, symbol: str):
    url = BINANCE_CANDLE_URL.format(symbol=symbol)
    try:
        async with session.get(url, timeout=20) as r:
            if r.status != 200:
                text = await r.text()
                raise RuntimeError(f"Klines HTTP {r.status}: {text[:200]}")
            data = await r.json()
            candles = [
                {"open": float(c[1]), "high": float(c[2]), "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])}
                for c in data
            ]
            return candles
    except Exception as e:
        return e

async def fetch_oi(session: aiohttp.ClientSession, symbol: str):
    url = BINANCE_OI_URL.format(symbol=symbol)
    try:
        async with session.get(url, timeout=10) as r:
            if r.status != 200:
                return None
            d = await r.json()
            return float(d.get("openInterest") or 0)
    except Exception:
        return None

# -----------------------
# Local candlestick pattern heuristics (simple)
# -----------------------
def detect_candlestick_patterns_for_symbol(candles):
    if not candles or len(candles) < 2:
        return []
    last = candles[-1]
    prev = candles[-2]
    patterns = []
    def body(c): return abs(c["close"] - c["open"])
    def range_size(c): return c["high"] - c["low"]
    def is_bull(c): return c["close"] > c["open"]
    def is_bear(c): return c["close"] < c["open"]
    last_body = body(last)
    last_range = range_size(last)
    prev_body = body(prev)
    # Doji
    if last_range > 0 and (last_body / last_range) < 0.15:
        patterns.append("Doji")
    lower_wick = min(last["open"], last["close"]) - last["low"]
    upper_wick = last["high"] - max(last["open"], last["close"])
    if last_body > 0:
        if lower_wick > 2 * last_body and lower_wick > upper_wick:
            if is_bull(last):
                patterns.append("Hammer (bullish)")
            else:
                patterns.append("Hammer-like")
        if upper_wick > 2 * last_body and upper_wick > lower_wick:
            if is_bear(last):
                patterns.append("Shooting Star (bearish)")
            else:
                patterns.append("Shooting Star-like")
    # Engulfing
    if is_bull(last) and is_bear(prev) and last["close"] > prev["open"] and last["open"] < prev["close"]:
        patterns.append("Bullish Engulfing")
    if is_bear(last) and is_bull(prev) and last["open"] > prev["close"] and last["close"] < prev["open"]:
        patterns.append("Bearish Engulfing")
    # Spinning top
    if last_range > 0 and (last_body / last_range) < 0.25 and (upper_wick / last_range) > 0.2 and (lower_wick / last_range) > 0.2:
        patterns.append("Spinning Top")
    return patterns

def detect_candlestick_patterns(candle_map):
    res = {}
    for s, candles in candle_map.items():
        if candles and isinstance(candles, list):
            res[s] = detect_candlestick_patterns_for_symbol(candles)
        else:
            res[s] = []
    return res

# -----------------------
# OpenAI analysis (robust)
# -----------------------
async def openai_analyze(market_map, candle_map):
    if not client:
        return None
    try:
        lines = []
        for s in SYMBOLS:
            d = market_map.get(s) or {}
            if d.get("price") is not None:
                lines.append(f"{s}: price={d['price']} vol24h={d.get('volume','NA')} OI={d.get('oi','NA')}")
            else:
                lines.append(f"{s}: DATA_MISSING")
        prompt_parts = [
            "You are a concise crypto technical analyst.",
            "Given the following spot summary and recent candlestick OHLC data for BTCUSDT, ETHUSDT, and SOLUSDT,",
            "identify common chart patterns (e.g., flag, triangle, head & shoulders, double top/bottom), state bias (bullish/bearish/neutral),",
            "and suggest possible buy/sell signals in one short paragraph.",
            "",
            "Spot summary (DATA_MISSING means that data fetch failed):",
            "\n".join(lines),
            ""
        ]
        for s in SYMBOLS:
            candles = candle_map.get(s) or []
            if candles:
                last10 = candles[-10:]
                candle_texts = [f"[{c.get('open')},{c.get('high')},{c.get('low')},{c.get('close')}]" for c in last10]
                prompt_parts.append(f"{s} last 10 candles (OHLC): " + ", ".join(candle_texts))
            else:
                prompt_parts.append(f"{s} last 10 candles (OHLC): DATA_MISSING")
        prompt = "\n".join(prompt_parts)
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.2,
            ),
        )
        text = resp.choices[0].message.content.strip()
        lines_out = [ln.strip() for ln in text.splitlines() if ln.strip()]
        summary = "\n".join(lines_out[:8]) if len(lines_out) > 8 else text
        return summary
    except Exception as e:
        print("[ERROR] OpenAI analyze failed:", e)
        return None

# -----------------------
# Alerts: keyword detection + numeric change + local candlestick patterns
# -----------------------
KEYWORD_LIST = [
    "breakout","breakouts","buy","sell","bullish","bearish","double bottom","double top",
    "head and shoulders","head & shoulders","flag","flags","triangle","ascending triangle",
    "descending triangle","engulfing","hammer","doji","spinning top","pullback","retest"
]

ALERT_PRICE_PCT = float(os.environ.get("ALERT_PRICE_PCT", 0.01))   # 1%
ALERT_OI_PCT = float(os.environ.get("ALERT_OI_PCT", 0.10))        # 10%

def detect_keywords(text):
    if not text:
        return []
    txt = text.lower()
    found = []
    for k in KEYWORD_LIST:
        if k in txt:
            found.append(k)
    return sorted(set(found))

def load_last_snapshot():
    try:
        if os.path.exists(LAST_SNAPSHOT_PATH):
            return json.load(open(LAST_SNAPSHOT_PATH, "r"))
    except Exception as e:
        print("[WARN] load_last_snapshot:", e)
    return {}

def save_last_snapshot(snapshot):
    try:
        with open(LAST_SNAPSHOT_PATH, "w") as f:
            json.dump(snapshot, f)
    except Exception as e:
        print("[WARN] save_last_snapshot:", e)

async def check_and_send_alerts(session, market_map, candle_map, analysis_text):
    # 1) OpenAI keyword alerts
    kws = detect_keywords(analysis_text)
    if kws:
        msg = "⚠️ *OpenAI Keywords Detected*: " + ", ".join(kws)
        if analysis_text and len(analysis_text) < 800:
            msg += "\n\nSummary:\n" + analysis_text
        else:
            msg += "\n\n(analysis longer - see snapshot)"
        await _really_send_telegram(session, msg)

    # 2) Local candlestick pattern alerts
    local_patterns = detect_candlestick_patterns(candle_map)
    pattern_msgs = []
    for s, pats in local_patterns.items():
        if pats:
            pattern_msgs.append(f"{s}: " + ", ".join(pats))
    if pattern_msgs:
        await _really_send_telegram(session, "🔔 *Local Candlestick Patterns:*\n" + "\n".join(pattern_msgs))

    # 3) Numeric change alerts vs last snapshot
    last = load_last_snapshot()
    now_snapshot = {}
    alerts = []
    for s, d in market_map.items():
        if not d or d.get("price") is None:
            continue
        now_price = float(d["price"])
        now_oi = None
        try:
            now_oi = float(d.get("oi")) if d.get("oi") is not None else None
        except:
            now_oi = None
        now_snapshot[s] = {"price": now_price, "oi": now_oi}

        prev = last.get(s)
        if prev:
            try:
                prev_price = float(prev.get("price"))
                if prev_price > 0:
                    price_pct = abs(now_price - prev_price) / prev_price
                    if price_pct >= ALERT_PRICE_PCT:
                        dirc = "↑" if now_price > prev_price else "↓"
                        alerts.append(f"{s}: price {dirc} {price_pct*100:.2f}% ({prev_price:.2f} → {now_price:.2f})")
            except Exception:
                pass
            try:
                prev_oi = prev.get("oi")
                if prev_oi is not None and now_oi is not None and float(prev_oi) > 0:
                    oi_pct = abs(now_oi - float(prev_oi)) / float(prev_oi)
                    if oi_pct >= ALERT_OI_PCT:
                        dirc = "↑" if now_oi > float(prev_oi) else "↓"
                        alerts.append(f"{s}: OI {dirc} {oi_pct*100:.2f}% ({prev_oi} → {now_oi})")
            except Exception:
                pass

    save_last_snapshot(now_snapshot)
    if alerts:
        text = "📣 *Numeric Alerts*\n" + "\n".join(alerts)
        await _really_send_telegram(session, text)

# -----------------------
# Snapshot builder + dedupe
# -----------------------
def _clean_analysis_text(text, max_lines=5):
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    filtered = []
    for ln in lines:
        if ln.lower().startswith("last 10 candles") or ln[:2].lstrip().isdigit():
            continue
        filtered.append(ln)
    if not filtered:
        filtered = lines
    short = []
    for ln in filtered:
        if len(ln) > 220:
            ln = ln[:200].rstrip() + "…"
        short.append(ln)
        if len(short) >= max_lines:
            break
    return "\\n".join(short)

async def send_snapshot_message(session, market_map, candle_map, analysis_raw):
    if _recent_message_sent():
        print("[INFO] Recent message sent — skipping duplicate snapshot (cooldown).")
        return False
    header_lines = []
    for s in SYMBOLS:
        d = market_map.get(s)
        if d and d.get("price") is not None:
            header_lines.append(f"*{s}*: {d['price']} (24hVol={d.get('volume','NA')}, OI={d.get('oi','NA')})")
        else:
            header_lines.append(f"*{s}*: DATA_MISSING")
    analysis_clean = _clean_analysis_text(analysis_raw, max_lines=5)
    msg = f"*Snapshot (UTC {datetime.utcnow().strftime('%H:%M')})*\\n" + "\\n".join(header_lines)
    if analysis_clean:
        msg += "\\n\\n🧠 Analysis:\\n" + analysis_clean
    sent = await _really_send_telegram(session, msg)
    if sent:
        _mark_message_sent()
    return sent

# -----------------------
# Startup helpers and main loop
# -----------------------
async def send_startup_once(session):
    if os.path.exists(STARTUP_FLAG_PATH):
        print("[INFO] startup flag exists, skipping startup alert for this container.")
        return
    ok = await _really_send_telegram(session, "*Bot online — Binance only*")
    if ok:
        try:
            with open(STARTUP_FLAG_PATH, "w") as f:
                f.write(str(int(time.time())))
        except Exception as e:
            print("[WARN] could not write startup flag:", e)

async def send_startup_test(session):
    symbols = ["BTCUSDT", "ETHUSDT"]
    market_map = {}
    candle_map = {}
    for s in symbols:
        t = await fetch_ticker(session, s)
        c = await fetch_candles(session, s)
        oi = await fetch_oi(session, s)
        if isinstance(t, Exception):
            market_map[s] = {"price": None, "volume": None, "oi": None}
        else:
            if t:
                t["oi"] = oi
                market_map[s] = t
            else:
                market_map[s] = {"price": None, "volume": None, "oi": None}
        candle_map[s] = c if not isinstance(c, Exception) else []
    analysis = await openai_analyze(market_map, candle_map)
    header = []
    for s in symbols:
        d = market_map.get(s)
        if d and d.get("price") is not None:
            header.append(f"*{s}*: {d['price']} (24hVol={d['volume']}, OI={d.get('oi','NA')})")
        else:
            header.append(f"*{s}*: DATA_MISSING")
    msg = "*Startup test — forced snapshot*\\n" + "\\n".join(header)
    if analysis:
        msg += "\\n\\n🧠 Analysis:\\n" + _clean_analysis_text(analysis, max_lines=6)
    await _really_send_telegram(session, msg)

async def periodic_task():
    async with aiohttp.ClientSession() as session:
        await send_startup_once(session)
        if TEST_MODE:
            print("[INFO] TEST_MODE=1 -> sending startup forced test (BTC/ETH).")
            await asyncio.sleep(1)
            await send_startup_test(session)
        while True:
            start = time.time()
            market_map = {s: {"price": None, "volume": None, "oi": None} for s in SYMBOLS}
            candle_map = {s: [] for s in SYMBOLS}
            tasks = []
            for s in SYMBOLS:
                tasks.append(fetch_ticker(session, s))
                tasks.append(fetch_candles(session, s))
                tasks.append(fetch_oi(session, s))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, s in enumerate(SYMBOLS):
                tick_res = results[i*3]
                cand_res = results[i*3 + 1]
                oi_res = results[i*3 + 2]
                if isinstance(tick_res, Exception):
                    print(f"[WARN] ticker fetch exception for {s}: {tick_res}")
                elif tick_res is None:
                    print(f"[WARN] ticker returned None for {s}")
                else:
                    market_map[s]["price"] = tick_res.get("price")
                    market_map[s]["volume"] = tick_res.get("volume")
                if isinstance(cand_res, Exception):
                    print(f"[WARN] candles fetch exception for {s}: {cand_res}")
                    candle_map[s] = []
                elif cand_res is None:
                    print(f"[WARN] candles None for {s}")
                    candle_map[s] = []
                else:
                    candle_map[s] = cand_res
                if isinstance(oi_res, Exception):
                    print(f"[WARN] oi fetch exception for {s}: {oi_res}")
                else:
                    market_map[s]["oi"] = oi_res
            print("[DEBUG] fetch summary:")
            for s in SYMBOLS:
                print(f"  {s}: price={market_map[s]['price']}  candles={len(candle_map[s])}  oi={market_map[s]['oi']}")
            analysis = await openai_analyze(market_map, candle_map)
            # alerts: keywords & local candlestick & numeric diffs
            await check_and_send_alerts(session, market_map, candle_map, analysis)
            # send deduped snapshot
            await send_snapshot_message(session, market_map, candle_map, analysis)
            elapsed = time.time() - start
            await asyncio.sleep(max(0, POLL_INTERVAL - elapsed))

def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] Telegram env vars missing. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return
    print("[INFO] Starting Binance-only bot (alerts + local patterns)...")
    try:
        asyncio.run(periodic_task())
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")

if __name__ == "__main__":
    main()
