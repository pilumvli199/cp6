# Binance-only Crypto Bot (Alerts + Candlestick Patterns)

This release adds:
- OpenAI analysis (GPT-4o-mini) of candles + spot + OI
- Immediate alerts if OpenAI analysis contains signal keywords
- Local candlestick heuristics (doji, hammer, shooting star, engulfing, spinning top)
- Numeric change alerts (price pct, OI pct) with last-snapshot persistence
- Deduped snapshot messages using cooldown

Files:
- main.py, requirements.txt, .env.example, Procfile, Dockerfile, README.md

Deploy on Railway:
1. Push repo.
2. Set Variables in Railway: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, OPENAI_API_KEY, OPENAI_MODEL (optional)
3. Set TEST_MODE=1 and POLL_INTERVAL=60 for quick testing then set TEST_MODE=0 for normal.
