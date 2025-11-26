# 📂 FinLagX Data Sources

This document lists the financial assets, macroeconomic indicators, and external data sources used in the **FinLagX Project**. The coverage is designed to provide a comprehensive multi-asset, multi-signal view of global market dynamics.

---

## 1. Global Equity Indices (7)

| Asset                     | Ticker | Coverage  | Notes                       |
| ------------------------- | ------ | --------- | --------------------------- |
| S&P 500 (US)              | ^GSPC  | 20+ years | Primary US equity benchmark |
| NASDAQ 100 (US)           | ^NDX   | 20+ years | Tech-heavy index            |
| Dow Jones Industrial Avg. | ^DJI   | 20+ years | Blue-chip US index          |
| Nikkei 225 (Japan)        | ^N225  | 20+ years | Proxy for Japanese economy  |
| FTSE 100 (UK)             | ^FTSE  | 20+ years | Major UK market indicator   |
| DAX 40 (Germany)          | ^GDAXI | 20+ years | Key European index          |
| NIFTY 50 (India)          | ^NSEI  | 20+ years | Indian market benchmark     |

---

## 2. Commodities (4)

| Commodity       | Ticker | Coverage  | Notes                        |
| --------------- | ------ | --------- | ---------------------------- |
| Crude Oil (WTI) | CL=F   | 20+ years | Global oil benchmark         |
| Gold            | GC=F   | 20+ years | Inflation & safe haven asset |
| Copper          | HG=F   | 20+ years | Economic cycle indicator     |
| Silver          | SI=F   | 20+ years | Industrial & monetary metal  |

---

## 3. Currency Pairs (4)

| Pair    | Ticker   | Coverage  | Notes                      |
| ------- | -------- | --------- | -------------------------- |
| USD/JPY | JPY=X    | 20+ years | Reflects Asia/US sentiment |
| EUR/USD | EURUSD=X | 20+ years | Most traded FX pair        |
| GBP/USD | GBPUSD=X | 20+ years | UK/US macro proxy          |
| USD/CNY | CNY=X    | 20+ years | China macro exposure       |

---

## 4. Volatility & Bonds (2)

| Asset                     | Ticker | Coverage  | Notes                       |
| ------------------------- | ------ | --------- | --------------------------- |
| VIX (US Volatility Index) | ^VIX   | 20+ years | Market fear index           |
| US 10-Year Treasury Yield | ^TNX   | 20+ years | Key interest rate indicator |

---

## 5. Emerging Markets (2)

| Asset                       | Ticker | Coverage  | Notes                |
| --------------------------- | ------ | --------- | -------------------- |
| MSCI Emerging Markets (ETF) | EEM    | 20+ years | Broad EM exposure    |
| Brazil Bovespa Index        | ^BVSP  | 20+ years | Latin American proxy |

---

## 6. Crypto (Optional Modern Signal)

| Asset         | Ticker  | Coverage   | Notes                       |
| ------------- | ------- | ---------- | --------------------------- |
| Bitcoin (USD) | BTC-USD | ~10 years  | High-impact narrative asset |

---

## 7. Macro-Economic Indicators

| Indicator         | Source                         | Coverage  |
| ----------------- | ------------------------------ | --------- |
| Inflation (CPI)   | World Bank / FRED              | 20+ years |
| GDP Growth        | World Bank / IMF               | 20+ years |
| Interest Rates    | Federal Reserve, RBI, ECB, BOJ | 20+ years |
| Unemployment Rate | FRED, World Bank               | 20+ years |
| Fed Funds Rate    | Federal Reserve                | 20+ years |

---

## 8. News & Social Media Data

- **Financial News Headlines**
  - Yahoo Finance, Reuters, Bloomberg (via RSS + archives)
  - Scraped with `newspaper3k` and `BeautifulSoup`
- **Social Media (Optional)**
  - Twitter/X API for financial sentiment
  - Reddit (`r/WallStreetBets`, `r/Investing`)

---

  This diverse set of assets + external signals ensures FinLagX can capture **global interdependencies**, **macro effects**, and **sentiment-driven lead-lag dynamics**.
