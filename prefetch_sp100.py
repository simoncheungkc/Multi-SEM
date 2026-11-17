"""Prefetch S&P 100 prices and fundamentals so the empirical run is offline-friendly."""

from run_sp100 import download_prices, fetch_fundamentals, load_constituents

if __name__ == "__main__":
    wiki = load_constituents()
    tickers = wiki["Symbol"].dropna().unique().tolist()
    print("tickers", len(tickers))
    px = download_prices(tickers)
    print("prices", px.shape)
    tickers = [t for t in tickers if t in px.columns]
    fund = fetch_fundamentals(tickers, wiki)
    print("fundamentals", fund.shape)
    print(fund.head())
    print("PREFETCH OK")
