"""Replace headquarters pins whose first geocode missed the SEC street."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
UA = "MultiSEM-replication/1.0 (academic headquarters geocode)"

# Explicit queries use the SEC business street, or, where that filing line is
# only a post-office box, the executive-office street named in the latest 10-K.
FIXES = {
    "ACN": ("1 Grand Canal Square, Dublin, Ireland", ["grand canal", "dublin", "ireland"], ["united states"], "DUBLIN", "L2"),
    "LIN": ("43 Church Street West, Woking, United Kingdom", ["church street", "woking", "united kingdom"], ["united states"], "WOKING", "X0"),
    "MDT": ("Parkmore Business Park West, Galway, Ireland", ["parkmore", "galway", "ireland"], ["united states"], "GALWAY", "L2"),
    "INTC": ("2200 Mission College Boulevard, Santa Clara, California", ["2200", "mission college", "california"], [], None, None),
    "CRM": ("415 Mission Street, San Francisco, California", ["415", "mission street", "california"], [], None, None),
    "DHR": ("2200 Pennsylvania Avenue NW, Washington, District of Columbia", ["2200", "pennsylvania", "district of columbia"], [], None, None),
    "EXC": ("10 South Dearborn Street, Chicago, Illinois", ["10", "dearborn", "illinois"], [], None, None),
    "HD": ("2455 Paces Ferry Road Northwest, Atlanta, Georgia", ["2455", "paces ferry", "georgia"], [], None, None),
    "UNH": ("1 Health Drive, Eden Prairie, Minnesota", ["health drive", "eden prairie", "minnesota"], [], None, None),
    "USB": ("200 South Sixth Street, Minneapolis, Minnesota", ["200", "sixth", "minnesota"], [], None, None),
    "MMM": ("3M Center, Maplewood, Minnesota", ["3m", "minnesota"], ["united states"] if False else [], None, None),
    "BMY": ("Province Line Road, Princeton, New Jersey", ["princeton", "new jersey"], [], None, None),
    "MO": ("6601 West Broad Street, Richmond, Virginia", ["6601", "broad", "virginia"], [], None, None),
    "MRK": ("126 East Lincoln Avenue, Rahway, New Jersey", ["126", "lincoln", "new jersey"], [], None, None),
    "ABBV": ("1 North Waukegan Road, North Chicago, Illinois", ["waukegan", "illinois"], [], None, None),
    "UNP": ("1400 Douglas Street, Omaha, Nebraska", ["1400", "douglas", "nebraska"], [], None, None),
    "SCHW": ("3000 Schwab Way, Westlake, Texas", ["schwab way", "texas"], [], None, None),
    "SBUX": ("2401 Utah Avenue South, Seattle, Washington", ["utah avenue", "seattle", "washington"], [], None, None),
    "TXN": ("12500 TI Boulevard, Dallas, Texas", ["12500", "dallas", "texas"], [], None, None),
    "V": ("900 Metro Center Boulevard, Foster City, California", ["metro center", "foster city", "california"], [], "FOSTER CITY", "CA"),
}


def nominatim(query: str) -> list[dict]:
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": query, "format": "jsonv2", "limit": 5}
    )
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    return json.loads(urllib.request.urlopen(request, timeout=60).read())


def main():
    frame = pd.read_csv(DATA / "sp100_headquarters.csv")
    audit = json.loads((DATA / "sp100_headquarters_audit.json").read_text(encoding="utf-8"))
    audit_by = {row["ticker"]: row for row in audit}
    for i, (ticker, (query, required, banned, city, state)) in enumerate(FIXES.items()):
        if i:
            time.sleep(1.1)
        hits = nominatim(query)
        chosen = None
        for hit in hits:
            display = (hit.get("display_name") or "").lower()
            if all(token in display for token in required) and not any(token in display for token in banned):
                chosen = hit
                break
        if chosen is None:
            print(f"FAIL {ticker}")
            for hit in hits:
                print("   ", hit.get("display_name"))
            continue
        lat = round(float(chosen["lat"]), 6)
        lon = round(float(chosen["lon"]), 6)
        if ticker in set(frame["ticker"]):
            frame.loc[frame["ticker"] == ticker, "latitude"] = lat
            frame.loc[frame["ticker"] == ticker, "longitude"] = lon
            if city:
                frame.loc[frame["ticker"] == ticker, "city"] = city
            if state:
                frame.loc[frame["ticker"] == ticker, "state"] = state
        else:
            frame = pd.concat(
                [
                    frame,
                    pd.DataFrame(
                        [{
                            "ticker": ticker,
                            "company": "Linde plc",
                            "city": city,
                            "state": state,
                            "latitude": lat,
                            "longitude": lon,
                        }]
                    ),
                ],
                ignore_index=True,
            )
        audit_by[ticker] = {
            "ticker": ticker,
            "quality": "repaired",
            "query": query,
            "matched_address": chosen.get("display_name"),
            "category": chosen.get("category"),
        }
        print(f"OK {ticker:6s} {lat:.5f}, {lon:.5f}  {chosen.get('display_name')[:110]}")
    # Restore the constituent order, including LIN.
    order = [t for t in pd.read_parquet(DATA / "sp100_prices.parquet").columns]
    # LIN is in the price file. ACN too. Keep price-file order.
    missing = [t for t in order if t not in set(frame["ticker"])]
    if missing:
        raise SystemExit(f"still missing {missing}")
    frame = frame.set_index("ticker").loc[order].reset_index()
    frame.to_csv(DATA / "sp100_headquarters.csv", index=False)
    (DATA / "sp100_headquarters_audit.json").write_text(
        json.dumps([audit_by[t] for t in order if t in audit_by], indent=2),
        encoding="utf-8",
    )
    print("rows", len(frame))


if __name__ == "__main__":
    main()
