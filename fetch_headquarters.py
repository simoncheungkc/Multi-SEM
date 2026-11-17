"""Principal executive offices for the S&P 100 replication sample.

Addresses come from the SEC submissions JSON (business address on the latest
filing index). US street addresses are geocoded with the Census Bureau
one-line geocoder. Addresses outside the United States, and US addresses the
Census geocoder does not match, are geocoded with Nominatim.
"""

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
OUT = DATA / "sp100_headquarters.csv"
AUDIT = DATA / "sp100_headquarters_audit.json"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def sec_ticker_map() -> dict[str, dict]:
    payload = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    out = {}
    for row in payload.values():
        out[str(row["ticker"]).upper()] = {
            "cik": int(row["cik_str"]),
            "company": str(row["title"]),
        }
    return out


def business_address(cik: int) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    payload = json.loads(_get(url))
    address = payload.get("addresses", {}).get("business", {})
    return {
        "company": payload.get("name") or "",
        "street": " ".join(
            part.strip()
            for part in (address.get("street1") or "", address.get("street2") or "")
            if part and part.strip()
        ),
        "city": (address.get("city") or "").strip(),
        "state": (address.get("stateOrCountry") or "").strip(),
        "zip": (address.get("zipCode") or "").strip(),
    }


_NUMBER_WORDS = {
    "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4", "FIVE": "5",
    "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9", "TEN": "10",
}
_STREET = {
    "AVE": "Avenue", "AV": "Avenue", "ST": "Street", "RD": "Road", "BLVD": "Boulevard",
    "PKWY": "Parkway", "DR": "Drive", "LN": "Lane", "HWY": "Highway", "CTR": "Center",
    "N": "North", "S": "South", "E": "East", "W": "West", "NE": "Northeast",
    "NW": "Northwest", "SE": "Southeast", "SW": "Southwest",
}
_DROP = {"FL", "FLOOR", "STE", "SUITE", "UNIT", "APT"}
_REJECT = {"highway", "bench", "bus_stop", "traffic_signals", "crossing"}
_TICKER_ALIAS = {"BK": "BNY"}


def _clean_street(street: str) -> str:
    parts = street.replace(".", " ").replace(",", " ").split()
    cleaned = []
    for i, part in enumerate(parts):
        token = part.upper()
        if token in _DROP:
            break
        if i == 0 and token in _NUMBER_WORDS:
            cleaned.append(_NUMBER_WORDS[token])
        elif token in _STREET:
            cleaned.append(_STREET[token])
        else:
            cleaned.append(part)
    return " ".join(cleaned)


def _nominatim(params: dict) -> list[dict]:
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(params)
    return json.loads(_get(url))


def _prefer(hits: list[dict]) -> dict | None:
    usable = [h for h in hits if h.get("category") not in _REJECT and h.get("type") not in _REJECT]
    if not usable:
        usable = hits
    preferred = [h for h in usable if h.get("category") in {"office", "building", "commercial", "amenity", "place"}]
    hit = (preferred or usable or [None])[0]
    if hit is None:
        return None
    return {
        "latitude": float(hit["lat"]),
        "longitude": float(hit["lon"]),
        "geocoder": "nominatim",
        "matched_address": hit.get("display_name"),
        "category": hit.get("category"),
        "type": hit.get("type"),
    }


def _score(hit: dict, street: str, city: str) -> int:
    display = (hit.get("display_name") or "").lower()
    score = 0
    tokens = street.split()
    number = tokens[0] if tokens and tokens[0][:1].isdigit() else ""
    if number and f" {number} " in f" {display} ":
        score += 5
    for token in tokens[1:]:
        if len(token) > 3 and token.lower() in display:
            score += 2
    if city and city.lower() in display:
        score += 3
    if hit.get("category") in {"office", "building", "commercial"}:
        score += 2
    if hit.get("category") in _REJECT or hit.get("type") in _REJECT:
        score -= 8
    return score


def _in_region(display: str, state: str, state_name: str) -> bool:
    text = (display or "").lower()
    if state == "DC":
        return "district of columbia" in text
    return state_name.lower() in text


def geocode_office(company: str, street: str, city: str, state: str, state_name: str, zip_code: str, country: str, domestic: bool) -> dict | None:
    street = _clean_street(street)
    zip5 = zip_code.split("-")[0][:5]
    place = state_name if domestic else country
    queries = [
        ", ".join(part for part in (street, city, place, zip5, country) if part),
        ", ".join(part for part in (company, city, place, country) if part),
        ", ".join(part for part in (city, place, country) if part),
    ]
    if not domestic and " " in city:
        queries.insert(1, ", ".join(part for part in (street, city.split()[0], country) if part))
    extra = {"countrycodes": "us"} if domestic else {}
    fallback = None
    for i, query in enumerate(queries):
        if i:
            time.sleep(1.1)
        hits = _nominatim({"q": query, "format": "jsonv2", "limit": 5, **extra})
        eligible = []
        for hit in hits:
            display = hit.get("display_name") or ""
            if domestic and not _in_region(display, state, state_name):
                continue
            if not domestic and country.lower() not in display.lower():
                continue
            eligible.append(hit)
        if not eligible:
            continue
        best = max(eligible, key=lambda hit: _score(hit, street, city))
        chosen = _prefer([best])
        chosen["quality"] = "street" if _score(best, street, city) >= 5 else "place"
        if chosen["quality"] == "street":
            return chosen
        fallback = fallback or chosen
    return fallback


def main():
    priced = set(pd.read_parquet(DATA / "sp100_prices.parquet").columns)
    tickers = [t for t in pd.read_csv(DATA / "sp100_constituents.csv")["Symbol"].dropna().tolist() if t in priced]
    if "GOOG" in tickers:
        raise SystemExit("GOOG is in the constituent file; the sample keeps GOOGL only")
    if tickers.count("GOOGL") != 1:
        raise SystemExit(f"expected one GOOGL row, found {tickers.count('GOOGL')}")
    sec = sec_ticker_map()
    rows = []
    audit = []
    for i, ticker in enumerate(tickers):
        key = _TICKER_ALIAS.get(ticker.upper(), ticker.upper())
        aliases = [key, key.replace("-", ".")]
        info = next((sec[a] for a in aliases if a in sec), None)
        if info is None:
            raise SystemExit(f"SEC ticker map has no entry for {ticker}")
        time.sleep(0.15)
        address = business_address(info["cik"])
        company = address["company"] or info["company"]
        state = address["state"]
        # SEC uses two-letter US postal codes for domestic offices and ISO-like
        # codes (for example L2, X0, I2) for foreign offices.
        codes = {
            code: name.title()
            for code, name in (
                line.split("|", 1) for line in (ROOT / "_sec_codes.txt").read_text(encoding="utf-8").splitlines() if "|" in line
            )
        }
        domestic = state in {
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI",
            "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN",
            "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH",
            "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA",
            "WI", "WV", "WY",
        }
        country = "United States" if domestic else codes.get(state, state)
        state_name = "District of Columbia" if state == "DC" else codes.get(state, state).title()
        time.sleep(1.1)
        geocode = geocode_office(
            company, address["street"], address["city"], state, state_name, address["zip"], country, domestic
        )
        if geocode is None:
            print(f"  FAILED {ticker}: {address['street']}, {address['city']}, {state}", flush=True)
            audit.append({"ticker": ticker, "failed": True, "street": address["street"], "city": address["city"], "state": state})
            continue
        lat = geocode["latitude"]
        lon = geocode["longitude"]
        continental = 24.0 <= lat <= 50.0 and -125.0 <= lon <= -66.0
        rows.append(
            {
                "ticker": ticker,
                "company": company,
                "city": address["city"],
                "state": state,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
            }
        )
        audit.append(
            {
                "ticker": ticker,
                "cik": info["cik"],
                "street": address["street"],
                "zip": address["zip"],
                "geocoder": geocode["geocoder"],
                "quality": geocode.get("quality"),
                "category": geocode.get("category"),
                "matched_address": geocode["matched_address"],
                "continental_us": continental,
                "domestic_postal": domestic,
            }
        )
        print(
            f"{i + 1:3d} {ticker:6s} {address['city']}, {state}  {lat:.4f}, {lon:.4f}  {geocode.get('quality')}  {geocode.get('category')}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.to_csv(OUT, index=False)
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if len(frame) != len(tickers) or frame["ticker"].nunique() != len(tickers):
        raise SystemExit(f"expected {len(tickers)} unique tickers, got {len(frame)}")
    googl = frame.loc[frame["ticker"] == "GOOGL"].iloc[0]
    if googl["city"].upper() != "MOUNTAIN VIEW" or googl["state"] != "CA":
        raise SystemExit(f"GOOGL office is not Mountain View, CA: {googl.to_dict()}")
    frame.to_csv(OUT, index=False)
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    outside = frame.loc[~frame.apply(lambda r: 24 <= r.latitude <= 50 and -125 <= r.longitude <= -66, axis=1)]
    print(f"wrote {OUT} ({len(frame)} rows)")
    print(f"outside continental bounds: {outside['ticker'].tolist()}")


if __name__ == "__main__":
    main()
