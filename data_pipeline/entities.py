"""Map free text (news headlines) to universe symbols by company-name aliases."""

from __future__ import annotations

import re

import pandas as pd

from .config import PipelineConfig
from .universe import load_universe

_SUFFIXES = re.compile(
    r"\b(ltd|limited|corporation|corp|company|co|the|india|indian|"
    r"industries|enterprises|holdings|group)\b",
    re.I,
)
_NONWORD = re.compile(r"[^a-z0-9 &]")

# Well-known short forms the normalised name misses. symbol -> extra aliases.
_MANUAL: dict[str, list[str]] = {
    "RELIANCE": ["reliance industries", "ril"],
    "INFY": ["infosys"],
    "TCS": ["tata consultancy", "tcs"],
    "HDFCBANK": ["hdfc bank"],
    "ICICIBANK": ["icici bank"],
    "SBIN": ["state bank of india", "sbi"],
    "LT": ["larsen & toubro", "larsen and toubro", "l&t"],
    "BHARTIARTL": ["bharti airtel", "airtel"],
    "MARUTI": ["maruti suzuki", "maruti"],
    "M&M": ["mahindra & mahindra", "mahindra and mahindra"],
    "KOTAKBANK": ["kotak mahindra bank", "kotak bank"],
    "AXISBANK": ["axis bank"],
    "BAJFINANCE": ["bajaj finance"],
    "BAJAJFINSV": ["bajaj finserv"],
    "HINDUNILVR": ["hindustan unilever", "hul"],
    "ITC": ["itc"],
    "SUNPHARMA": ["sun pharmaceutical", "sun pharma"],
    "ULTRACEMCO": ["ultratech cement", "ultratech"],
    "TITAN": ["titan company", "titan"],
    "ASIANPAINT": ["asian paints"],
    "NESTLEIND": ["nestle india", "nestle"],
    "POWERGRID": ["power grid"],
    "NTPC": ["ntpc"],
    "ONGC": ["oil and natural gas", "ongc"],
    "COALINDIA": ["coal india"],
    "TATAMOTORS": ["tata motors"],
    "TATASTEEL": ["tata steel"],
    "JSWSTEEL": ["jsw steel"],
    "ADANIENT": ["adani enterprises"],
    "ADANIPORTS": ["adani ports"],
    "WIPRO": ["wipro"],
    "HCLTECH": ["hcl technologies", "hcl tech"],
    "TECHM": ["tech mahindra"],
    "DMART": ["avenue supermarts", "dmart"],
    "PIDILITIND": ["pidilite"],
    "DIVISLAB": ["divi's laboratories", "divis lab"],
    "DRREDDY": ["dr reddy's", "dr reddys"],
    "GRASIM": ["grasim industries", "grasim"],
    "HINDALCO": ["hindalco"],
    "VEDL": ["vedanta"],
    "IOC": ["indian oil"],
    "BPCL": ["bharat petroleum", "bpcl"],
    "GAIL": ["gail india", "gail"],
    "DLF": ["dlf"],
    "SBILIFE": ["sbi life"],
    "HDFCLIFE": ["hdfc life"],
    "ICICIPRULI": ["icici prudential life", "icici pru life"],
    "BAJAJ-AUTO": ["bajaj auto"],
    "EICHERMOT": ["eicher motors", "royal enfield"],
    "HEROMOTOCO": ["hero motocorp"],
    "SHRIRAMFIN": ["shriram finance"],
    "PFC": ["power finance corporation"],
    "RECLTD": ["rec ", "rural electrification"],
    "ZOMATO": ["zomato", "eternal"],
    "PAYTM": ["one 97", "paytm"],
    "NYKAA": ["fsn e-commerce", "nykaa"],
}

_STOPWORDS = {
    "india", "steel", "power", "finance", "motors", "bank", "cement", "life",
    "auto", "pharma", "tech", "energy", "gold", "capital", "chemicals", "paints",
}


def _norm(text: str) -> str:
    t = _NONWORD.sub(" ", str(text).lower())
    t = _SUFFIXES.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


class EntityMatcher:
    def __init__(self, universe: pd.DataFrame) -> None:
        alias_to_syms: dict[str, set[str]] = {}
        for _, row in universe.iterrows():
            sym = row["symbol"]
            # Manual aliases are hand-curated (short forms like "TCS", "ITC", "L&T"),
            # so unlike the auto-derived company name below they are exempt from the
            # min-length check, but still need the same normalization as article
            # text or they'd silently never match.
            for raw in _MANUAL.get(sym, []):
                a = _norm(raw)
                if a:
                    alias_to_syms.setdefault(a, set()).add(sym)
            norm = _norm(row.get("name") or "")
            if len(norm) >= 6 and norm not in _STOPWORDS:
                alias_to_syms.setdefault(norm, set()).add(sym)
        self._alias_to_syms = alias_to_syms
        # longest aliases first so "bajaj finance" wins over "bajaj"
        patterns = sorted(alias_to_syms, key=len, reverse=True)
        self._rx = re.compile(
            r"(?<![a-z0-9])(" + "|".join(re.escape(p) for p in patterns) + r")(?![a-z0-9])",
            re.I,
        )

    def match(self, *texts: str) -> list[str]:
        blob = _norm(" ".join(t for t in texts if t))
        if not blob:
            return []
        found: set[str] = set()
        for m in self._rx.finditer(blob):
            found.update(self._alias_to_syms.get(m.group(1).lower(), ()))
        return sorted(found)


def load_matcher(cfg: PipelineConfig) -> EntityMatcher:
    return EntityMatcher(load_universe(cfg))
