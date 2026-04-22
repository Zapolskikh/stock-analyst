from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StockData:
    """
    Normalised snapshot of a single ticker.
    All ratio fields are floats; None means data unavailable.
    """

    # Identity
    ticker:   str = ""
    name:     str = ""
    sector:   str = ""
    industry: str = ""

    # Price & market
    price:       Optional[float] = None
    market_cap:  Optional[float] = None   # USD

    # Growth (YoY %)
    revenue_growth_yoy:  Optional[float] = None
    eps_growth_yoy:      Optional[float] = None

    # Margins (0–1)
    gross_margin:     Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin:       Optional[float] = None

    # Returns
    roe:  Optional[float] = None
    roic: Optional[float] = None

    # Free cash flow
    fcf:          Optional[float] = None   # USD absolute
    fcf_margin:   Optional[float] = None   # FCF / Revenue
    fcf_yield:    Optional[float] = None   # FCF / Market Cap

    # Valuation
    pe:           Optional[float] = None
    forward_pe:   Optional[float] = None
    ev_ebitda:    Optional[float] = None
    ps:           Optional[float] = None
    peg:          Optional[float] = None
    p_fcf:        Optional[float] = None

    # Debt
    debt_to_equity:  Optional[float] = None
    interest_coverage: Optional[float] = None

    # Dividends
    dividend_yield:  Optional[float] = None
    payout_ratio:    Optional[float] = None

    # Technical (computed from price history)
    price_vs_50ma:   Optional[float] = None   # % above/below
    price_vs_200ma:  Optional[float] = None
    momentum_3m:     Optional[float] = None   # total return %
    momentum_6m:     Optional[float] = None
    momentum_12m:    Optional[float] = None
    drawdown_52w:    Optional[float] = None   # max drawdown from 52w high
    relative_strength_vs_spy: Optional[float] = None  # 12m RS

    # Volatility
    beta:         Optional[float] = None
    volatility_annualised: Optional[float] = None   # std dev of daily returns * sqrt(252)

    # Sector median benchmarks (populated by cleaner)
    sector_median: dict = field(default_factory=dict)

    # Derived
    stock_type: Optional[str] = None   # set by classifier
