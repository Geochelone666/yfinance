"""
Offline tests for PriceHistory._fix_zeroes() with interval "1d".

The fine-grained fetch inside _reconstruct_intervals_batch() is replaced with
synthetic intraday data, so these tests never touch the network.

Regression coverage for two bugs in Adj Close reconstruction:

1. Price-basis mixing: the fine-grained block can sit on a different price
   basis than the daily block (e.g. with prepost=True the aggregated "last"
   Close is a post-market trade, not the official close). Copying its absolute
   'Adj Close' corrupted repaired rows; the adjustment ratio must be applied
   to the block's own Close instead.

2. Missing anchor when the tagged row is the last row of the data and an
   ex-dividend date: the adjustment ratio changes on the ex-date, so the
   anchor ratio must be derived by walking forward through the dividend with
   Yahoo's factor (1 - div/prev_close), not copied from the previous row.
"""

import datetime
import unittest

import numpy as np
import pandas as pd

import yfinance as yf


TZ = "America/New_York"
DAILY_RATIO_PRE = 0.96  # daily Adj ratio before the ex-dividend date
FINE_RATIO_PRE = 0.995  # fine-grained fetch uses its own adjustment basis
DIV = 1.0


def _make_daily(closes, div_day):
    """5 recent 'daily' rows. Dividend DIV paid on div_day index."""
    idx = pd.date_range(end=pd.Timestamp.now(tz=TZ).normalize() - pd.Timedelta(days=2),
                        periods=5, freq="B", tz=TZ)
    closes = np.array(closes, dtype=float)
    df = pd.DataFrame(index=idx)
    df["Open"] = closes
    df["High"] = closes + 1.0
    df["Low"] = closes - 1.0
    df["Close"] = closes
    ratio_post = DAILY_RATIO_PRE / (1.0 - DIV / closes[div_day - 1])
    ratios = np.where(np.arange(5) < div_day, DAILY_RATIO_PRE, ratio_post)
    df["Adj Close"] = closes * ratios
    df["Volume"] = 1_000_000
    df["Dividends"] = 0.0
    df.loc[df.index[div_day], "Dividends"] = DIV
    df["Stock Splits"] = 0.0
    return df


def _make_fine(daily, drift=None, nan_close_days=()):
    """Synthetic 1h bars matching 'daily', but with a different adjustment
    basis (FINE_RATIO_PRE/1.0) and optionally a drifting post-market last bar.
    """
    div = daily["Dividends"] != 0.0
    bars = []
    for dt, row in daily.iterrows():
        ratio = 1.0 if div.loc[dt] or (div.idxmax() < dt and div.any()) else FINE_RATIO_PRE
        for hour in range(9, 17):
            bdt = dt + datetime.timedelta(hours=hour)
            close = row["Close"]
            bars.append((bdt, close, close, close, close, 1000.0,
                         DIV if (div.loc[dt] and hour == 9) else 0.0, ratio))
        # post-market last bar: this is what prepost=True appends
        last_close = row["Close"]
        if drift and dt in drift:
            last_close = drift[dt]
        nan_close = dt in nan_close_days
        if nan_close:
            bars[-8:] = [(b[0], np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, b[7])
                         for b in bars[-8:]]
            continue
        bars.append((dt + datetime.timedelta(hours=20),
                     row["Close"], row["Close"], row["Close"], last_close,
                     1000.0, 0.0, ratio))
    idx = pd.DatetimeIndex([b[0] for b in bars])
    fine = pd.DataFrame({
        "Open": [b[1] for b in bars],
        "High": [b[2] for b in bars],
        "Low": [b[3] for b in bars],
        "Close": [b[4] for b in bars],
        "Volume": [b[5] for b in bars],
        "Dividends": [b[6] for b in bars],
        "Stock Splits": 0.0,
    }, index=idx)
    fine["Adj Close"] = fine["Close"] * pd.Series([b[7] for b in bars], index=idx)
    fine["Repaired?"] = False
    return fine


class TestFixZeroesOffline(unittest.TestCase):

    def _hist(self, fine):
        dat = yf.Ticker("OFFLINE.TEST")
        dat._tz = TZ  # avoid network tz lookup
        hist = dat._lazy_load_price_history()
        hist.history = lambda **kwargs: fine.copy()  # no network fetch
        return hist

    def test_adj_close_repaired_on_block_close_basis(self):
        closes = [100, 102, 104, 103, 105]
        daily = _make_daily(closes, div_day=3)
        dt0 = daily.index[0]
        # fine last trade of day 0 drifted 3% post-market
        fine = _make_fine(daily, drift={dt0: closes[0] * 1.03})
        hist = self._hist(fine)

        df_bad = daily.copy()
        df_bad.loc[dt0, "Adj Close"] = 0.0
        repaired = hist._fix_zeroes(df_bad, "1d", TZ, prepost=False)

        # Repaired Adj Close must equal daily Close x true daily ratio,
        # not the fine data's post-market-basis value.
        self.assertAlmostEqual(repaired.loc[dt0, "Adj Close"],
                               daily.loc[dt0, "Adj Close"], places=6)
        self.assertAlmostEqual(repaired.loc[dt0, "Adj Close"],
                               closes[0] * DAILY_RATIO_PRE, places=6)

    def test_adj_close_anchor_walks_dividend_on_tagged_last_row(self):
        closes = [100, 102, 104, 103, 105]
        daily = _make_daily(closes, div_day=4)  # dividend on the last row
        dt_last = daily.index[-1]
        fine = _make_fine(daily, drift={dt_last: closes[-1] * 1.03})
        hist = self._hist(fine)

        df_bad = daily.copy()
        df_bad.loc[dt_last, "Adj Close"] = 0.0
        repaired = hist._fix_zeroes(df_bad, "1d", TZ, prepost=False)

        expected_ratio = DAILY_RATIO_PRE / (1.0 - DIV / closes[-2])
        self.assertAlmostEqual(repaired.loc[dt_last, "Adj Close"],
                               daily.loc[dt_last, "Adj Close"], places=6)
        self.assertAlmostEqual(repaired.loc[dt_last, "Adj Close"],
                               closes[-1] * expected_ratio, places=6)

    def test_close_and_adj_close_both_tagged_use_fine_row_consistently(self):
        closes = [100, 102, 104, 103, 105]
        daily = _make_daily(closes, div_day=3)
        dt0 = daily.index[0]
        drifted = closes[0] * 1.03
        fine = _make_fine(daily, drift={dt0: drifted})
        hist = self._hist(fine)

        df_bad = daily.copy()
        df_bad.loc[dt0, ["Close", "Adj Close"]] = 0.0
        repaired = hist._fix_zeroes(df_bad, "1d", TZ, prepost=False)

        # When Close itself is repaired from fine data, Adj Close comes from
        # the same fine row (consistent basis), scaled to the daily anchor.
        ratio_post = DAILY_RATIO_PRE / (1.0 - DIV / closes[2])
        correction = (ratio_post * (1.0 - DIV / closes[2])) / FINE_RATIO_PRE
        self.assertAlmostEqual(repaired.loc[dt0, "Close"], drifted, places=6)
        self.assertAlmostEqual(repaired.loc[dt0, "Adj Close"],
                               drifted * FINE_RATIO_PRE * correction, places=6)

    def test_tagged_row_restored_when_fine_data_missing(self):
        closes = [100, 102, 104, 103, 105]
        daily = _make_daily(closes, div_day=3)
        dt0 = daily.index[0]
        fine = _make_fine(daily, nan_close_days=(dt0,))
        hist = self._hist(fine)

        df_bad = daily.copy()
        df_bad.loc[dt0, "Adj Close"] = 0.0
        repaired = hist._fix_zeroes(df_bad, "1d", TZ, prepost=False)

        # No fine data for the tagged day -> no Adj Close fabrication;
        # the original value is restored instead.
        self.assertEqual(repaired.loc[dt0, "Adj Close"], 0.0)
        self.assertEqual(repaired.loc[dt0, "Close"], closes[0])


if __name__ == "__main__":
    unittest.main()
