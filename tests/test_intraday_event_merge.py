"""Offline regressions for merging corporate actions into intraday prices."""
import unittest

import pandas as pd

from yfinance.utils import safe_merge_dfs


class TestIntradayEventMerge(unittest.TestCase):
    def test_discarded_events_preserve_existing_dividends(self):
        for column in ("Stock Splits", "Capital Gains"):
            for event_date in ("2026-10-19", "2026-10-21"):
                with self.subTest(column=column, event_date=event_date):
                    index = pd.date_range("2026-10-20 09:30", periods=2, freq="h", tz="UTC")
                    prices = pd.DataFrame({"Close": [100.0, 101.0], "Dividends": [0.5, 0.0]}, index=index)
                    original = prices.copy(deep=True)
                    events = pd.DataFrame({column: [2.0]}, index=pd.DatetimeIndex([event_date], tz="UTC"))
                    result = safe_merge_dfs(prices, events, "60m")
                    pd.testing.assert_series_equal(result["Dividends"], original["Dividends"])
                    pd.testing.assert_series_equal(result["Close"], original["Close"])
                    self.assertEqual(result[column].tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
