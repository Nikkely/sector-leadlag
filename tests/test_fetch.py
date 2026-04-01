from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from leadlag.fetch import _validate_download, align_dates


class TestAlignDates:
    def test_shift_and_intersect(self):
        us_dates = pd.bdate_range("2024-01-08", periods=5)  # Mon-Fri
        jp_dates = pd.bdate_range("2024-01-09", periods=5)  # Tue-Mon

        us_df = pd.DataFrame({"A": range(5)}, index=us_dates)
        jp_df = pd.DataFrame({"B": range(5)}, index=jp_dates)

        us_aligned, jp_aligned = align_dates(us_df, jp_df)

        # US shifted by 1 bday, so US Mon->Tue, etc.
        # Common dates should be the intersection
        assert len(us_aligned) == len(jp_aligned)
        assert (us_aligned.index == jp_aligned.index).all()

    def test_no_overlap_returns_empty(self):
        us_dates = pd.bdate_range("2024-01-01", periods=3)
        jp_dates = pd.bdate_range("2024-06-01", periods=3)

        us_df = pd.DataFrame({"A": [1, 2, 3]}, index=us_dates)
        jp_df = pd.DataFrame({"B": [4, 5, 6]}, index=jp_dates)

        us_aligned, jp_aligned = align_dates(us_df, jp_df)
        assert len(us_aligned) == 0
        assert len(jp_aligned) == 0


class TestValidateDownload:
    def test_raises_on_empty_data(self):
        data = pd.DataFrame()
        with pytest.raises(ValueError, match="no valid data returned"):
            _validate_download(data, ["BAD1", "BAD2"], "Test")

    def test_warns_and_drops_missing_tickers(self):
        dates = pd.bdate_range("2024-01-08", periods=3)
        arrays = [["Close", "Close"], ["GOOD", "BAD"]]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        data = pd.DataFrame(
            [[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]],
            index=dates,
            columns=index,
        )
        result = _validate_download(data, ["GOOD", "BAD"], "Test")
        assert "BAD" not in result["Close"].columns

    def test_passes_with_valid_data(self):
        dates = pd.bdate_range("2024-01-08", periods=3)
        arrays = [["Close", "Close"], ["A", "B"]]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        data = pd.DataFrame(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
            index=dates,
            columns=index,
        )
        result = _validate_download(data, ["A", "B"], "Test")
        assert list(result["Close"].columns) == ["A", "B"]
