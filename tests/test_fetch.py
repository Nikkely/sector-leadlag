import numpy as np
import pandas as pd
import pytest

from leadlag.fetch import align_dates


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
