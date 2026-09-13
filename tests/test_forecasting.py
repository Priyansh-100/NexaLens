import pytest
import pandas as pd
import numpy as np

from nexalens.analytics.forecasting import ForecastingService
from nexalens.models.schemas import ForecastModelType, ForecastRequest


class TestForecastingService:
    @pytest.fixture
    def service(self):
        return ForecastingService()

    @pytest.fixture
    def sample_timeseries(self):
        dates = pd.date_range("2020-01-01", periods=36, freq="M")
        values = 1000 + np.cumsum(np.random.randn(36) * 50) + 50 * np.sin(np.arange(36) * 2 * np.pi / 12)
        return pd.DataFrame({"ds": dates, "y": values})

    def test_prepare_timeseries(self, service, sample_timeseries):
        df = sample_timeseries.rename(columns={"ds": "date", "y": "revenue"})
        prepared = service.prepare_timeseries(df, "date", "revenue", "M")
        assert "ds" in prepared.columns
        assert "y" in prepared.columns
        assert prepared["y"].isna().sum() == 0

    def test_prophet_forecast(self, service, sample_timeseries):
        request = ForecastRequest(
            metric_column="revenue",
            date_column="date",
            periods=6,
            frequency="M",
            model_type=ForecastModelType.PROPHET,
        )
        result = service.run_forecast(sample_timeseries, request)
        assert result.model_type == "prophet"
        assert len(result.forecast) == 6
        assert all(p.yhat_lower <= p.yhat <= p.yhat_upper for p in result.forecast)
        assert result.plot_base64 is not None
        assert "mae" in result.metrics

    def test_arima_forecast(self, service, sample_timeseries):
        request = ForecastRequest(
            metric_column="revenue",
            date_column="date",
            periods=6,
            frequency="M",
            model_type=ForecastModelType.ARIMA,
        )
        result = service.run_forecast(sample_timeseries, request)
        assert result.model_type == "arima"
        assert len(result.forecast) == 6

    def test_ets_forecast(self, service, sample_timeseries):
        request = ForecastRequest(
            metric_column="revenue",
            date_column="date",
            periods=6,
            frequency="M",
            model_type=ForecastModelType.ETS,
        )
        result = service.run_forecast(sample_timeseries, request)
        assert result.model_type == "ets"
        assert len(result.forecast) == 6

    def test_insufficient_data(self, service):
        df = pd.DataFrame({
            "ds": pd.date_range("2020-01-01", periods=2, freq="M"),
            "y": [100, 110]
        })
        request = ForecastRequest(
            metric_column="y",
            date_column="ds",
            periods=6,
            frequency="M",
        )
        result = service.run_forecast(df, request)
        assert len(result.forecast) == 6