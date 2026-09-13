import base64
import io
import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Optional Prophet import
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Prophet not available, using statsmodels only")

from nexalens.core.logging import get_logger
from nexalens.models.schemas import ForecastModelType, ForecastPoint, ForecastRequest, ForecastResult

logger = get_logger(__name__)


class ForecastingService:
    def __init__(self):
        self.freq_map = {
            "D": "D",
            "W": "W",
            "M": "M",
            "Q": "Q",
            "Y": "A",
        }

    def prepare_timeseries(
        self,
        df: pd.DataFrame,
        date_col: str,
        metric_col: str,
        freq: str,
    ) -> pd.DataFrame:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df = df.set_index(date_col)
        df = df[[metric_col]].rename(columns={metric_col: "y"})
        df = df.asfreq(self.freq_map.get(freq, "M"))
        df["y"] = df["y"].interpolate(method="time")
        df = df.reset_index().rename(columns={date_col: "ds"})
        return df

    def train_prophet(
        self,
        df: pd.DataFrame,
        periods: int,
        confidence: float,
        holidays: bool,
    ) -> ForecastResult:
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, falling back to ARIMA")
            return self.train_arima(df, periods)

        model = Prophet(
            interval_width=confidence,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        if holidays:
            model.add_country_holidays(country_name="US")
        model.fit(df)

        future = model.make_future_dataframe(periods=periods, freq=self._infer_freq(df["ds"]))
        forecast = model.predict(future)

        forecast_points = []
        for _, row in forecast.tail(periods).iterrows():
            forecast_points.append(
                ForecastPoint(
                    date=row["ds"].strftime("%Y-%m-%d"),
                    yhat=row["yhat"],
                    yhat_lower=row["yhat_lower"],
                    yhat_upper=row["yhat_upper"],
                )
            )

        metrics = self._backtest_prophet(df, periods)
        params = {
            "changepoint_prior_scale": 0.05,
            "seasonality_mode": "additive",
            "holidays": holidays,
        }

        plot_b64 = self._generate_plot(df, forecast, periods)

        return ForecastResult(
            forecast=forecast_points,
            metrics=metrics,
            model_type="prophet",
            parameters=params,
            plot_base64=plot_b64,
            explanation="",
        )

    def _infer_freq(self, dates: pd.Series) -> str:
        diff = dates.diff().dropna()
        if diff.empty:
            return "M"
        median_days = diff.dt.days.median()
        if median_days <= 1:
            return "D"
        elif median_days <= 7:
            return "W"
        elif median_days <= 31:
            return "M"
        elif median_days <= 92:
            return "Q"
        return "A"

    def _backtest_prophet(self, df: pd.DataFrame, periods: int) -> dict[str, float]:
        try:
            if len(df) < periods * 2:
                return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

            model = Prophet(interval_width=0.95)
            model.fit(df)

            cv_results = cross_validation(
                model,
                initial=f"{len(df) - periods} days",
                period=f"{periods // 2} days",
                horizon=f"{periods} days",
                parallel="threads",
            )
            perf = performance_metrics(cv_results)
            return {
                "mae": float(perf["mae"].mean()),
                "rmse": float(perf["rmse"].mean()),
                "mape": float(perf["mape"].mean()) if "mape" in perf.columns else 0.0,
            }
        except Exception as e:
            logger.warning("prophet_backtest_failed", error=str(e))
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    def train_arima(
        self,
        df: pd.DataFrame,
        periods: int,
        order: tuple[int, int, int] | None = None,
    ) -> ForecastResult:
        y = df["y"].values
        if order is None:
            order = self._select_arima_order(y)

        model = ARIMA(y, order=order)
        fitted = model.fit()

        forecast = fitted.forecast(steps=periods)
        conf_int = fitted.get_forecast(periods).conf_int(alpha=0.05)

        forecast_points = []
        for i in range(periods):
            forecast_points.append(
                ForecastPoint(
                    date=f"Period_{i+1}",
                    yhat=float(forecast[i]),
                    yhat_lower=float(conf_int[i, 0]),
                    yhat_upper=float(conf_int[i, 1]),
                )
            )

        metrics = {"aic": fitted.aic, "bic": fitted.bic}
        params = {"order": order}

        plot_b64 = self._generate_plot(df, None, periods, yhat=forecast, conf_int=conf_int)

        return ForecastResult(
            forecast=forecast_points,
            metrics=metrics,
            model_type="arima",
            parameters=params,
            plot_base64=plot_b64,
            explanation="",
        )

    def _select_arima_order(self, y: np.ndarray) -> tuple[int, int, int]:
        best_aic = float("inf")
        best_order = (1, 1, 1)
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    if p == 0 and d == 0 and q == 0:
                        continue
                    try:
                        model = ARIMA(y, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue
        return best_order

    def train_ets(
        self,
        df: pd.DataFrame,
        periods: int,
    ) -> ForecastResult:
        y = df["y"].values
        model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12)
        fitted = model.fit()

        forecast = fitted.forecast(periods)
        forecast_points = []
        for i in range(periods):
            forecast_points.append(
                ForecastPoint(
                    date=f"Period_{i+1}",
                    yhat=float(forecast[i]),
                    yhat_lower=float(forecast[i]) * 0.9,
                    yhat_upper=float(forecast[i]) * 1.1,
                )
            )

        metrics = {"aic": fitted.aic}
        params = {"trend": "add", "seasonal": "add"}

        return ForecastResult(
            forecast=forecast_points,
            metrics=metrics,
            model_type="ets",
            parameters=params,
            plot_base64=None,
            explanation="",
        )

    def run_forecast(self, df: pd.DataFrame, request: ForecastRequest) -> ForecastResult:
        prepared = self.prepare_timeseries(
            df, request.date_column, request.metric_column, request.frequency
        )

        if request.model_type == ForecastModelType.PROPHET:
            result = self.train_prophet(
                prepared,
                request.periods,
                request.confidence_interval,
                request.include_holidays,
            )
        elif request.model_type == ForecastModelType.ARIMA:
            result = self.train_arima(prepared, request.periods)
        elif request.model_type == ForecastModelType.ETS:
            result = self.train_ets(prepared, request.periods)
        else:
            result = self.train_prophet(
                prepared,
                request.periods,
                request.confidence_interval,
                request.include_holidays,
            )

        result.explanation = self._generate_explanation(result, request)
        return result

    def _generate_explanation(self, result: ForecastResult, request: ForecastRequest) -> str:
        if not result.forecast:
            return "Insufficient data for forecasting."

        next_value = result.forecast[0].yhat
        last_historical = result.forecast[0].yhat
        trend = "increasing" if len(result.forecast) > 1 and result.forecast[-1].yhat > result.forecast[0].yhat else "decreasing"

        parts = [
            f"{request.model_type.value.upper()} forecast for {request.periods} periods ({request.frequency}).",
            f"Next period prediction: {next_value:,.2f}.",
            f"Overall trend: {trend}.",
        ]

        if result.metrics:
            if "mae" in result.metrics:
                parts.append(f"Backtest MAE: {result.metrics['mae']:,.2f}")
            if "rmse" in result.metrics:
                parts.append(f"Backtest RMSE: {result.metrics['rmse']:,.2f}")

        ci = request.confidence_interval
        parts.append(f"{int(ci*100)}% confidence intervals provided.")

        return "\n".join(parts)

    def _generate_plot(
        self,
        df: pd.DataFrame,
        forecast_df: pd.DataFrame | None = None,
        periods: int = 12,
        yhat: np.ndarray | None = None,
        conf_int: np.ndarray | None = None,
    ) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))

        historical_dates = df["ds"]
        historical_values = df["y"]
        ax.plot(historical_dates, historical_values, "b-", label="Historical", linewidth=1.5)

        if forecast_df is not None:
            future_dates = forecast_df["ds"].tail(periods)
            forecast_values = forecast_df["yhat"].tail(periods)
            lower = forecast_df["yhat_lower"].tail(periods)
            upper = forecast_df["yhat_upper"].tail(periods)
            ax.plot(future_dates, forecast_values, "r--", label="Forecast", linewidth=1.5)
            ax.fill_between(future_dates, lower, upper, alpha=0.2, color="red")
        elif yhat is not None:
            future_idx = range(len(historical_dates), len(historical_dates) + periods)
            ax.plot(future_idx, yhat, "r--", label="Forecast", linewidth=1.5)
            if conf_int is not None:
                ax.fill_between(future_idx, conf_int[:, 0], conf_int[:, 1], alpha=0.2, color="red")

        ax.set_title("Forecast", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")


forecasting_service = ForecastingService()