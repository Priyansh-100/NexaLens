from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from nexalens.api.auth import get_current_user
from nexalens.core.exceptions import ValidationError
from nexalens.core.logging import get_logger
from nexalens.models.schemas import (
    FinancialModelRequest,
    FinancialModelResult,
    FinancialModelType,
    ForecastModelType,
    ForecastRequest,
    ForecastResult,
    User,
)
from nexalens.models.session import get_db_session
from nexalens.analytics.financial import financial_modeling_service
from nexalens.analytics.forecasting import forecasting_service

router = APIRouter(prefix="/analytics", tags=["analytics"])
logger = get_logger(__name__)


@router.get("/financial-model/templates")
async def get_financial_templates(
    current_user: User = Depends(get_current_user),
) -> dict[str, dict]:
    templates = {}
    for model_type in FinancialModelType:
        templates[model_type.value] = financial_modeling_service.get_assumptions_template(model_type)
    return templates


@router.post("/financial-model", response_model=FinancialModelResult)
async def run_financial_model(
    request: FinancialModelRequest,
    current_user: User = Depends(get_current_user),
) -> FinancialModelResult:
    try:
        result = financial_modeling_service.run_model(request)
        return result
    except Exception as e:
        logger.error("financial_model_api_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/forecast", response_model=ForecastResult)
async def run_forecast(
    request: ForecastRequest,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> ForecastResult:
    try:
        from nexalens.sql.schema import schema_service
        from nexalens.sql.service import sql_service
        import pandas as pd

        if not request.data_source_ids:
            raise ValidationError("data_source_ids required for forecasting")

        schema = await schema_service.get_schema(session, request.data_source_ids[0])

        table = request.metric_column
        metric_col = request.metric_column
        date_col = request.date_column

        sql = f"SELECT {date_col}, {metric_col} FROM {table} ORDER BY {date_col}"
        sql_result = await sql_service.execute_sql(session, sql)

        if sql_result.row_count < 3:
            raise ValidationError(f"Insufficient data: {sql_result.row_count} rows (minimum 3)")

        df = pd.DataFrame(sql_result.rows)

        result = forecasting_service.run_forecast(df, request)
        return result

    except ValidationError:
        raise
    except Exception as e:
        logger.error("forecast_api_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/forecast/backtest")
async def backtest_forecast(
    request: ForecastRequest,
    test_periods: int = 6,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
):
    try:
        from nexalens.sql.schema import schema_service
        from nexalens.sql.service import sql_service
        import pandas as pd

        if not request.data_source_ids:
            raise ValidationError("data_source_ids required")

        schema = await schema_service.get_schema(session, request.data_source_ids[0])

        table = request.metric_column
        metric_col = request.metric_column
        date_col = request.date_column

        sql = f"SELECT {date_col}, {metric_col} FROM {table} ORDER BY {date_col}"
        sql_result = await sql_service.execute_sql(session, sql)

        if sql_result.row_count < test_periods + 3:
            raise ValidationError("Insufficient data for backtesting")

        df = pd.DataFrame(sql_result.rows)

        from nexalens.analytics.forecasting import ForecastingService
        fs = ForecastingService()
        prepared = fs.prepare_timeseries(df, date_col, metric_col, request.frequency)

        train_df = prepared.iloc[:-test_periods]
        test_df = prepared.iloc[-test_periods:]

        fc_request = ForecastRequest(
            metric_column=metric_col,
            date_column=date_col,
            periods=test_periods,
            frequency=request.frequency,
            confidence_interval=request.confidence_interval,
            include_holidays=request.include_holidays,
            model_type=request.model_type,
        )

        result = forecasting_service.run_forecast(train_df, fc_request)

        actuals = test_df["y"].values
        predicted = [p.yhat for p in result.forecast]

        import numpy as np
        mae = float(np.mean(np.abs(np.array(actuals) - np.array(predicted))))
        rmse = float(np.sqrt(np.mean((np.array(actuals) - np.array(predicted)) ** 2)))
        mape = float(np.mean(np.abs((np.array(actuals) - np.array(predicted)) / np.array(actuals))) * 100)

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "actuals": actuals.tolist(),
            "predicted": predicted,
            "periods_tested": test_periods,
        }

    except ValidationError:
        raise
    except Exception as e:
        logger.error("backtest_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))