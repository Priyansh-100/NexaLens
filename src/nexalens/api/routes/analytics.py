import re
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
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
from nexalens.models.database import DataSourceModel
from nexalens.sql.connection_manager import datasource_connection_manager
from nexalens.sql.schema import schema_service
from nexalens.sql.service import sql_service
from nexalens.analytics.financial import financial_modeling_service
from nexalens.analytics.forecasting import forecasting_service
import pandas as pd

router = APIRouter(prefix="/analytics", tags=["analytics"])
logger = get_logger(__name__)

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(name: str, field: str) -> str:
    if not IDENTIFIER_RE.fullmatch(name):
        raise ValidationError(f"Invalid {field} identifier: {name}")
    return name


async def get_session_for_datasource(
    session: AsyncSession,
    datasource_id: UUID,
):
    """Get the appropriate session for a data source (external or app)."""
    result = await session.execute(select(DataSourceModel).where(DataSourceModel.id == datasource_id))
    data_source = result.scalar_one_or_none()
    if not data_source:
        raise ValidationError(f"Data source {datasource_id} not found")

    if data_source.type == "sql" and data_source.config.get("host"):
        # External data source
        from nexalens.sql.connection_manager import datasource_connection_manager
        async with datasource_connection_manager.get_session(data_source) as ext_session:
            yield ext_session
    else:
        # Application database
        yield session


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


async def _run_forecast_on_session(
    request: ForecastRequest,
    ext_session: AsyncSession,
    datasource_id: UUID,
) -> ForecastResult:
    """Run forecast on a specific session."""
    # Validate identifiers
    table = validate_identifier(request.table_name, "table")
    metric_col = validate_identifier(request.metric_column, "metric column")
    date_col = validate_identifier(request.date_column, "date column")

    # Verify table exists in schema
    schema = await schema_service.get_schema(ext_session, datasource_id)
    if table not in schema:
        raise ValidationError(f"Table '{table}' not found in data source schema")

    # Verify columns exist
    table_columns = {col["name"] for col in schema[table]}
    if metric_col not in table_columns:
        raise ValidationError(f"Column '{metric_col}' not found in table '{table}'")
    if date_col not in table_columns:
        raise ValidationError(f"Column '{date_col}' not found in table '{table}'")

    sql = f"SELECT {date_col}, {metric_col} FROM {table} ORDER BY {date_col}"
    sql_result = await sql_service.execute_sql(ext_session, sql)

    if sql_result.row_count < 3:
        raise ValidationError(f"Insufficient data: {sql_result.row_count} rows (minimum 3)")

    df = pd.DataFrame(sql_result.rows)

    result = forecasting_service.run_forecast(df, request)
    return result


@router.post("/forecast", response_model=ForecastResult)
async def run_forecast(
    request: ForecastRequest,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> ForecastResult:
    try:
        if not request.data_source_ids:
            raise ValidationError("data_source_ids required for forecasting")

        datasource_id = request.data_source_ids[0]

        async for ext_session in get_session_for_datasource(session, datasource_id):
            return await _run_forecast_on_session(request, ext_session, datasource_id)

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
        if not request.data_source_ids:
            raise ValidationError("data_source_ids required")

        datasource_id = request.data_source_ids[0]

        async for ext_session in get_session_for_datasource(session, datasource_id):
            table = validate_identifier(request.table_name, "table")
            metric_col = validate_identifier(request.metric_column, "metric column")
            date_col = validate_identifier(request.date_column, "date column")

            schema = await schema_service.get_schema(ext_session, datasource_id)
            if table not in schema:
                raise ValidationError(f"Table '{table}' not found in data source schema")

            sql = f"SELECT {date_col}, {metric_col} FROM {table} ORDER BY {date_col}"
            sql_result = await sql_service.execute_sql(ext_session, sql)

            if sql_result.row_count < test_periods + 3:
                raise ValidationError("Insufficient data for backtesting")

            df = pd.DataFrame(sql_result.rows)

            from nexalens.analytics.forecasting import ForecastingService
            fs = ForecastingService()
            prepared = fs.prepare_timeseries(df, date_col, metric_col, request.frequency)

            train_df = prepared.iloc[:-test_periods]
            test_df = prepared.iloc[-test_periods:]

            fc_request = ForecastRequest(
                table_name=table,
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
            mask = actuals != 0
            mape = float(np.mean(np.abs((np.array(actuals)[mask] - np.array(predicted)[mask]) / np.array(actuals)[mask])) * 100) if mask.any() else 0.0

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