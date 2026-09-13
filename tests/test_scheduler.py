import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from nexalens.analytics.scheduler import ReportSchedulerService
from nexalens.models.database import ReportScheduleModel, ReportExecutionModel
from nexalens.models.schemas import (
    ReportFormat,
    ReportSchedule,
    ReportScheduleCreate,
    ReportScheduleUpdate,
    UserRole,
)
from nexalens.models.database import UserModel


class TestReportSchedulerService:
    @pytest.fixture
    def service(self):
        return ReportSchedulerService()

    @pytest.fixture
    def mock_session(self):
        return AsyncMock()

    @pytest.fixture
    def sample_schedule(self):
        return ReportScheduleCreate(
            name="Test Report",
            query="What was revenue last month?",
            data_source_ids=[uuid4()],
            cron_expression="0 9 * * MON",
            recipients=["test@example.com"],
            format=ReportFormat.PDF,
            template="default",
        )

    @pytest.fixture
    def mock_user(self):
        return UserModel(
            id=uuid4(),
            email="test@test.com",
            name="Test",
            role=UserRole.ANALYST.value,
            hashed_password="hashed",
        )

    def test_format_currency(self, service):
        assert service._format_currency(1000) == "$1,000.00"
        assert service._format_currency(1500000) == "$1.50M"
        assert service._format_currency(2500000000) == "$2.50B"
        assert service._format_currency(None) == "N/A"

    def test_format_pct(self, service):
        assert service._format_pct(0.15) == "15.00%"
        assert service._format_pct(0.0123) == "1.23%"
        assert service._format_pct(None) == "N/A"

    def test_format_number(self, service):
        assert service._format_number(1000) == "1,000.00"
        assert service._format_number(1500000) == "1.50M"
        assert service._format_number(None) == "N/A"

    @pytest.mark.asyncio
    async def test_create_schedule(self, service, mock_session, sample_schedule, mock_user):
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.refresh = AsyncMock()

        with patch.object(service, "_schedule_job", new=AsyncMock()):
            result = await service.create_schedule(sample_schedule, mock_session, mock_user.id)

        assert isinstance(result, ReportSchedule)
        assert result.name == "Test Report"
        assert result.cron_expression == "0 9 * * MON"
        assert result.format == ReportFormat.PDF

    @pytest.mark.asyncio
    async def test_update_schedule(self, service, mock_session):
        schedule_id = uuid4()
        existing = ReportScheduleModel(
            id=schedule_id,
            name="Old Name",
            query="Old query",
            data_source_ids=[],
            cron_expression="0 9 * * MON",
            recipients=["test@example.com"],
            format="pdf",
            template="default",
            is_active=True,
            created_by=uuid4(),
        )
        mock_session.get = AsyncMock(return_value=existing)
        mock_session.flush = AsyncMock()

        with patch.object(service, "_remove_job", new=AsyncMock()), \
             patch.object(service, "_schedule_job", new=AsyncMock()):
            updates = ReportScheduleUpdate(name="New Name", is_active=False)
            result = await service.update_schedule(schedule_id, updates, mock_session)

        assert result.name == "New Name"
        assert result.is_active is False

    @pytest.mark.asyncio
    async def test_delete_schedule(self, service, mock_session):
        schedule_id = uuid4()
        existing = ReportScheduleModel(
            id=schedule_id,
            name="Test",
            query="Query",
            data_source_ids=[],
            cron_expression="0 9 * * MON",
            recipients=["test@example.com"],
            format="pdf",
            is_active=True,
            created_by=uuid4(),
        )
        mock_session.get = AsyncMock(return_value=existing)
        mock_session.delete = MagicMock()

        with patch.object(service, "_remove_job", new=AsyncMock()):
            result = await service.delete_schedule(schedule_id, mock_session)

        assert result is True
        mock_session.delete.assert_called_once_with(existing)

    @pytest.mark.asyncio
    async def test_get_schedules(self, service, mock_session, mock_user):
        schedules = [
            ReportScheduleModel(
                id=uuid4(),
                name=f"Schedule {i}",
                query="Query",
                data_source_ids=[],
                cron_expression="0 9 * * MON",
                recipients=["test@example.com"],
                format="pdf",
                is_active=True,
                created_by=mock_user.id,
            )
            for i in range(3)
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = schedules
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await service.get_schedules(mock_session, mock_user.id)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_render_csv(self, service):
        from nexalens.models.schemas import QueryResponse, SQLResult

        response = QueryResponse(
            question="Test",
            intent="sql",
            answer="Answer",
            confidence=0.9,
            processing_time_ms=100,
            sql_result=SQLResult(
                sql="SELECT 1",
                columns=["col1", "col2"],
                rows=[{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}],
                row_count=2,
                execution_time_ms=50,
            ),
        )

        csv = service._render_csv(response)
        assert "col1,col2" in csv
        assert "1,a" in csv
        assert "2,b" in csv

    @pytest.mark.asyncio
    async def test_render_excel(self, service):
        from nexalens.models.schemas import QueryResponse, SQLResult

        response = QueryResponse(
            question="Test",
            intent="sql",
            answer="Answer",
            confidence=0.9,
            processing_time_ms=100,
            sql_result=SQLResult(
                sql="SELECT 1",
                columns=["col1"],
                rows=[{"col1": 1}, {"col1": 2}],
                row_count=2,
                execution_time_ms=50,
            ),
        )

        excel = service._render_excel(response)
        assert len(excel) > 100