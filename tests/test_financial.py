import pytest
from uuid import uuid4

from nexalens.analytics.financial import FinancialModelingService
from nexalens.models.schemas import FinancialModelRequest, FinancialModelType


class TestFinancialModelingService:
    @pytest.fixture
    def service(self):
        return FinancialModelingService()

    def test_npv_calculation(self, service):
        cash_flows = [-100000, 30000, 40000, 50000]
        npv = service.calculate_npv(cash_flows, 0.1)
        assert npv > 0

    def test_negative_npv(self, service):
        cash_flows = [-100000, 10000, 10000, 10000]
        npv = service.calculate_npv(cash_flows, 0.1)
        assert npv < 0

    def test_irr_calculation(self, service):
        cash_flows = [-100000, 30000, 40000, 50000]
        irr = service.calculate_irr(cash_flows)
        assert irr is not None
        assert 0 < irr < 1

    def test_payback_calculation(self, service):
        cash_flows = [-100000, 30000, 40000, 50000]
        payback = service.calculate_payback(cash_flows)
        assert payback is not None
        assert 2 < payback < 3

    def test_dcf_calculation(self, service):
        fcf = [100000, 110000, 121000, 133100, 146410]
        result = service.calculate_dcf(fcf, terminal_growth=0.03, discount_rate=0.12)
        assert result["dcf_value"] > 0
        assert result["terminal_value"] > 0

    def test_sensitivity_table(self, service):
        base = {"discount_rate": 0.1, "cash_flows": [-100, 40, 50, 60]}
        ranges = {"discount_rate": (0.08, 0.15)}
        table = service.build_sensitivity_table(base, ["discount_rate"], ranges, steps=4)
        assert "discount_rate" in table
        assert len(table["discount_rate"]) == 4

    def test_run_npv_model(self, service):
        request = FinancialModelRequest(
            model_type=FinancialModelType.NPV,
            assumptions={"discount_rate": 0.1},
            cash_flows=[-100000, 30000, 40000, 50000],
        )
        result = service.run_model(request)
        assert result.model_type == FinancialModelType.NPV
        assert result.npv is not None
        assert result.explanation != ""

    def test_run_irr_model(self, service):
        request = FinancialModelRequest(
            model_type=FinancialModelType.IRR,
            assumptions={},
            cash_flows=[-100000, 30000, 40000, 50000],
        )
        result = service.run_model(request)
        assert result.model_type == FinancialModelType.IRR
        assert result.irr is not None

    def test_run_dcf_model(self, service):
        request = FinancialModelRequest(
            model_type=FinancialModelType.DCF,
            assumptions={"discount_rate": 0.12, "terminal_growth": 0.03},
            cash_flows=[100000, 110000, 121000, 133100, 146410],
        )
        result = service.run_model(request)
        assert result.model_type == FinancialModelType.DCF
        assert result.dcf_value is not None

    def test_run_payback_model(self, service):
        request = FinancialModelRequest(
            model_type=FinancialModelType.PAYBACK,
            assumptions={},
            cash_flows=[-100000, 30000, 40000, 50000],
        )
        result = service.run_model(request)
        assert result.model_type == FinancialModelType.PAYBACK
        assert result.payback_period is not None

    def test_run_sensitivity_model(self, service):
        request = FinancialModelRequest(
            model_type=FinancialModelType.SENSITIVITY,
            assumptions={
                "discount_rate": 0.1,
                "cash_flows": [-100, 40, 50, 60],
                "sensitivity_variables": ["discount_rate"],
                "sensitivity_ranges": {"discount_rate": (0.05, 0.15)},
            },
            cash_flows=[-100, 40, 50, 60],
        )
        result = service.run_model(request)
        assert result.model_type == FinancialModelType.SENSITIVITY
        assert result.sensitivity_table is not None

    def test_excel_export(self, service):
        request = FinancialModelRequest(
            model_type=FinancialModelType.NPV,
            assumptions={"discount_rate": 0.1},
            cash_flows=[-100000, 30000, 40000, 50000],
            output_format="excel",
        )
        result = service.run_model(request)
        assert result.excel_base64 is not None
        assert len(result.excel_base64) > 100

    def test_assumptions_template(self, service):
        for model_type in FinancialModelType:
            template = service.get_assumptions_template(model_type)
            assert isinstance(template, dict)