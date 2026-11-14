
from langchain.tools import tool

from app.analytics.payments import (
    get_payment_overview,
    get_payment_method_distribution,
    get_payment_installment_analysis,
    get_payment_value_distribution,
    get_payment_anomalies,
    get_payment_analytics_summary,
)


@tool
def get_payment_overview_tool() -> list:
    """
    Retrieve an overall marketplace payment summary.

    Includes payment records, unique orders, unique customers,
    delivered and canceled orders, total recorded payment value,
    and average payment values.

    Use this for general questions about marketplace payments.
    """
    result = get_payment_overview()
    return result.to_dict(orient="records")


@tool
def get_payment_method_distribution_tool() -> list:
    """
    Analyze marketplace payment methods.

    Returns payment record counts, unique orders, unique customers,
    total payment value, record percentage, and payment value percentage
    for each payment method.

    Use this to understand customer payment preferences and
    payment method contribution.
    """
    result = get_payment_method_distribution()
    return result.to_dict(orient="records")


@tool
def get_payment_installment_analysis_tool() -> list:
    """
    Analyze payment installment patterns.

    Returns installment counts, payment records, unique orders,
    total payment value, average payment value, and record percentage.

    Use this to investigate installment preferences and payment behavior.
    """
    result = get_payment_installment_analysis()
    return result.to_dict(orient="records")


@tool
def get_payment_value_distribution_tool() -> list:
    """
    Analyze the distribution of payment record values across value bands.

    Returns payment record counts, unique orders, total payment value,
    average payment value, and record percentage.

    Use this to understand payment ticket-size distribution.

    Note: these are payment-record-level bands, not necessarily
    order-level transaction values.
    """
    result = get_payment_value_distribution()
    return result.to_dict(orient="records")


@tool
def get_payment_anomalies_tool() -> list:
    """
    Retrieve payment data-quality and anomaly indicators.

    Includes zero-value records, unspecified payment types,
    negative payment records, credit-card records with zero installments,
    orders without payment records, and missing payment types.

    Use this when investigating payment data quality or unusual records.

    An anomaly indicator identifies records for investigation;
    it does not automatically mean fraud.
    """
    result = get_payment_anomalies()
    return result.to_dict(orient="records")


@tool
def get_payment_analytics_summary_tool() -> list:
    """
    Retrieve a consolidated payment analytics summary.

    Includes total payment records, unique orders,
    recorded payment value, top payment method,
    and most common installment count.

    Use this for high-level payment performance questions.
    """
    result = get_payment_analytics_summary()
    return result.to_dict(orient="records")


payment_tools = [
    get_payment_overview_tool,
    get_payment_method_distribution_tool,
    get_payment_installment_analysis_tool,
    get_payment_value_distribution_tool,
    get_payment_anomalies_tool,
    get_payment_analytics_summary_tool,
]
