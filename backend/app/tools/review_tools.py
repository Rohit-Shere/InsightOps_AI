
from langchain.tools import tool

from app.analytics.reviews import (
    get_review_overview,
    get_review_score_distribution,
    get_review_sentiment_analysis,
    get_review_delivery_relationship,
    get_review_monthly_trend,
    get_review_analytics_summary,
)


@tool
def get_review_overview_tool() -> list:
    """
    Retrieve an overall marketplace review summary.

    Includes review records, unique reviews, reviewed orders,
    unique customers, average review score, negative reviews,
    and reviews containing comments.

    Use this for general questions about customer feedback.
    """
    result = get_review_overview()
    return result.to_dict(orient="records")


@tool
def get_review_score_distribution_tool() -> list:
    """
    Retrieve the distribution of customer review scores.

    Returns review counts and percentages for each score.

    Use this to understand the overall rating distribution
    and identify the most common review scores.
    """
    result = get_review_score_distribution()
    return result.to_dict(orient="records")


@tool
def get_review_sentiment_analysis_tool() -> list:
    """
    Retrieve rating-based review sentiment distribution.

    Returns positive, neutral, and negative review counts
    and their percentages.

    Sentiment is derived from review scores:
    1-2 = negative, 3 = neutral, 4-5 = positive.

    This is rating-based sentiment, not NLP analysis of comments.
    """
    result = get_review_sentiment_analysis()
    return result.to_dict(orient="records")


@tool
def get_review_delivery_relationship_tool() -> list:
    """
    Analyze the relationship between delivery performance
    and customer review scores.

    Returns review counts, reviewed orders, average review scores,
    negative review counts, and negative review percentages
    for late, on-time, and unknown delivery groups.

    Use this to investigate whether late deliveries are associated
    with worse customer ratings.

    Association does not establish causation.
    """
    result = get_review_delivery_relationship()
    return result.to_dict(orient="records")


@tool
def get_review_monthly_trend_tool(
    start_date: str = None,
    end_date: str = None
) -> list:
    """
    Retrieve monthly customer review trends.

    Optional dates filter the review creation date.

    Returns monthly review metrics, including average scores
    and negative review percentages.

    Use this to investigate changes in customer satisfaction over time.
    """
    result = get_review_monthly_trend(
        start_date=start_date,
        end_date=end_date
    )
    return result.to_dict(orient="records")


@tool
def get_review_analytics_summary_tool() -> list:
    """
    Retrieve a consolidated customer review analytics summary.

    Includes total review records, unique reviews, average score,
    negative review percentage, and other key review indicators.

    Use this for high-level customer satisfaction questions.
    """
    result = get_review_analytics_summary()
    return result.to_dict(orient="records")


review_tools = [
    get_review_overview_tool,
    get_review_score_distribution_tool,
    get_review_sentiment_analysis_tool,
    get_review_delivery_relationship_tool,
    get_review_monthly_trend_tool,
    get_review_analytics_summary_tool,
]
