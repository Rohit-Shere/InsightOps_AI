
from app.database.query_executor import fetch_dataframe


def get_review_overview():
    """
    Overall review metrics.

    Grain: one row per review record.
    """
    query = """
        SELECT
            COUNT(*) AS total_review_records,
            COUNT(DISTINCT review_id) AS unique_reviews,
            COUNT(DISTINCT order_id) AS reviewed_orders,
            COUNT(DISTINCT customer_unique_id) AS unique_customers,

            COUNT(*) - COUNT(DISTINCT review_id)
                AS duplicate_review_id_records,

            COUNT(*) FILTER (
                WHERE review_score IS NULL
            ) AS missing_review_scores,

            COUNT(*) FILTER (
                WHERE review_score IS NOT NULL
                  AND review_score NOT BETWEEN 1 AND 5
            ) AS invalid_review_scores,

            ROUND(AVG(review_score), 2)
                AS average_review_score,

            COUNT(*) FILTER (
                WHERE review_score IN (1, 2)
            ) AS negative_reviews,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE review_score IN (1, 2)
                ) /
                NULLIF(
                    COUNT(*) FILTER (
                        WHERE review_score BETWEEN 1 AND 5
                    ),
                    0
                ),
                2
            ) AS negative_review_percentage,

            COUNT(*) FILTER (
                WHERE NULLIF(TRIM(review_comment_message), '') IS NOT NULL
            ) AS reviews_with_comments,

            COUNT(*) FILTER (
                WHERE NULLIF(TRIM(review_comment_title), '') IS NOT NULL
            ) AS reviews_with_titles

        FROM olist.vw_review_analytics;
    """
    return fetch_dataframe(query)


def get_review_score_distribution():
    """
    Review score distribution.

    Percentages use valid review scores as the denominator.
    """
    query = """
        WITH score_counts AS (
            SELECT
                review_score,
                COUNT(*) AS review_records,
                COUNT(DISTINCT order_id) AS unique_orders
            FROM olist.vw_review_analytics
            WHERE review_score BETWEEN 1 AND 5
            GROUP BY review_score
        )
        SELECT
            review_score,
            review_records,
            unique_orders,

            ROUND(
                100.0 * review_records /
                NULLIF(SUM(review_records) OVER (), 0),
                2
            ) AS review_percentage

        FROM score_counts
        ORDER BY review_score;
    """
    return fetch_dataframe(query)


def get_review_sentiment_analysis():
    """
    Rule-based rating sentiment classification.

    Negative: 1-2
    Neutral: 3
    Positive: 4-5

    This is rating-based sentiment, not NLP analysis of comments.
    """
    query = """
        WITH classified_reviews AS (
            SELECT
                order_id,
                review_score,
                CASE
                    WHEN review_score IN (1, 2) THEN 'Negative'
                    WHEN review_score = 3 THEN 'Neutral'
                    WHEN review_score IN (4, 5) THEN 'Positive'
                    ELSE 'Unclassified'
                END AS sentiment
            FROM olist.vw_review_analytics
        ),
        sentiment_counts AS (
            SELECT
                sentiment,
                COUNT(*) AS review_records,
                COUNT(DISTINCT order_id) AS unique_orders,
                ROUND(AVG(review_score), 2) AS avg_review_score
            FROM classified_reviews
            GROUP BY sentiment
        )
        SELECT
            sentiment,
            review_records,
            unique_orders,
            avg_review_score,

            ROUND(
                100.0 * review_records /
                NULLIF(SUM(review_records) OVER (), 0),
                2
            ) AS review_percentage

        FROM sentiment_counts
        ORDER BY
            CASE sentiment
                WHEN 'Negative' THEN 1
                WHEN 'Neutral' THEN 2
                WHEN 'Positive' THEN 3
                ELSE 4
            END;
    """
    return fetch_dataframe(query)


def get_review_delivery_relationship():
    """
    Compares review scores for late and on-time delivered orders.

    Only delivered orders with a known late-delivery flag
    are classified as late or on-time.
    Other records are shown separately.
    """
    query = """
        WITH delivery_groups AS (
            SELECT
                CASE
                    WHEN order_status = 'delivered'
                         AND is_late_delivery = TRUE
                        THEN 'Late Delivery'

                    WHEN order_status = 'delivered'
                         AND is_late_delivery = FALSE
                        THEN 'On-time Delivery'

                    ELSE 'Not Eligible / Unknown'
                END AS delivery_group,

                order_id,
                review_score

            FROM olist.vw_review_analytics
        ),
        delivery_metrics AS (
            SELECT
                delivery_group,
                COUNT(*) AS review_records,
                COUNT(DISTINCT order_id) AS unique_orders,

                ROUND(AVG(review_score), 2)
                    AS average_review_score,

                COUNT(*) FILTER (
                    WHERE review_score IN (1, 2)
                ) AS negative_reviews,

                ROUND(
                    100.0 * COUNT(*) FILTER (
                        WHERE review_score IN (1, 2)
                    ) /
                    NULLIF(
                        COUNT(*) FILTER (
                            WHERE review_score BETWEEN 1 AND 5
                        ),
                        0
                    ),
                    2
                ) AS negative_review_percentage

            FROM delivery_groups
            GROUP BY delivery_group
        )
        SELECT *
        FROM delivery_metrics
        ORDER BY
            CASE delivery_group
                WHEN 'Late Delivery' THEN 1
                WHEN 'On-time Delivery' THEN 2
                ELSE 3
            END;
    """
    return fetch_dataframe(query)


def get_review_monthly_trend(start_date=None, end_date=None):
    """
    Monthly review trend based on review creation date.
    """
    _validate_date_range(start_date, end_date)

    query = """
        SELECT
            DATE_TRUNC(
                'month',
                review_creation_date
            )::date AS review_month,

            COUNT(*) AS total_review_records,

            COUNT(DISTINCT review_id) AS unique_reviews,

            COUNT(DISTINCT order_id) AS reviewed_orders,

            ROUND(AVG(review_score), 2)
                AS average_review_score,

            COUNT(*) FILTER (
                WHERE review_score IN (1, 2)
            ) AS negative_reviews,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE review_score IN (1, 2)
                ) /
                NULLIF(
                    COUNT(*) FILTER (
                        WHERE review_score BETWEEN 1 AND 5
                    ),
                    0
                ),
                2
            ) AS negative_review_percentage

        FROM olist.vw_review_analytics

        WHERE review_creation_date IS NOT NULL
          AND (
              CAST(:start_date AS date) IS NULL
              OR review_creation_date >= CAST(:start_date AS date)
          )
          AND (
              CAST(:end_date AS date) IS NULL
              OR review_creation_date < CAST(:end_date AS date) + INTERVAL '1 day'
          )

        GROUP BY DATE_TRUNC(
            'month',
            review_creation_date
        )

        ORDER BY review_month;
    """
    return fetch_dataframe(
        query,
        {
            "start_date": start_date,
            "end_date": end_date
        }
    )


def get_review_analytics_summary():
    """
    Consolidated review analytics summary.
    """
    query = """
        WITH review_metrics AS (
            SELECT
                COUNT(*) AS total_review_records,
                COUNT(DISTINCT review_id) AS unique_reviews,
                COUNT(DISTINCT order_id) AS reviewed_orders,

                ROUND(AVG(review_score), 2)
                    AS average_review_score,

                COUNT(*) FILTER (
                    WHERE review_score IN (1, 2)
                ) AS negative_reviews,

                ROUND(
                    100.0 * COUNT(*) FILTER (
                        WHERE review_score IN (1, 2)
                    ) /
                    NULLIF(
                        COUNT(*) FILTER (
                            WHERE review_score BETWEEN 1 AND 5
                        ),
                        0
                    ),
                    2
                ) AS negative_review_percentage

            FROM olist.vw_review_analytics
        ),

        score_ranking AS (
            SELECT
                review_score,
                COUNT(*) AS score_count,
                ROW_NUMBER() OVER (
                    ORDER BY COUNT(*) DESC, review_score DESC
                ) AS score_rank
            FROM olist.vw_review_analytics
            WHERE review_score BETWEEN 1 AND 5
            GROUP BY review_score
        )

        SELECT
            r.*,
            s.review_score AS most_common_review_score,
            s.score_count AS most_common_score_records

        FROM review_metrics r
        LEFT JOIN score_ranking s
            ON s.score_rank = 1;
    """
    return fetch_dataframe(query)


def _validate_date_range(start_date=None, end_date=None):
    """
    Validate optional date parameters.
    """
    if start_date is not None and end_date is not None:
        if str(start_date) > str(end_date):
            raise ValueError("start_date cannot be after end_date")
