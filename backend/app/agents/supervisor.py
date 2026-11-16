
"""
InsightOps AI - Supervisor Agent

Responsibilities:
- Understand the business question.
- Select the required analytical agents.
- Select approved tool categories and datasets.
- Generate an ordered execution plan.
- Handle clarification requests.

The Supervisor does NOT execute analytics tools.
"""

import os
import json
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage

from app.agents.state import InsightOpsState

load_dotenv()


# ==================================================
# 1. APPROVED OPTIONS
# ==================================================

AnalystName = Literal[
    "business_analyst",
    "statistical_analyst",
]

ToolCategory = Literal[
    "marketplace",
    "customers",
    "products",
    "sellers",
    "payments",
    "reviews",
    "statistics",
]

DatasetName = Literal[
    "review_delivery",
    "order_performance",
    "customer_rfm",
    "product_performance",
    "seller_performance",
    "payment_performance",
]


# ==================================================
# 2. STRUCTURED EXECUTION PLAN
# ==================================================

class AnalysisStep(BaseModel):

    agent: AnalystName = Field(
        description="Analytical agent responsible for this step."
    )

    objective: str = Field(
        description="Specific objective assigned to the agent."
    )

    tool_categories: list[ToolCategory] = Field(
        description="Relevant approved tool categories."
    )

    datasets: list[DatasetName] = Field(
        description="Relevant approved datasets."
    )


class SupervisorDecision(BaseModel):

    intent: str = Field(
        description="Short classification of the user's business question."
    )

    analysis_steps: list[AnalysisStep] = Field(
        description="Ordered analytical steps. Do not include validation or reporting steps."
    )

    requires_clarification: bool = Field(
        description="Whether essential information is missing."
    )

    clarification_question: str = Field(
        description="A concise clarification question, or empty string."
    )

    reasoning_summary: str = Field(
        description="Brief explanation of the selected agents and analysis plan."
    )


# ==================================================
# 3. GEMINI CLIENT
# ==================================================

@lru_cache(maxsize=1)
def get_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing from the backend environment."
        )

    return genai.Client(api_key=api_key)


# ==================================================
# 4. SUPERVISOR INSTRUCTIONS
# ==================================================

SUPERVISOR_INSTRUCTIONS = """
You are the Supervisor of InsightOps AI.

InsightOps AI is a business intelligence system analyzing
the Olist Brazilian E-Commerce dataset.

YOUR RESPONSIBILITIES:
1. Understand the user's question.
2. Identify the business domain.
3. Select the minimum required analytical agents.
4. Select relevant tool categories and datasets.
5. Create an ordered execution plan.

YOU MUST NOT:
- Execute SQL.
- Execute Python analytics.
- Calculate KPIs or statistical tests.
- Invent numerical findings.
- Make conclusions before evidence is collected.
- Select datasets outside the approved list.

AVAILABLE ANALYTICAL AGENTS:

1. business_analyst

Responsible for business-domain investigation.

Available tool categories:
- marketplace
- customers
- products
- sellers
- payments
- reviews

The Business Analyst can investigate multiple domains
within one task.

2. statistical_analyst

Responsible for:
- Descriptive statistics
- Group comparisons
- Hypothesis testing
- Correlation
- Confidence intervals
- Categorical tests
- ANOVA
- Diagnostics
- Post-hoc tests
- Outlier analysis

Its only tool category is statistics.

APPROVED DATASETS:

review_delivery:
Review records, ratings and delivery information.

order_performance:
Order-level marketplace and delivery information.

customer_rfm:
Customer-level RFM information.

product_performance:
Product-level sales and GMV information.

seller_performance:
Seller-level performance information.

payment_performance:
Payment transaction information.

PLANNING RULES:

1. Use the minimum number of agents required.

2. Use Business Analyst for business-domain investigation.

3. Use Statistical Analyst only when statistical validation
   would materially improve the answer.

4. Do not invoke Statistical Analyst for simple retrieval
   questions such as top products or total GMV.

5. For root-cause investigations, explore relevant business
   metrics and patterns before statistical validation.

6. Statistical analysis must use appropriate datasets and
   should not assume that correlation establishes causation.

7. Do not select unrelated tool categories or datasets.

8. If the question can be investigated with available data,
   do not ask unnecessary clarification questions.

9. If clarification is essential, return no analysis steps.

10. Keep every objective specific and actionable.

11. Do not include evidence_validator or report_generator
    in analysis_steps. They are added by the application
    after analytical execution.

12. Do not assume an observed pattern is a proven cause.
"""


# ==================================================
# 5. GEMINI SUPERVISOR DECISION
# ==================================================

def create_supervisor_decision(
    user_query: str
) -> SupervisorDecision:

    client = get_gemini_client()

    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.8-flash"
    )

    response = client.models.generate_content(
        model=model_name,
        contents=user_query,
        config=types.GenerateContentConfig(
            system_instruction=SUPERVISOR_INSTRUCTIONS,
            response_mime_type="application/json",
            response_schema=SupervisorDecision,
            temperature=0.1,
        ),
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty Supervisor response."
        )

    if response.parsed is not None:
        if isinstance(response.parsed, SupervisorDecision):
            return response.parsed

        return SupervisorDecision.model_validate(
            response.parsed
        )

    return SupervisorDecision.model_validate_json(
        response.text
    )


# ==================================================
# 6. APPLICATION-LEVEL PLAN VALIDATION
# ==================================================

BUSINESS_CATEGORIES = {
    "marketplace",
    "customers",
    "products",
    "sellers",
    "payments",
    "reviews",
}


def validate_analysis_steps(steps):

    if not steps:
        raise ValueError(
            "Supervisor produced no analytical steps."
        )

    seen_agents = set()
    previous_position = -1

    for step in steps:

        if step.agent in seen_agents:
            raise ValueError(
                f"Duplicate analytical agent: {step.agent}"
            )

        seen_agents.add(step.agent)

        if step.agent == "business_analyst":

            if not set(step.tool_categories).issubset(
                BUSINESS_CATEGORIES
            ):
                raise ValueError(
                    "Business Analyst received an invalid tool category."
                )

        elif step.agent == "statistical_analyst":

            if set(step.tool_categories) - {"statistics"}:
                raise ValueError(
                    "Statistical Analyst can only use statistics tools."
                )

        position = (
            0 if step.agent == "business_analyst" else 1
        )

        if position < previous_position:
            raise ValueError(
                "Business analysis must precede statistical analysis."
            )

        previous_position = position


# ==================================================
# 7. BUILD FINAL EXECUTION PLAN
# ==================================================

def build_execution_plan(decision):

    analysis_steps = [
        step.model_dump()
        for step in decision.analysis_steps
    ]

    validate_analysis_steps(
        decision.analysis_steps
    )

    # Analytical agents first.
    plan = analysis_steps.copy()

    # Evidence validation is mandatory.
    plan.append({
        "agent": "evidence_validator",
        "objective": (
            "Validate analytical findings, calculations, "
            "data quality and supporting evidence."
        ),
        "tool_categories": [],
        "datasets": [],
    })

    # Reporting happens only after validation.
    plan.append({
        "agent": "report_generator",
        "objective": (
            "Prepare the final business report using "
            "validated findings and evidence."
        ),
        "tool_categories": [],
        "datasets": [],
    })

    return plan


# ==================================================
# 8. LANGGRAPH SUPERVISOR NODE
# ==================================================

def supervisor_node(state: InsightOpsState) -> dict:

    user_query = state.get("user_query", "").strip()

    if not user_query:
        return {
            "execution_status": "failed",
            "errors": [{
                "agent": "supervisor",
                "error": "User query is empty.",
            }],
            "next_agent": "END",
        }

    try:

        decision = create_supervisor_decision(
            user_query
        )

        # Handle clarification.
        if decision.requires_clarification:

            if not decision.clarification_question.strip():
                raise ValueError(
                    "Clarification was requested without a question."
                )

            return {
                "intent": decision.intent,
                "execution_plan": [],
                "active_agents": [],
                "needs_clarification": True,
                "clarification_question": (
                    decision.clarification_question
                ),
                "execution_status": "needs_clarification",
                "next_agent": "END",
                "current_agent": "supervisor",
                "iteration_count": (
                    state.get("iteration_count", 0) + 1
                ),
                "messages": [
                    AIMessage(
                        content=decision.clarification_question
                    )
                ],
            }

        # Construct and validate the plan.
        plan = build_execution_plan(decision)

        # Extract analytical and downstream agents.
        active_agents = [
            step["agent"]
            for step in plan
        ]

        # Collect unique tool categories.
        selected_categories = list(dict.fromkeys(
            category
            for step in plan
            for category in step["tool_categories"]
        ))

        # Collect unique datasets.
        selected_datasets = list(dict.fromkeys(
            dataset
            for step in plan
            for dataset in step["datasets"]
        ))

        if not decision.analysis_steps:
            raise ValueError(
                "No analytical steps were generated."
            )

        first_agent = plan[0]["agent"]

        return {
            "intent": decision.intent,
            "execution_plan": plan,
            "active_agents": active_agents,

            "selected_tool_categories": selected_categories,
            "selected_datasets": selected_datasets,

            "current_agent": "supervisor",
            "current_task": {},

            "next_agent": first_agent,

            "needs_clarification": False,
            "clarification_question": "",

            "execution_status": "planned",

            "iteration_count": (
                state.get("iteration_count", 0) + 1
            ),

            "messages": [
                AIMessage(
                    content=decision.reasoning_summary
                )
            ],
        }

    except Exception as exc:

        return {
            "execution_status": "failed",
            "next_agent": "END",
            "errors": [{
                "agent": "supervisor",
                "error": str(exc),
            }],
        }
