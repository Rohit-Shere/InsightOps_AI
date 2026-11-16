
from langgraph.graph import StateGraph, START, END

from app.agents.state import InsightOpsState

from app.agents.supervisor import supervisor_node
from app.agents.business_analyst import business_analyst_node
from app.agents.statistical_analyst import statistical_analyst_node
from app.agents.evidence_validator import evidence_validator_node
from app.agents.report_generator import report_generator_node


# =========================================================
# ROUTING FUNCTIONS
# =========================================================

def route_after_supervisor(state: InsightOpsState) -> str:
    if state.get("execution_status") == "failed":
        return "execution_failure"

    active_agents = set(state.get("active_agents", []))
    completed_agents = set(state.get("completed_agents", []))

    if (
        "business_analyst" in active_agents
        and "business_analyst" not in completed_agents
    ):
        return "business_analyst"

    if (
        "statistical_analyst" in active_agents
        and "statistical_analyst" not in completed_agents
    ):
        return "statistical_analyst"

    if "evidence_validator" in active_agents:
        return "evidence_validator"

    return END


def route_after_business_analyst(state: InsightOpsState) -> str:
    if state.get("execution_status") == "failed":
        return "execution_failure"

    active_agents = set(state.get("active_agents", []))
    completed_agents = set(state.get("completed_agents", []))

    if (
        "statistical_analyst" in active_agents
        and "statistical_analyst" not in completed_agents
    ):
        return "statistical_analyst"

    return "evidence_validator"


def route_after_statistical_analyst(state: InsightOpsState) -> str:
    if state.get("execution_status") == "failed":
        return "execution_failure"

    return "evidence_validator"


def route_after_validator(state: InsightOpsState) -> str:
    if state.get("validation_status") in (
        "passed",
        "passed_with_warnings",
    ):
        return "report_generator"

    return "validation_failure"


# =========================================================
# FAILURE HANDLERS
# =========================================================

def execution_failure_node(state: InsightOpsState) -> dict:
    return {
        "current_agent": state.get("current_agent", "supervisor"),
        "next_agent": "END",
        "execution_status": "failed",
        "final_answer": (
            "Workflow execution stopped because an agent "
            "failed. Review the execution errors before "
            "retrying the analysis."
        ),
    }


def validation_failure_node(state: InsightOpsState) -> dict:
    return {
        "current_agent": "evidence_validator",
        "next_agent": "END",
        "execution_status": "failed",
        "final_answer": (
            "Report generation stopped because evidence "
            "validation did not pass. Please review the "
            "validation results before proceeding."
        ),
    }

# =========================================================
# GRAPH CONSTRUCTION
# =========================================================

def build_workflow():
    """
    Construct and compile the InsightOps AI workflow.
    """

    builder = StateGraph(InsightOpsState)

    # Register agent nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("business_analyst", business_analyst_node)
    builder.add_node("statistical_analyst", statistical_analyst_node)
    builder.add_node("evidence_validator", evidence_validator_node)
    builder.add_node("execution_failure", execution_failure_node)
    builder.add_node("report_generator", report_generator_node)

    # Register failure handler
    builder.add_node("validation_failure", validation_failure_node)

    # Entry point
    builder.add_edge(START, "supervisor")

    # Supervisor routing
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
            {
            "business_analyst": "business_analyst",
            "statistical_analyst": "statistical_analyst",
            "evidence_validator": "evidence_validator",
            "execution_failure": "execution_failure",
            END: END,
        },
    )

    # Business Analyst routing
    builder.add_conditional_edges(
        "business_analyst",
        route_after_business_analyst,
        {
            "statistical_analyst": "statistical_analyst",
            "evidence_validator": "evidence_validator",
            "execution_failure": "execution_failure",
        },
    )

    # Statistical Analyst always proceeds to validation
    builder.add_conditional_edges(
        "statistical_analyst",
        route_after_statistical_analyst,
        {
            "evidence_validator": "evidence_validator",
            "execution_failure": "execution_failure",
        },
    )

    # Evidence Validator routing
    builder.add_conditional_edges(
        "evidence_validator",
        route_after_validator,
        {
            "report_generator": "report_generator",
            "validation_failure": "validation_failure",
        },
    )

    # Terminal nodes
    builder.add_edge("report_generator", END)
    builder.add_edge("execution_failure", END)
    builder.add_edge("validation_failure", END)

    return builder.compile()


# Compiled workflow instance
workflow = build_workflow()
