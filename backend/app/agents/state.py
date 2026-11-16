
"""
Shared LangGraph State for InsightOps AI.

Five-agent architecture:
1. Supervisor
2. Business Analyst
3. Statistical Analyst
4. Evidence Validator
5. Report Generator
"""

from typing import Annotated, TypedDict, Any, Literal
from operator import add

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


AgentName = Literal[
    "supervisor",
    "business_analyst",
    "statistical_analyst",
    "evidence_validator",
    "report_generator",
]

ExecutionStatus = Literal[
    "pending",
    "planned",
    "running",
    "needs_clarification",
    "completed",
    "failed",
]


class InsightOpsState(TypedDict, total=False):

    # --------------------------------
    # User input and conversation
    # --------------------------------
    user_query: str
    messages: Annotated[list[AnyMessage], add_messages]

    # --------------------------------
    # Supervisor planning
    # --------------------------------
    intent: str
    execution_plan: list[dict[str, Any]]
    active_agents: list[AgentName]

    # --------------------------------
    # Current execution context
    # --------------------------------
    current_agent: AgentName
    next_agent: str
    current_task: dict[str, Any]

    selected_tool_categories: list[str]
    selected_datasets: list[str]
    data_caveats: Annotated[list[str], add]

    # --------------------------------
    # Tool execution tracking
    # --------------------------------
    tool_calls: Annotated[list[dict[str, Any]], add]
    tool_results: Annotated[list[dict[str, Any]], add]

    # --------------------------------
    # Agent outputs and evidence
    # --------------------------------
    agent_outputs: Annotated[list[dict[str, Any]], add]
    completed_agents: Annotated[list[str], add]

    evidence: Annotated[list[dict[str, Any]], add]

    # --------------------------------
    # Evidence validation
    # --------------------------------
    validation_results: Annotated[list[dict[str, Any]], add]
    validation_status: str
    validation_feedback: str

    # --------------------------------
    # Error and retry management
    # --------------------------------
    errors: Annotated[list[dict[str, Any]], add]
    retry_count: int
    max_retries: int

    iteration_count: int
    max_iterations: int

    # --------------------------------
    # Clarification management
    # --------------------------------
    needs_clarification: bool
    clarification_question: str

    # --------------------------------
    # Final response
    # --------------------------------
    execution_status: ExecutionStatus
    final_answer: str


def create_initial_state(user_query: str) -> dict:

    return {
        "user_query": user_query,
        "messages": [
            HumanMessage(content=user_query)
        ],

        "intent": "",
        "execution_plan": [],
        "active_agents": [],

        "current_agent": "supervisor",
        "next_agent": "",
        "current_task": {},

        "selected_tool_categories": [],
        "selected_datasets": [],
        "data_caveats": [],

        "tool_calls": [],
        "tool_results": [],

        "agent_outputs": [],
        "completed_agents": [],
        "evidence": [],

        "validation_results": [],
        "validation_status": "pending",
        "validation_feedback": "",

        "errors": [],
        "retry_count": 0,
        "max_retries": 2,

        "iteration_count": 0,
        "max_iterations": 12,

        "needs_clarification": False,
        "clarification_question": "",

        "execution_status": "pending",
        "final_answer": "",
    }
