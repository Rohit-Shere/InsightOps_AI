
import json
from pathlib import Path
from datetime import date, datetime

from langgraph.graph.message import add_messages

from app.agents.state import create_initial_state
from app.agents.supervisor import supervisor_node
from app.agents.business_analyst import business_analyst_node
from app.agents.statistical_analyst import statistical_analyst_node


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

USER_QUERY = "Why are customers giving low ratings?"

OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "InsightOps_stats_testing.json"
)

# These are the only agents we want to execute.
ALLOWED_AGENTS = {
    "business_analyst",
    "statistical_analyst",
}

# Fields whose LangGraph reducers append new values.
APPEND_FIELDS = {
    "data_caveats",
    "tool_calls",
    "tool_results",
    "agent_outputs",
    "completed_agents",
    "evidence",
    "validation_results",
    "errors",
}


# --------------------------------------------------
# JSON SERIALIZATION
# --------------------------------------------------

def json_safe(value):
    """Convert state objects into JSON-serializable values."""

    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())

    if hasattr(value, "dict") and callable(value.dict):
        return json_safe(value.dict())

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if value is None or isinstance(
        value, (str, int, float, bool)
    ):
        return value

    return str(value)


def save_state(state):
    """Save the current execution state."""

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            json_safe(state),
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nState saved: {OUTPUT_PATH}")


# --------------------------------------------------
# STATE MERGING
# --------------------------------------------------

def merge_node_update(state, update):
    """
    Apply the same reducers declared in InsightOpsState.

    Nodes return only newly generated values for
    reducer-enabled fields.
    """

    for key, value in update.items():

        if key == "messages":
            state[key] = add_messages(
                state.get(key, []),
                value,
            )

        elif key in APPEND_FIELDS:
            state[key] = (
                state.get(key, []) + value
            )

        else:
            state[key] = value

    return state


# --------------------------------------------------
# NODE EXECUTION
# --------------------------------------------------

def execute_node(node_name, node_function, state):

    print("\n" + "=" * 70)
    print(f"EXECUTING: {node_name.upper()}")
    print("=" * 70)

    state["current_agent"] = node_name

    try:
        update = node_function(state)

        if not isinstance(update, dict):
            raise TypeError(
                f"{node_name} must return a dictionary."
            )

        state = merge_node_update(state, update)

        save_state(state)

        print(f"\n{node_name} execution finished.")
        print("Status:", state.get("execution_status"))
        print("Tool calls:", len(state.get("tool_calls", [])))

        return state

    except Exception as error:

        state["execution_status"] = "failed"

        state.setdefault("errors", []).append({
            "agent": node_name,
            "error_type": type(error).__name__,
            "error": str(error),
        })

        save_state(state)

        raise


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run_pipeline():

    print("\nStarting InsightOps AI pipeline...")

    # 1. Initialize state
    state = create_initial_state(USER_QUERY)

    save_state(state)

    # 2. Supervisor
    state = execute_node(
        "supervisor",
        supervisor_node,
        state,
    )

    if state.get("needs_clarification"):
        print(
            "\nSupervisor requires clarification:",
            state.get("clarification_question"),
        )
        return state

    if state.get("execution_status") == "failed":
        return state

    # 3. Restrict execution to currently implemented agents
    planned_agents = state.get("active_agents", [])

    execution_agents = [
        agent
        for agent in planned_agents
        if agent in ALLOWED_AGENTS
    ]

    print("\nSelected execution sequence:")
    for index, agent in enumerate(execution_agents, start=1):
        print(f"{index}. {agent}")

    # Do not execute future agents.
    state["active_agents"] = execution_agents

    # 4. Business Analyst
    if "business_analyst" in execution_agents:

        state = execute_node(
            "business_analyst",
            business_analyst_node,
            state,
        )

        if state.get("execution_status") == "failed":
            return state

    # 5. Statistical Analyst
    if "statistical_analyst" in execution_agents:

        state = execute_node(
            "statistical_analyst",
            statistical_analyst_node,
            state,
        )

        if state.get("execution_status") == "failed":
            return state

    # 6. Stop before Evidence Validator and Report Generator.
    state["current_agent"] = "statistical_analyst"
    state["next_agent"] = "evidence_validator"
    state["execution_status"] = "running"

    save_state(state)

    print("\n" + "=" * 70)
    print("PIPELINE PAUSED AFTER STATISTICAL ANALYST")
    print("=" * 70)

    print("Execution status:", state["execution_status"])
    print("Completed agents:", state.get("completed_agents"))
    print("Total tool calls:", len(state.get("tool_calls", [])))
    print("Total tool results:", len(state.get("tool_results", [])))
    print("Errors:", len(state.get("errors", [])))

    return state


if __name__ == "__main__":
    final_state = run_pipeline()
