
"""
InsightOps AI - Business Analyst

One domain analyst with dynamically restricted tool access.
"""

import os
import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.agents.state import InsightOpsState
from app.tools import get_tools_by_category

load_dotenv()





# ==================================================
# 1. MODEL
# ==================================================

def get_business_model():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        google_api_key=api_key,
        temperature=0.1,
    )


# ==================================================
# 2. TOOL SELECTION
# ==================================================

ALLOWED_BUSINESS_CATEGORIES = {
    "marketplace",
    "customers",
    "products",
    "sellers",
    "payments",
    "reviews",
}


def select_business_tools(categories: list[str]):

    selected_tools = []
    seen_names = set()

    for category in categories:

        category = category.strip().lower()

        if category not in ALLOWED_BUSINESS_CATEGORIES:
            raise ValueError(
                f"Invalid business tool category: {category}"
            )

        category_tools = get_tools_by_category(category)

        for tool in category_tools:

            if tool.name not in seen_names:
                selected_tools.append(tool)
                seen_names.add(tool.name)

    if not selected_tools:
        raise ValueError(
            "No business tools were selected for this task."
        )

    return selected_tools


# ==================================================
# 3. SYSTEM PROMPT
# ==================================================

BUSINESS_ANALYST_PROMPT = """
You are the Business Analyst of InsightOps AI.

Your responsibility is to investigate business questions
using the analytical tools provided to you.

You do NOT have unrestricted database access.

RULES:

1. Use the available tools to retrieve actual business metrics.
2. Never invent numerical results.
3. Do not assume that a correlation establishes causation.
4. Select tools based on the assigned objective.
5. Avoid unnecessary tool calls.
6. Compare relevant metrics when investigating business patterns.
7. Respect the grain and definitions of each analytical dataset.
8. Mention missing data and important limitations.
9. Do not claim that a hypothesis is proven without evidence.
10. Do not perform statistical tests yourself when a
    Statistical Analyst is assigned.

OUTPUT:

Return structured findings in your final response:

- Business question
- Analytical work performed
- Key findings
- Supporting metrics
- Potential explanations
- Data limitations
- Further statistical investigation required

Your output will be passed to another agent for validation.
"""


# ==================================================
# 4. JSON-SAFE CONVERSION
# ==================================================

def make_json_safe(value):

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def parse_tool_content(content):

    if not isinstance(content, str):
        return make_json_safe(content)

    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content


# ==================================================
# 5. GET CURRENT TASK
# ==================================================

def get_business_task(state: InsightOpsState):

    current_task = state.get("current_task", {})

    if current_task.get("agent") == "business_analyst":
        return current_task

    for step in state.get("execution_plan", []):

        if step.get("agent") == "business_analyst":
            return step

    raise ValueError(
        "No Business Analyst task found in execution plan."
    )


# ==================================================
# 6. BUSINESS ANALYST NODE
# ==================================================

def business_analyst_node(state: InsightOpsState) -> dict:

    try:

        task = get_business_task(state)

        objective = task.get("objective", "")
        categories = task.get("tool_categories", [])

        if not objective:
            raise ValueError(
                "Business Analyst objective is missing."
            )

        # Dynamically select only assigned tools.
        selected_tools = select_business_tools(categories)

        model = get_business_model()

        # Create the agent with its restricted toolset.
        agent = create_agent(
            model=model,
            tools=selected_tools,
            system_prompt=BUSINESS_ANALYST_PROMPT,
        )

        prompt = f"""
        USER QUESTION:
        {state.get("user_query", "")}

        ASSIGNED OBJECTIVE:
        {objective}

        RELEVANT DATASETS:
        {task.get("datasets", [])}

        Investigate this objective using your available tools.
        Return findings supported by actual tool results.
        """

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content=prompt)
                ]
            },
            config={
                "recursion_limit": 15
            },
        )

        messages = result.get("messages", [])

        tool_calls = []
        tool_results = []

        for message in result["messages"]:

            # Capture AI-generated tool calls
            if isinstance(message, AIMessage):

                for tc in message.tool_calls:

                    tool_calls.append({
                        "agent": "business_analyst",
                        "tool_name": tc.get("name"),
                        "arguments": tc.get("args", {}),
                        "tool_call_id": tc.get("id")
                    })

            # Capture tool execution results
            elif isinstance(message, ToolMessage):

                tool_results.append({
                    "agent": "business_analyst",
                    "tool_name": message.name,
                    "tool_call_id": message.tool_call_id,
                    "result": message.content
                })

        # Extract final model response.
        final_message = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, AIMessage)
                and message.content
                and not message.tool_calls
            ),
            None,
        )

        if final_message is None:
            raise ValueError(
                "Business Analyst did not produce a final response."
            )

        findings = {
            "agent": "business_analyst",
            "objective": objective,
            "tool_categories": categories,
            "datasets": task.get("datasets", []),
            "tools_used": [
                call["tool_name"]
                for call in tool_calls
            ],
            "findings": make_json_safe(
                final_message.content
            ),
            "tool_call_count": len(tool_calls),
        }

        return {
            "current_agent": "business_analyst",
            "agent_outputs": [findings],
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "completed_agents": ["business_analyst"],
            "execution_status": "running",
        }

    except Exception as exc:

        return {
            "errors": [{
                "agent": "business_analyst",
                "error": str(exc),
            }],
            "execution_status": "failed",
        }
