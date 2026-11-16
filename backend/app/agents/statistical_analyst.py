
import os
import json

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.agents.state import InsightOpsState
from app.tools import get_tools_by_category


load_dotenv()


# ---------------------------------------------------------
# 1. MODEL
# ---------------------------------------------------------

def get_statistical_model():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        google_api_key=api_key,
        temperature=0.1,
    )


# ---------------------------------------------------------
# 2. TOOL SELECTION
# ---------------------------------------------------------

def select_statistical_tools():

    tools = get_tools_by_category("statistics")

    if not tools:
        raise ValueError("No statistical tools registered.")

    return tools


# ---------------------------------------------------------
# 3. AGENT INSTRUCTIONS
# ---------------------------------------------------------


STATISTICAL_ANALYST_PROMPT = """
You are the Statistical Analyst of InsightOps AI.

Your responsibility is to statistically investigate business
patterns identified by the Business Analyst.

You have access to deterministic statistical tools.
You must use these tools for all calculations.

IMPORTANT TOOL USAGE RULES:

1. METHOD NAMING
- For independent group comparisons, use exactly:
  "mannwhitney" or "welch".
- Never use "mann_whitney", "mann-whitney",
  "mannwhitneyu", or other variations.
- For correlation, use exactly:
  "pearson", "spearman", or "kendall".
- Follow the exact method names supported by each tool.

2. BOOLEAN GROUP ARGUMENTS:

- The statistical tool schemas expect group values as strings.
- For boolean grouping columns, pass exactly "True" or "False".
- Never pass native Python booleans to these tools.
- Example:
    group1_value = "True"
    group2_value = "False"
- Do not change the argument type unless the tool schema
  explicitly supports it.

3. DATASET USAGE
- Use only datasets explicitly provided in the task.
- Inspect dataset metadata before selecting columns.
- Respect the grain and data types of each dataset.
- Do not assume that repeated records are independent.

4. TEST SELECTION
- Use Mann-Whitney or Welch t-test for appropriate
  independent two-group comparisons.
- Use Spearman for ordinal or monotonic relationships
  where appropriate.
- Use confidence intervals to quantify uncertainty.
- Include effect sizes, not just p-values.
- Apply multiple-comparison correction when needed.

5. TOOL EXECUTION
- Never repeatedly call a tool with identical arguments.
- If a tool returns a validation error, inspect the
  error and correct the arguments before retrying.
- Do not repeat an analysis that has already succeeded.
- Complete the investigation using at most 5 tool calls.
- Once sufficient evidence is collected, stop calling tools
  and produce the final report.
- If an analysis cannot be completed, explain why.

6. STATISTICAL INTERPRETATION
- Never invent statistical results.
- Never claim causation from observational data.
- Distinguish statistical significance from practical
  significance.
- If p-value is returned as 0.0 due to numerical
  underflow, report p < 0.001, not p = 0.
- Report sample sizes, test statistics, effect sizes
  and confidence intervals when available.

7. DATA QUALITY
- Consider missing values, duplicate records, small groups,
  unequal group sizes and assumption violations.
- Flag limitations rather than hiding them.

8. OUTPUT
Provide:
- Research question
- Datasets and variables
- Statistical methods
- Sample sizes
- Numerical results
- Effect sizes and confidence intervals
- Assumption checks
- Interpretation
- Limitations
- Evidence-supported conclusions

EVIDENCE AND INTERPRETATION RULES

1. Never report a numerical result unless it appears
   in a successful tool response.

2. Clearly distinguish successful, failed, and
   unperformed analyses.

3. Match the stated alternative hypothesis to the
   alternative parameter actually passed to the test.

4. Never describe observational associations as
   causal effects or proven drivers.

5. Report the dataset grain and discuss repeated
   observations and independence assumptions.

6. Do not infer product, seller, or other causes
   without directly supporting analysis.

7. Do not describe a failed database query as a
   successful statistical analysis.

8. Report p-values of 0.0 due to numerical underflow
   as p < 0.001, not p = 0.

9. Keep review-record-level and order-level results
   clearly separated.
"""


# ---------------------------------------------------------
# 4. SERIALIZATION HELPERS
# ---------------------------------------------------------

def make_json_safe(value):

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def extract_message_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


def parse_tool_content(content):

    if not isinstance(content, str):
        return make_json_safe(content)

    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content


# ---------------------------------------------------------
# 5. TASK RETRIEVAL
# ---------------------------------------------------------

def get_statistical_task(state):

    current_task = state.get("current_task") or {}

    if current_task.get("agent") == "statistical_analyst":
        return current_task

    for step in state.get("execution_plan", []):

        if step.get("agent") == "statistical_analyst":
            return step

    raise ValueError(
        "No Statistical Analyst task found in execution plan."
    )


# ---------------------------------------------------------
# 6. BUSINESS FINDINGS CONTEXT
# ---------------------------------------------------------

def get_business_findings(state):

    findings = []

    for output in state.get("agent_outputs", []):

        if output.get("agent") == "business_analyst":
            findings.append(output)

    return make_json_safe(findings)


# ---------------------------------------------------------
# 7. STATISTICAL ANALYST NODE
# ---------------------------------------------------------

def statistical_analyst_node(
    state: InsightOpsState
) -> dict:

    try:

        # Retrieve task
        task = get_statistical_task(state)

        objective = task.get("objective", "")

        selected_datasets = task.get("datasets", [])

        # Select statistical tools
        selected_tools = select_statistical_tools()

        # Initialize model and agent
        model = get_statistical_model()

        agent = create_agent(
            model=model,
            tools=selected_tools,
            system_prompt=STATISTICAL_ANALYST_PROMPT,
        )

        # Retrieve Business Analyst findings
        business_findings = get_business_findings(state)

        # Construct investigation prompt
        prompt = f"""
USER QUESTION:
{state.get("user_query", "")}

STATISTICAL INVESTIGATION OBJECTIVE:
{objective}

APPROVED DATASETS FOR THIS TASK:
{json.dumps(selected_datasets, indent=2)}

BUSINESS ANALYST FINDINGS:
{json.dumps(business_findings, indent=2, default=str)}

YOUR TASK:

1. Review the Business Analyst findings.
2. Inspect the relevant approved datasets.
3. Choose appropriate statistical tests.
4. Test the important relationships identified.
5. Quantify effect sizes and uncertainty.
6. Check assumptions and data limitations.
7. Independently verify important business claims.

Do not use datasets outside the approved list.

Return a structured, evidence-based statistical report.
"""

        # Execute agent
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content=prompt)
                ]
            },
            config={"recursion_limit": 20},
        )
        

        messages = result.get("messages", [])

        recorded_tool_calls = []
        recorded_tool_results = []

        # Capture tool calls and results
        for message in messages:

            if isinstance(message, AIMessage):

                for tool_call in message.tool_calls:

                    recorded_tool_calls.append({
                        "agent": "statistical_analyst",
                        "tool_name": tool_call.get("name"),
                        "arguments": make_json_safe(
                            tool_call.get("args", {})
                        ),
                        "tool_call_id": tool_call.get("id"),
                    })

            elif isinstance(message, ToolMessage):

                recorded_tool_results.append({
                    "agent": "statistical_analyst",
                    "tool_name": message.name,
                    "tool_call_id": message.tool_call_id,
                    "result": parse_tool_content(message.content),
                })

        # Extract final response
        final_message = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, AIMessage)
                and not message.tool_calls
            ),
            None,
        )

        if final_message is None:
            raise RuntimeError(
                "Statistical Analyst did not produce a final response."
            )

        findings_text = extract_message_text(
            final_message.content
        )

        findings = {
            "agent": "statistical_analyst",
            "objective": objective,
            "datasets": selected_datasets,
            "tools_used": list({
                call["tool_name"]
                for call in recorded_tool_calls
            }),
            "tool_call_count": len(recorded_tool_calls),
            "findings": findings_text,
        }

        return {
            "agent_outputs": [make_json_safe(findings)],
            "tool_calls": recorded_tool_calls,
            "tool_results": recorded_tool_results,
            "completed_agents": ["statistical_analyst"],
            "current_agent": "statistical_analyst",
            "execution_status": "running",
        }

    except Exception as exc:

        return {
            "errors": [
                {
                    "agent": "statistical_analyst",
                    "error": str(exc),
                }
            ],
            "current_agent": "statistical_analyst",
            "execution_status": "failed",
        }
