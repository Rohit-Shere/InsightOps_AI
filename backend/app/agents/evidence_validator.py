import os
import json
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.state import InsightOpsState
from app.agents.evidence_schemas import EvidenceValidation


load_dotenv()


# =========================================================
# 1. MODEL
# =========================================================

def get_evidence_validator_model():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        google_api_key=api_key,
        temperature=0,
    )


# =========================================================
# 2. SERIALIZATION HELPERS
# =========================================================

def make_json_safe(value: Any):
    """Recursively convert values into JSON-serializable objects."""

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def normalize_tool_payload(payload: Any):
    """
    Parse JSON-encoded tool results where possible.
    Preserve non-JSON strings without attempting unsafe evaluation.
    """

    if not isinstance(payload, str):
        return payload

    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return payload


def is_successful_tool_result(result: Any) -> bool:
    """
    Determine whether a tool result represents a successful execution.

    Supports:
    - Statistical tool dictionaries
    - Business tool dictionaries
    - Lists
    - JSON-encoded strings
    - Non-JSON result strings
    """

    if result is None:
        return False

    result = normalize_tool_payload(result)

    if isinstance(result, dict):

        # Explicit top-level failure
        if result.get("success") is False:
            return False

        # Explicit top-level success
        if result.get("success") is True:
            return True

        # Statistical tools may nest success inside analysis
        analysis = result.get("analysis")

        if isinstance(analysis, dict):

            if analysis.get("success") is False:
                return False

            if analysis.get("success") is True:
                return True

        # Detect error-shaped responses
        if result.get("error") or result.get("error_type"):
            return False

        # Business tools may return data without a success field
        return bool(result)

    if isinstance(result, (list, str)):
        return len(result) > 0

    return False


def model_to_dict(value: Any) -> dict:
    """Support both Pydantic models and regular dictionaries."""

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if isinstance(value, dict):
        return value

    raise TypeError(
        f"Expected a Pydantic model or dict, got {type(value).__name__}"
    )


# =========================================================
# 3. PREPARE EVIDENCE CONTEXT
# =========================================================

def prepare_evidence_context(state: InsightOpsState) -> dict:
    """
    Separate successful and failed tool results.

    A tool call ID is considered valid evidence only if its
    corresponding tool result was successful.
    """

    successful_results = []
    failed_results = []
    successful_call_ids = []
    successful_call_map = {}

    for entry in state.get("tool_results", []):

        if not isinstance(entry, dict):
            continue

        payload = normalize_tool_payload(
            entry.get("result")
        )

        call_id = entry.get("tool_call_id")
        tool_name = entry.get("tool_name")

        normalized_entry = {
            "agent": entry.get("agent"),
            "tool_name": tool_name,
            "tool_call_id": call_id,
            "result": make_json_safe(payload),
        }

        if is_successful_tool_result(payload):

            successful_results.append(normalized_entry)

            if call_id:
                successful_call_ids.append(call_id)
                successful_call_map[call_id] = tool_name

        else:
            failed_results.append(normalized_entry)

    return {
        "user_query": state.get("user_query"),
        "agent_outputs": make_json_safe(
            state.get("agent_outputs", [])
        ),
        "tool_calls": make_json_safe(
            state.get("tool_calls", [])
        ),
        "successful_tool_results": successful_results,
        "failed_tool_results": failed_results,
        "successful_call_ids": successful_call_ids,
        "successful_call_map": successful_call_map,
        "data_caveats": make_json_safe(
            state.get("data_caveats", [])
        ),
        "errors": make_json_safe(
            state.get("errors", [])
        ),
        "execution_plan": make_json_safe(
            state.get("execution_plan", [])
        ),
    }


# =========================================================
# 4. VALIDATOR INSTRUCTIONS
# =========================================================

EVIDENCE_VALIDATOR_PROMPT = """
You are the Evidence Validator of InsightOps AI.

Your responsibility is to independently validate the findings
produced by the Business Analyst and Statistical Analyst.

You are the quality-control layer of the system.

You do NOT perform new analytics.
You do NOT have access to SQL, Python, or statistical tools.
You must validate claims using only the supplied tool results
and existing execution evidence.

=========================================================
1. NUMERICAL ACCURACY
=========================================================

- Verify numerical claims against successful tool results.
- Check sample sizes, means, medians, percentages,
  test statistics, p-values, confidence intervals,
  and effect sizes.
- Do not accept numbers merely because an analyst reported them.
- Do not invent or estimate missing values.
- A p-value of 0.0 caused by numerical underflow must
  be represented as p < 0.001, not p = 0.

=========================================================
2. EVIDENCE TRACEABILITY
=========================================================

- Every important factual or analytical claim must have
  supporting evidence from a successful tool result.
- Use the exact tool_call_id and tool_name.
- Never fabricate evidence references.
- A failed tool result cannot support a successful
  analytical conclusion.
- Distinguish successful, failed, and unperformed analyses.

=========================================================
3. STATISTICAL VALIDITY
=========================================================

Check:

- Whether the selected test matches the research question.
- Whether the alternative hypothesis matches the actual
  test alternative.
- Whether p-values are interpreted correctly.
- Whether confidence intervals are interpreted correctly.
- Whether effect sizes are interpreted correctly.
- Whether statistical significance is confused with
  practical significance.
- Whether sample sizes are reported accurately.
- Whether assumptions and limitations are acknowledged.

Do not claim that an assumption was tested unless
the supplied evidence confirms it.

=========================================================
4. DATASET GRAIN
=========================================================

Check the unit of observation for every analysis.

Examples:

- One row per review record.
- One row per order.
- One row per customer.
- One row per product.

Do not allow review-level and order-level results
to be mixed or presented as interchangeable.

Identify repeated observations and potential
independence problems.

=========================================================
5. BUSINESS LOGIC
=========================================================

Check whether business metrics use appropriate
definitions and denominators.

Examples:

- GMV is not automatically revenue or profit.
- Late delivery rate must use eligible delivered orders.
- Customer counts must use the correct customer identifier.
- Product and seller aggregations can overlap.
- Missing values must not silently become zero.

Flag unsupported metric definitions.

=========================================================
6. CAUSAL LANGUAGE
=========================================================

Observational associations do not establish causality.

Reject unsupported claims containing interpretations such as:

- caused
- resulted in
- impact
- drove
- led to
- responsible for

Prefer:

- associated with
- correlated with
- observed difference
- tends to be lower
- relationship between

Do not infer product quality, seller behavior, payment
issues, or other causes without supporting analysis.

=========================================================
7. CLAIM CLASSIFICATION
=========================================================

Classify claims as:

- supported:
  Directly supported by successful tool evidence.

- partially_supported:
  Some elements are supported, but the claim is
  broader or less precise than the evidence allows.

- unsupported:
  No sufficient evidence is available.

- contradicted:
  Available evidence directly conflicts with the claim.

Provide a corrected_claim wherever useful.

=========================================================
8. OVERALL VALIDATION DECISION
=========================================================

passed:
All critical claims are supported and no material
validation problem remains.

passed_with_warnings:
The core findings are supported, but non-critical
limitations or warnings must accompany the report.

failed:
There are material numerical errors, unsupported
conclusions, invalid statistical interpretations,
or missing critical evidence.

A failed tool call alone does not automatically mean
the entire investigation failed. Evaluate whether the
remaining successful evidence is sufficient.

=========================================================
9. OUTPUT RULES
=========================================================

Return only the requested structured EvidenceValidation.

Include:

- Overall validation status.
- All six validation dimensions.
- Supported claims.
- Rejected or partially supported claims.
- Exact evidence references.
- Warnings.
- Required corrections.
- Missing evidence.
- Concise feedback.

Do not rewrite the entire business report.
Do not perform calculations.
Do not create new statistical results.
Do not treat analyst-generated text as ground truth.

=========================================================
10. IMPORTANT INVESTIGATION-SPECIFIC CHECKS
=========================================================

- Review-level results and order-level results are different
  analyses and must not be mixed.
- A review-level dataset may contain multiple review records
  for the same order or customer.
- A significant association between late delivery and low
  review scores does not establish that delivery delays
  caused the low scores.
- "Not eligible" or "unknown" delivery status does not
  automatically mean canceled, lost, or unfulfilled.
- A failed ANOVA call caused by an invalid grouping column
  is not a successful statistical test.
- Do not claim that an unperformed analysis supports a finding.
- If marketplace late-delivery metrics conflict across tools,
  flag the discrepancy instead of selecting one value.
"""


# =========================================================
# 5. RUN STRUCTURED VALIDATION
# =========================================================

def validate_evidence(context: dict) -> dict:
    """
    Invoke Gemini with structured EvidenceValidation output.
    Returns a regular Python dictionary.
    """

    model = get_evidence_validator_model()

    structured_model = model.with_structured_output(
        EvidenceValidation
    )

    context_json = json.dumps(
        make_json_safe(context),
        ensure_ascii=False,
        indent=2,
    )

    prompt = (
        EVIDENCE_VALIDATOR_PROMPT
        + "\n\n"
        + "=========================================================\n"
        + "INVESTIGATION EVIDENCE\n"
        + "=========================================================\n"
        + context_json
    )

    response = structured_model.invoke(prompt)

    return model_to_dict(response)


# =========================================================
# 6. DETERMINISTIC EVIDENCE REFERENCE VERIFICATION
# =========================================================

def verify_claim_references(
    claims: list,
    successful_call_map: dict,
) -> tuple[list, list]:
    """
    Verify each claim's evidence references.

    Both tool_call_id and tool_name must match a successful
    tool execution.
    """

    invalid_references = []
    verified_claims = []

    for original_claim in claims:

        claim = model_to_dict(original_claim)

        claim_has_invalid_reference = False
        valid_evidence = []

        for original_evidence in claim.get("evidence", []):

            evidence = model_to_dict(original_evidence)

            call_id = evidence.get("tool_call_id")
            cited_tool_name = evidence.get("tool_name")

            actual_tool_name = successful_call_map.get(call_id)

            reference_is_valid = (
                call_id in successful_call_map
                and actual_tool_name == cited_tool_name
            )

            if reference_is_valid:

                valid_evidence.append(evidence)

            else:

                claim_has_invalid_reference = True

                invalid_references.append({
                    "claim": claim.get("claim"),
                    "invalid_tool_call_id": call_id,
                    "cited_tool_name": cited_tool_name,
                    "actual_tool_name": actual_tool_name,
                })

        claim["evidence"] = valid_evidence

        if claim_has_invalid_reference:

            claim["status"] = "unsupported"

            previous_explanation = claim.get(
                "explanation", ""
            )

            claim["explanation"] = (
                previous_explanation
                + " Deterministic evidence verification failed: "
                "one or more cited references did not match "
                "a successful tool call and its tool name."
            ).strip()

            if not claim.get("corrected_claim"):
                claim["corrected_claim"] = (
                    "This claim cannot be accepted until "
                    "valid supporting evidence is provided."
                )

        # A supported claim must have at least one reference.
        if (
            claim.get("status") in [
                "supported",
                "partially_supported",
            ]
            and not valid_evidence
        ):

            claim["status"] = "unsupported"

            claim["explanation"] = (
                claim.get("explanation", "")
                + " No valid evidence reference was supplied."
            ).strip()

        verified_claims.append(claim)

    return verified_claims, invalid_references


def verify_evidence_references(
    validation_data: dict,
    successful_call_map: dict,
) -> dict:
    """
    Deterministically verify evidence references and update
    the overall validation decision.
    """

    validation_data = dict(validation_data)

    # Validate claims presented as supported/partially supported.
    validated_claims, invalid_validated_refs = (
        verify_claim_references(
            validation_data.get("validated_claims", []),
            successful_call_map,
        )
    )

    # Also verify references attached to rejected claims.
    rejected_claims, invalid_rejected_refs = (
        verify_claim_references(
            validation_data.get("rejected_claims", []),
            successful_call_map,
        )
    )

    validation_data["validated_claims"] = validated_claims
    validation_data["rejected_claims"] = rejected_claims

    invalid_references = (
        invalid_validated_refs + invalid_rejected_refs
    )

    warnings = list(validation_data.get("warnings") or [])

    for item in invalid_references:
        warnings.append(
            "Invalid evidence reference: "
            + json.dumps(make_json_safe(item))
        )

    validation_data["warnings"] = list(dict.fromkeys(warnings))

    # Determine whether any claim accepted by the LLM
    # failed deterministic verification.
    accepted_claims_have_critical_failure = any(
        claim.get("status") in ["unsupported", "contradicted"]
        for claim in validated_claims
    )

    has_partial_claims = any(
        claim.get("status") == "partially_supported"
        for claim in validated_claims
    )

    # Preserve the model's failed decision.
    original_status = validation_data.get(
        "validation_status",
        "failed",
    )

    if (
        original_status == "failed"
        or accepted_claims_have_critical_failure
    ):

        final_status = "failed"

    elif has_partial_claims:

        final_status = "passed_with_warnings"

    else:

        # Preserve a warning decision from the LLM.
        final_status = (
            "passed_with_warnings"
            if original_status == "passed_with_warnings"
            else "passed"
        )

    validation_data["validation_status"] = final_status

    # Update feedback after deterministic verification.
    if invalid_references:

        validation_data["feedback"] = (
            "Evidence validation found invalid or mismatched "
            "evidence references. Affected claims were rejected. "
            "The report must not present those claims as verified."
        )

    elif final_status == "failed":

        validation_data["feedback"] = (
            validation_data.get("feedback")
            or "Material evidence or validation problems remain. "
               "Do not proceed to report generation."
        )

    elif final_status == "passed_with_warnings":

        validation_data["feedback"] = (
            validation_data.get("feedback")
            or "Core findings are supported, but the report must "
               "include the listed limitations and warnings."
        )

    else:

        validation_data["feedback"] = (
            validation_data.get("feedback")
            or "Evidence validation passed."
        )

    return validation_data


# =========================================================
# 7. EVIDENCE VALIDATOR NODE
# =========================================================

def evidence_validator_node(
    state: InsightOpsState,
) -> dict:
    """
    LangGraph node:
    1. Prepares evidence.
    2. Runs structured LLM validation.
    3. Verifies references deterministically.
    4. Decides whether report generation can proceed.
    """

    try:

        # -------------------------------------------------
        # Step 1: Prepare evidence context
        # -------------------------------------------------

        context = prepare_evidence_context(state)

        successful_call_map = context["successful_call_map"]

        # -------------------------------------------------
        # Step 2: Run LLM structured validation
        # -------------------------------------------------

        validation_data = validate_evidence(context)

        # -------------------------------------------------
        # Step 3: Verify evidence references
        # -------------------------------------------------

        validation_data = verify_evidence_references(
            validation_data=validation_data,
            successful_call_map=successful_call_map,
        )

        validation_status = validation_data[
            "validation_status"
        ]

        # -------------------------------------------------
        # Step 4: Determine next agent
        # -------------------------------------------------

        if validation_status in [
            "passed",
            "passed_with_warnings",
        ]:

            next_agent = "report_generator"

        else:

            next_agent = "supervisor"

        # -------------------------------------------------
        # Step 5: Construct node output
        # -------------------------------------------------

        feedback = validation_data.get(
            "feedback",
            "Evidence validation completed.",
        )

        return {
            "validation_results": [
                make_json_safe(validation_data)
            ],

            "validation_status": validation_status,

            "validation_feedback": feedback,

            "evidence": make_json_safe(
                validation_data.get("validated_claims", [])
            ),

            "completed_agents": [
                "evidence_validator"
            ],

            "current_agent": "evidence_validator",

            "next_agent": next_agent,

            "execution_status": "running",
        }

    except Exception as exc:

        return {
            "errors": [
                {
                    "agent": "evidence_validator",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            ],

            "validation_status": "failed",

            "validation_feedback": (
                "Evidence validation could not be completed. "
                "The pipeline must not proceed to report generation."
            ),

            "current_agent": "evidence_validator",

            "next_agent": "supervisor",

            "execution_status": "failed",
        }