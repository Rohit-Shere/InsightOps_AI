
import os
import json
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.state import InsightOpsState
from app.agents.report_schemas import BusinessReport


load_dotenv()


# =========================================================
# 1. MODEL
# =========================================================

def get_report_generator_model():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        google_api_key=api_key,
        temperature=0.2,
    )


# =========================================================
# 2. SERIALIZATION HELPERS
# =========================================================

def make_json_safe(value: Any):

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_safe(item)
            for item in value
        ]

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def model_to_dict(value: Any) -> dict:

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if isinstance(value, dict):
        return value

    raise TypeError(
        f"Expected dict or Pydantic model, got {type(value).__name__}"
    )


def normalize_payload(payload):

    if not isinstance(payload, str):
        return payload

    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return payload


def is_successful_result(result):

    if result is None:
        return False

    result = normalize_payload(result)

    if isinstance(result, dict):

        if result.get("success") is False:
            return False

        if result.get("error") or result.get("error_type"):
            return False

        analysis = result.get("analysis")

        if isinstance(analysis, dict):
            if analysis.get("success") is False:
                return False

        return bool(result)

    if isinstance(result, (list, str)):
        return len(result) > 0

    return False


# =========================================================
# 3. PREPARE VALIDATED REPORT INPUT
# =========================================================

def prepare_report_context(state: InsightOpsState) -> dict:
    """
    Extract only evidence approved by the Evidence Validator.

    Supported claims:
        Passed to the report generator.

    Partially supported claims:
        Passed only through their corrected_claim.

    Unsupported / contradicted claims:
        Excluded from report findings.

    Only successful tool results referenced by accepted claims
    are included as source evidence.
    """

    validation_results = state.get("validation_results", [])

    if not validation_results:
        raise ValueError(
            "No Evidence Validator result found in state."
        )

    validation_data = model_to_dict(validation_results[-1])

    validation_status = validation_data.get(
        "validation_status",
        state.get("validation_status"),
    )

    if validation_status not in [
        "passed",
        "passed_with_warnings",
    ]:
        raise ValueError(
            f"Report generation blocked. "
            f"Validation status: {validation_status}"
        )

    validated_claims = validation_data.get(
        "validated_claims", []
    )

    accepted_claims = []
    allowed_call_ids = set()

    for original_claim in validated_claims:

        claim = model_to_dict(original_claim)

        status = claim.get("status")
        evidence = claim.get("evidence", [])

        # Fully supported claims
        if status == "supported":

            report_claim = {
                "claim": claim.get("claim"),
                "status": "supported",
                "evidence": evidence,
            }

        # Only use corrected wording for partial claims
        elif status == "partially_supported":

            corrected_claim = claim.get("corrected_claim")

            if not corrected_claim:
                continue

            report_claim = {
                "claim": corrected_claim,
                "status": "corrected_from_partially_supported",
                "evidence": evidence,
                "original_claim": claim.get("claim"),
            }

        # Never pass unsupported or contradicted claims
        else:
            continue

        accepted_claims.append(report_claim)

        for reference in evidence:

            reference = model_to_dict(reference)

            call_id = reference.get("tool_call_id")

            if call_id:
                allowed_call_ids.add(call_id)

    # -----------------------------------------------------
    # Retrieve only successful source results
    # referenced by accepted claims.
    # -----------------------------------------------------

    source_evidence = []
    verified_call_ids = set()

    for entry in state.get("tool_results", []):

        call_id = entry.get("tool_call_id")

        if call_id not in allowed_call_ids:
            continue

        result = normalize_payload(entry.get("result"))

        if not is_successful_result(result):
            continue

        source_evidence.append({
            "tool_call_id": call_id,
            "tool_name": entry.get("tool_name"),
            "agent": entry.get("agent"),
            "result": make_json_safe(result),
        })

        verified_call_ids.add(call_id)

    # -----------------------------------------------------
    # Remove claims whose evidence is no longer available
    # in the actual successful tool results.
    # -----------------------------------------------------

    final_claims = []

    for claim in accepted_claims:

        valid_evidence = []

        for reference in claim["evidence"]:

            reference = model_to_dict(reference)

            if reference.get("tool_call_id") in verified_call_ids:
                valid_evidence.append(reference)

        if not valid_evidence:
            continue

        claim["evidence"] = valid_evidence
        final_claims.append(claim)

    if not final_claims:
        raise ValueError(
            "No validated claims with available successful "
            "tool evidence. Report generation blocked."
        )

    # Rejected claims are included only as exclusions.
    # They must never become report findings.
    excluded_claims = []

    for original_claim in validation_data.get(
        "rejected_claims", []
    ):

        claim = model_to_dict(original_claim)

        excluded_claims.append({
            "claim": claim.get("claim"),
            "status": claim.get("status"),
            "explanation": claim.get("explanation"),
        })

    return {
        "user_query": state.get("user_query"),

        "validation_status": validation_status,

        "validated_claims": make_json_safe(final_claims),

        "source_evidence": source_evidence,

        "excluded_claims": make_json_safe(excluded_claims),

        "warnings": make_json_safe(
            validation_data.get("warnings", [])
        ),

        "corrections": make_json_safe(
            validation_data.get("corrections", [])
        ),

        "missing_evidence": make_json_safe(
            validation_data.get("missing_evidence", [])
        ),

        "validation_feedback": validation_data.get(
            "feedback", ""
        ),
    }


# =========================================================
# 4. REPORT GENERATOR INSTRUCTIONS
# =========================================================


REPORT_GENERATOR_PROMPT = """
You are the Report Generator of InsightOps AI.

Your responsibility is to convert validated business and
statistical evidence into a professional business intelligence
report.

You are the final content-generation agent.

You do NOT perform analytics.
You do NOT execute SQL or Python.
You do NOT call statistical tools.
You do NOT independently verify or recalculate metrics.

You must use only the validated claims and source evidence
provided in the input.

=========================================================
1. EXECUTIVE SUMMARY
=========================================================

- Directly answer the user's original question.
- Highlight the most important findings.
- Keep it concise and business-oriented.
- Do not introduce unsupported explanations.
- Clearly distinguish observed associations from causal
  explanations.
- Do not overstate what the analysis establishes.

=========================================================
2. KEY FINDINGS
=========================================================

- Use only validated claims.
- Preserve their numerical values exactly.
- Explain findings in clear business language.
- Include relevant evidence references.
- Do not mix different dataset grains.
- Distinguish percentages, proportions, and absolute counts.
- Use neutral, descriptive titles.
- Do not use titles that imply causality or an unsupported
  explanation.

=========================================================
3. STATISTICAL EVIDENCE
=========================================================

- Explain statistical findings in understandable language.
- Preserve test names, statistics, p-values, effect sizes,
  sample sizes, and confidence intervals.
- A p-value represented as 0.0 due to numerical underflow
  must be described as p < 0.001.
- Statistical significance does not automatically mean
  practical significance.
- Do not claim causality from observational analyses.
- Do not describe failed or unperformed analyses as results.
- Clearly identify the analytical unit and dataset grain.
- Do not describe review records as independent orders.
- Interpret statistical tests according to what they establish.

For Mann-Whitney U tests:
- Describe differences between group distributions.
- Do not claim that every observation in one group is lower
  than every observation in the other group.
- Cliff's delta describes pairwise stochastic dominance.
- Do not interpret Cliff's delta as a difference in rating
  points or as an individual-level outcome.

For confidence intervals:
- Describe observational group differences as estimated
  differences in group means.
- Avoid causal terminology such as "penalty", "impact",
  or "effect on customers" when causal identification
  has not been established.
- State the confidence interval with its correct direction
  and group comparison.

=========================================================
4. BUSINESS IMPLICATIONS
=========================================================

Explain what the validated findings may mean for the business.

Clearly distinguish:
- Observed facts.
- Reasonable interpretations.
- Hypotheses requiring further investigation.

Rules:
- Never present a hypothesis as a proven cause.
- Do not infer the reason behind an observation unless
  supporting evidence establishes it.
- Do not use observational associations to make
  counterfactual claims.
- Do not claim that eliminating an observed condition
  would necessarily produce a particular business outcome.
- Do not describe an observed association as a confirmed
  root cause or primary driver.

For example:
- Prefer "Late deliveries are associated with lower
  review scores."
- Avoid "Late deliveries cause lower review scores."
- Prefer "On-time deliveries account for the majority
  of negative reviews in absolute volume."
- Avoid assuming that these reviews must be caused by
  product or non-logistical problems.

=========================================================
5. RECOMMENDATIONS
=========================================================

Provide practical, prioritized recommendations.

Each recommendation must:
- Be relevant to the user's business question.
- Have a rationale grounded in validated evidence.
- Include supporting evidence references.
- Describe expected outcomes as potential outcomes,
  not guaranteed results.

Recommendations may suggest:
- Further investigation.
- Data-quality audits.
- Operational monitoring.
- Customer feedback analysis.
- Controlled experiments.
- Additional data collection.

Rules:
- Clearly distinguish observed evidence from the proposed
  action and its potential expected outcome.
- Do not claim that an untested intervention will
  definitely improve business performance.
- Do not claim that a proposed investigation will
  necessarily identify a particular root cause.
- Do not recommend a specific operational remedy unless
  the evidence supports its relevance.
- Prioritize investigations when the underlying cause
  is not yet established.

=========================================================
6. LIMITATIONS
=========================================================

Include relevant warnings, such as:

- Dataset grain and repeated observations.
- Missing data.
- Statistical assumptions not verified.
- Failed analyses.
- Inconsistent metrics.
- Observational design and lack of causal identification.
- Missing or unavailable variables.
- Unverified classifications.

Rules:
- Do not omit important validator warnings.
- Do not hide limitations to make the report sound stronger.
- Do not speculate about missing data classifications.
- Clearly state when an analysis could not be completed.
- Do not present unavailable information as an established
  fact.

=========================================================
7. EVIDENCE REFERENCES
=========================================================

- Use only the exact tool_call_id values supplied.
- Never invent a reference.
- Every key finding must have supporting evidence.
- Statistical results must have their corresponding
  statistical tool references.
- Recommendations should reference the findings that
  motivate them.
- Do not cite excluded or rejected claims as evidence.
- Do not attach unrelated evidence references merely
  to increase the number of citations.
- Ensure references support the specific finding or
  recommendation to which they are attached.

=========================================================
8. DATA GRAIN AND DENOMINATOR RULES
=========================================================

- Preserve the grain of every dataset used.
- review_delivery represents review-record-level data.
- Statistical sample sizes must refer to the actual
  observations used in the test.
- Do not describe review records as unique orders.
- Do not assume that repeated observations are independent
  unless this has been established.
- Distinguish the negative review rate within a group
  from that group's share of all negative reviews.

For example:
- "62.41% of late-delivery review records are negative"
  is a within-group rate.
- "57.26% of all negative review records are from
  on-time deliveries" is an absolute-volume share.

These are different metrics and must never be confused.

For unclassified delivery categories:
- The "Not Eligible / Unknown" category must not be
  described as canceled, lost, unfulfilled, or undelivered
  unless specific evidence establishes that classification.
- State that the operational reason for missing or
  incomplete delivery milestones is unknown.
- Recommendations may propose investigating the
  classification or data-quality issue.

=========================================================
9. STRICT LANGUAGE AND INTERPRETATION RULES
=========================================================

Avoid causal or unsupported explanatory expressions such as:
- causes
- stems from
- explains
- drives
- results in
- primary driver
- root cause
- penalty
- impact
- eliminates dissatisfaction

Unless causal evidence explicitly supports the statement.

Prefer neutral expressions such as:
- is associated with
- is observed among
- accounts for
- is more prevalent in
- is statistically different
- is consistent with
- warrants further investigation

Additional rules:
- Never convert correlation into causation.
- Never use statistical significance as proof of causality.
- Never interpret effect sizes beyond their statistical meaning.
- Never make individual-level claims from group-level results.
- Never make unsupported counterfactual predictions.
- Never assume that an observed association explains
  all instances of an outcome.
- Never reintroduce rejected or unsupported claims.
- Never modify validated numerical values.
- Never perform new calculations.
- Never invent missing values.
- Never use information outside the supplied evidence.
- Never present recommendations as proven outcomes.
- Preserve uncertainty consistently throughout the report,
  including the executive summary, business implications,
  recommendations, and conclusion.

=========================================================
10. FINAL OUTPUT REQUIREMENTS
=========================================================

- Return a complete BusinessReport matching the supplied
  structured output schema.
- Include all required fields.
- Keep the report professional, concise, and actionable.
- Avoid unnecessary repetition across sections.
- Ensure the conclusion summarizes validated findings
  without introducing new claims.
- Do not include commentary outside the structured report.

Return only the requested structured BusinessReport.
"""


# =========================================================
# 5. GENERATE STRUCTURED REPORT
# =========================================================

def generate_business_report(context: dict) -> dict:

    model = get_report_generator_model()

    structured_model = model.with_structured_output(
        BusinessReport
    )

    context_json = json.dumps(
        make_json_safe(context),
        ensure_ascii=False,
        indent=2,
    )

    prompt = (
        REPORT_GENERATOR_PROMPT
        + "\n\n"
        + "=========================================================\n"
        + "VALIDATED INVESTIGATION DATA\n"
        + "=========================================================\n"
        + context_json
    )

    response = structured_model.invoke(prompt)

    return model_to_dict(response)


# =========================================================
# 6. VERIFY GENERATED REPORT REFERENCES
# =========================================================

def verify_report_references(
    report_data: dict,
    allowed_call_ids: set,
) -> tuple[dict, list]:

    report_data = dict(report_data)
    invalid_references = []

    sections = [
        "key_findings",
        "statistical_evidence",
        "recommendations",
    ]

    for section in sections:

        verified_items = []

        for original_item in report_data.get(section, []):

            item = model_to_dict(original_item)

            valid_references = []

            for call_id in item.get(
                "evidence_references", []
            ):

                if call_id in allowed_call_ids:
                    valid_references.append(call_id)

                else:
                    invalid_references.append({
                        "section": section,
                        "item": item.get(
                            "title",
                            item.get(
                                "analysis",
                                item.get("recommendation"),
                            ),
                        ),
                        "invalid_reference": call_id,
                    })

            item["evidence_references"] = valid_references

            # A finding without evidence must not be published.
            if section in [
                "key_findings",
                "statistical_evidence",
            ] and not valid_references:

                invalid_references.append({
                    "section": section,
                    "item": item,
                    "reason": "No valid evidence references.",
                })

                continue

            # Recommendations without evidence should also
            # be excluded rather than presented as grounded.
            if section == "recommendations" and not valid_references:

                invalid_references.append({
                    "section": section,
                    "item": item,
                    "reason": "Recommendation has no valid evidence.",
                })

                continue

            verified_items.append(item)

        report_data[section] = verified_items

    return report_data, invalid_references


# =========================================================
# 7. REPORT GENERATOR NODE
# =========================================================

def report_generator_node(
    state: InsightOpsState,
) -> dict:

    try:

        # Step 1: Prepare approved input
        context = prepare_report_context(state)

        allowed_call_ids = {
            entry["tool_call_id"]
            for entry in context["source_evidence"]
        }

        # Step 2: Generate structured report
        report_data = generate_business_report(context)

        # Step 3: Verify all report references
        report_data, invalid_references = (
            verify_report_references(
                report_data=report_data,
                allowed_call_ids=allowed_call_ids,
            )
        )

        if invalid_references:

            report_data["limitations"].append(
                "Some generated report items were excluded "
                "because their evidence references were invalid "
                "or unavailable."
            )

        # Step 4: Ensure report contains findings
        if not report_data.get("key_findings"):

            raise ValueError(
                "Generated report contains no findings "
                "with valid evidence references."
            )

        # Step 5: Serialize final report
        report_json = json.dumps(
            make_json_safe(report_data),
            ensure_ascii=False,
            indent=2,
        )

        return {

            "final_answer": report_json,

            "agent_outputs": [
                {
                    "agent": "report_generator",
                    "output": make_json_safe(report_data),
                    "invalid_references": make_json_safe(
                        invalid_references
                    ),
                }
            ],

            "completed_agents": [
                "report_generator"
            ],

            "current_agent": "report_generator",

            "next_agent": "END",

            "execution_status": "completed",
        }

    except Exception as exc:

        return {

            "errors": [
                {
                    "agent": "report_generator",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            ],

            "current_agent": "report_generator",

            "next_agent": "END",

            "execution_status": "failed",

        }
