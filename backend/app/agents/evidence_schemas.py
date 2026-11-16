from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------
# 1. EVIDENCE REFERENCES
# ---------------------------------------------------------

class EvidenceReference(BaseModel):
    """
    Identifies the exact tool result supporting a claim.
    """

    tool_call_id: str = Field(
        description="ID of the successful tool call supporting this claim."
    )

    tool_name: str = Field(
        description="Name of the tool that produced the evidence."
    )

    supporting_detail: str = Field(
        description="Specific value or finding from the tool result."
    )


# ---------------------------------------------------------
# 2. CLAIM VALIDATION
# ---------------------------------------------------------

class ClaimValidation(BaseModel):
    """
    Validation result for one factual or analytical claim.
    """

    claim: str = Field(
        description="The claim being evaluated."
    )

    status: Literal[
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
    ] = Field(
        description="How well the available evidence supports the claim."
    )

    evidence: list[EvidenceReference] = Field(
        default_factory=list,
        description="Tool results supporting or contradicting the claim."
    )

    explanation: str = Field(
        description="Why the claim received this validation status."
    )

    corrected_claim: str | None = Field(
        default=None,
        description="A corrected evidence-based version, if necessary."
    )


# ---------------------------------------------------------
# 3. OVERALL EVIDENCE VALIDATION
# ---------------------------------------------------------

class EvidenceValidation(BaseModel):
    """
    Complete validation report produced by Evidence Validator.
    """

    validation_status: Literal[
        "passed",
        "passed_with_warnings",
        "failed",
    ] = Field(
        description="Overall validation decision."
    )

    # Individual validation dimensions
    numerical_accuracy: bool = Field(
        description="Whether reported numerical values match successful tool results."
    )

    evidence_traceability: bool = Field(
        description="Whether factual claims can be traced to tool results."
    )

    statistical_validity: bool = Field(
        description="Whether statistical methods and interpretations are valid."
    )

    grain_consistency: bool = Field(
        description="Whether dataset grain and observation units are handled consistently."
    )

    business_logic_validity: bool = Field(
        description="Whether business metrics and their definitions are used correctly."
    )

    causal_language_validity: bool = Field(
        description="Whether causal claims are avoided unless supported by causal evidence."
    )

    # Detailed findings
    validated_claims: list[ClaimValidation] = Field(
        default_factory=list,
        description="Claims that are supported by available evidence."
    )

    rejected_claims: list[ClaimValidation] = Field(
        default_factory=list,
        description="Claims that are unsupported, contradicted, or only partially supported."
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Important limitations and potential concerns."
    )

    corrections: list[str] = Field(
        default_factory=list,
        description="Specific corrections required before report generation."
    )

    missing_evidence: list[str] = Field(
        default_factory=list,
        description="Claims or conclusions that require additional evidence."
    )

    feedback: str = Field(
        description="Concise overall explanation of the validation decision."
    )