from typing import Literal
from pydantic import BaseModel, Field


class ReportFinding(BaseModel):
    title: str
    finding: str
    evidence_references: list[str] = Field(default_factory=list)


class StatisticalEvidence(BaseModel):
    analysis: str
    result: str
    interpretation: str
    evidence_references: list[str] = Field(default_factory=list)


class BusinessRecommendation(BaseModel):
    priority: Literal["high", "medium", "low"]
    recommendation: str
    rationale: str
    evidence_references: list[str] = Field(default_factory=list)
    expected_outcome: str


class BusinessReport(BaseModel):
    report_title: str

    executive_summary: str

    key_findings: list[ReportFinding]

    statistical_evidence: list[StatisticalEvidence]

    business_implications: list[str]

    recommendations: list[BusinessRecommendation]

    limitations: list[str]

    conclusion: str