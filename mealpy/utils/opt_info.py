#!/usr/bin/env python
# Created by "Thieu" at 10:44, 24/07/2026 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Literal


Difficulty = Literal["easy", "medium", "hard", "nightmare"]

AlgorithmKind = Literal["original", "variant", "hybrid", "sota", "developed"]

ScientificStatus = Literal["normal", "questionable", "under_investigation", "withdrawn", "retracted"]


class ScientificConcern(str, Enum):
    """
    Scientific concerns associated with an optimization algorithm.
    """

    SUSPECTED_PLAGIARISM = "suspected_plagiarism"
    SUSPECTED_SELF_PLAGIARISM = "suspected_self_plagiarism"
    LACK_OF_NOVELTY = "lack_of_novelty"
    HIGH_SIMILARITY = "high_similarity_to_existing_algorithms"
    QUESTIONABLE_MATH = "questionable_mathematical_model"
    INCORRECT_EQUATIONS = "incorrect_equations"
    POOR_REPRODUCIBILITY = "poor_reproducibility"
    INSUFFICIENT_VALIDATION = "insufficient_experimental_validation"
    RESEARCH_MISCONDUCT = "research_misconduct"
    FABRICATED_RESULTS = "fabricated_results"
    PUBLIC_INTEGRITY_DISCUSSION = "public_integrity_discussion"
    EXPRESSION_OF_CONCERN = "expression_of_concern"
    AMBIGUOUS_METHODOLOGY = "ambiguous_methodology"
    CODE_PSEUDOCODE_MISMATCH = "code_pseudocode_mismatch"

    def __str__(self) -> str:
        """Return the string value of the enum member."""
        return self.value


@dataclass(frozen=True)
class OptInfo:
    """
    Metadata used to generate the algorithm classification table.

    Parameters
    ----------
    difficulty : {"easy", "medium", "hard", "nightmare"}
        Estimated implementation or conceptual difficulty.
    kind : {"original", "variant", "hybrid", "sota", "developed"}
        The type of algorithm.
    name : str, optional
        Full algorithm name. It is normally defined only by the original class.
    year : int, optional
        Publication year. It is normally defined only by the original class.
    family : str, optional
        Internal family identifier. This is only needed when one Python module
        contains multiple unrelated algorithm families.
    scientific_status : {"normal", "questionable", "under_investigation", "withdrawn", "retracted"}
        Current scientific integrity or publication status.
    concerns : tuple of ScientificConcern, default=()
        Documented concerns regarding originality, novelty, validity, or publication integrity.
    evidence_urls : tuple of str, default=()
        Public sources supporting the assigned scientific status.
    """

    difficulty: Difficulty
    kind: AlgorithmKind
    name: str | None = None
    year: int | None = None
    family: str | None = None

    scientific_status: ScientificStatus = "normal"
    concerns: tuple[ScientificConcern, ...] = ()
    evidence_urls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        valid_difficulties = {"easy", "medium", "hard", "nightmare"}
        valid_kind = {"original", "variant", "hybrid", "sota", "developed"}
        valid_statuses = {"normal", "questionable", "under_investigation", "withdrawn", "retracted"}

        if self.difficulty not in valid_difficulties:
            raise ValueError(
                f"Invalid difficulty: {self.difficulty!r}. "
                f"Expected one of {sorted(valid_difficulties)}."
            )

        if self.kind not in valid_kind:
            raise ValueError(
                f"Invalid kind: {self.kind!r}. "
                f"Expected one of {sorted(valid_kind)}."
            )

        if self.scientific_status not in valid_statuses:
            raise ValueError(
                f"Invalid scientific status: {self.scientific_status!r}. "
                f"Expected one of {sorted(valid_statuses)}."
            )

        if any(not isinstance(concern, ScientificConcern) for concern in self.concerns):
            raise TypeError("Every concern must be a ScientificConcern member.")

        if self.name is not None and not self.name.strip():
            raise ValueError("Algorithm name cannot be empty.")

        if self.year is not None and not 1800 <= self.year <= 2100:
            raise ValueError(
                f"Invalid publication year: {self.year}. "
                "Expected a value between 1800 and 2100."
            )

        if self.family is not None and not self.family.strip():
            raise ValueError("Algorithm family cannot be empty.")
