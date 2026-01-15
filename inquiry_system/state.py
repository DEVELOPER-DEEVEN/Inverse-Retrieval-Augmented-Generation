from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Variable:
    name: str # e.g. "Risk Tolerance", "Decision Style"
    value: str # e.g. "High", "Analytical"
    confidence: float # 0.0 to 1.0
    reasoning: str # Why we believe this

@dataclass
class BeliefState:
    """
    Represents the internal probabilistic model of the user.
    Maintained by BeliefManager, consumed by InquiryAgent.
    """
    
    # Core probabilistic model
    latent_traits: Dict[str, Variable] = field(default_factory=dict)
    
    # Meta-analysis
    overall_confidence: float = 0.0
    top_uncertainties: List[str] = field(default_factory=list) # e.g. ["Political Views", "Spending Habits"]
    
    # Context
    conversation_history: List[Dict[str, str]] = field(default_factory=list) # [{"role": "agent", "content": "..."}, ...]
    profile_summary: str = "" # A concise textual summary for context
    
    def to_prompt_context(self) -> str:
        """
        Formats the belief state for the LLM Agent to understand what it knows and what it doesn't.
        """
        traits_str = "\n".join(
            f"- {k}: {v.value} (Conf: {v.confidence:.2f}) - {v.reasoning}"
            for k, v in self.latent_traits.items()
        )
        
        uncertainties_str = ", ".join(self.top_uncertainties)
        
        return f"""
        [CURRENT USER MODEL]
        Summary: {self.profile_summary}
        
        [LATENT TRAITS]
        {traits_str}
        
        [UNCERTAINTIES / KNOWLEDGE GAPS]
        {uncertainties_str}
        
        [OVERALL CONFIDENCE]
        {self.overall_confidence:.2f}
        """
