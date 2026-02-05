"""Runner modules for different prompt strategies."""
from .run import run_experiment, main
from .multiagent import run_smartaudit, SmartAuditResult, extract_final_findings
from .iaudit_runner import run_iaudit_pipeline, iAuditConfig, iAuditResult

__all__ = [
    # Main runner
    "run_experiment",
    "main",
    # P5 LLM-SmartAudit (Wei et al. 2024)
    "run_smartaudit",
    "SmartAuditResult",
    "extract_final_findings",
    # P6 iAudit (Ma et al. 2024)
    "run_iaudit_pipeline",
    "iAuditConfig",
    "iAuditResult",
]
