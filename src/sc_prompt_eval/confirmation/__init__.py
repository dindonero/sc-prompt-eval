"""
Static confirmation modules for GPTScan-style post-LLM validation.

These modules use Slither's internal APIs to verify LLM-extracted findings
before final reporting, reducing false positives by 65.84% (per Sun2023).

Modules:
- DF (DataFlow): Traces variable dependencies using Slither's data dependency analysis
- VC (ValueComparison): Checks if variables are compared in require/if conditions
- OC (OrderCheck): Verifies statement execution order using CFG analysis
- FA (FunctionArgs): Validates argument sources and taint analysis
"""

from .base import BaseConfirmer, ConfirmationResult
from .dataflow import DataFlowConfirmer
from .value_comparison import ValueComparisonConfirmer
from .order_check import OrderCheckConfirmer
from .function_args import FunctionArgConfirmer

__all__ = [
    "BaseConfirmer",
    "ConfirmationResult",
    "DataFlowConfirmer",
    "ValueComparisonConfirmer",
    "OrderCheckConfirmer",
    "FunctionArgConfirmer",
]
