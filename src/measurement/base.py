from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time

@dataclass
class MeasurementResult:
    """Container for measurement results."""
    voltages: List[float]
    currents: List[float]
    resistances: List[float]
    pulse_widths: List[int]
    timestamps: List[float]
    is_read: List[bool]
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseMeasurement:
    """Base class for all measurement types."""
    
    def __init__(self, instrument):
        """Initialize with an ArC Two instrument instance."""
        self.instrument = instrument
        
    # def ground_all_pins(self):
    #     """Ground all pins for safety."""
    #     try:
    #         self.instrument.ground_all()
    #         self.instrument.execute()
    #         self.instrument.wait()
    #         return True
    #     except Exception as e:
    #         print(f"Error grounding pins: {e}")
    #         return False
    
    # def get_write_voltages(self, result):
    #     """Extract write voltages from result metadata or direct fields"""
    #     # First check if result has is_read flags
    #     if hasattr(result, 'is_read') and hasattr(result, 'voltages') and len(result.is_read) == len(result.voltages):
    #         # Filter voltages where is_read is False
    #         return [v for v, is_r in zip(result.voltages, result.is_read) if not is_r]
            
    #     return []

    # def get_write_pulse_widths(self, result):
    #     """Extract write pulse widths from result metadata or direct fields"""
    #     # First check if result has is_read flags
    #     if hasattr(result, 'is_read') and hasattr(result, 'pulse_widths') and len(result.is_read) == len(result.pulse_widths):
    #         # Filter pulse_widths where is_read is False
    #         return [pw for pw, is_r in zip(result.pulse_widths, result.is_read) if not is_r]
            
    #     return []

    # def get_read_voltages(self, result):
    #     """Get read voltages from result fields or metadata"""
    #     # First check if result has is_read flags
    #     if hasattr(result, 'is_read') and hasattr(result, 'voltages') and len(result.is_read) == len(result.voltages):
    #         # Filter voltages where is_read is True
    #         return [v for v, is_r in zip(result.voltages, result.is_read) if is_r]
        
    # def get_read_pulse_widths(self, result):
    #     """Get read pulse widths from result fields or metadata"""
    #     # First check if result has is_read flags
    #     if hasattr(result, 'is_read') and hasattr(result, 'pulse_widths') and len(result.is_read) == len(result.pulse_widths):
    #         # Filter pulse_widths where is_read is True
    #         return [pw for pw, is_r in zip(result.pulse_widths, result.is_read) if is_r]