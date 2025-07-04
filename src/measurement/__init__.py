from .base import BaseMeasurement, MeasurementResult
from .pulse import PulseMeasurement, PulseOperation, PulseConfiguration
from .IV import IVCurveMeasurement, IVSweepConfiguration
# from .ltp_ltd import LTPLTDMeasurement, LTPLTDConfiguration

# Import the functions
from ..analysis.plotting import plot_measurement_results as _plot_func
from ..utils.data_io import save_measurement_results as _save_func
from ..analysis.plotting import plot_iv_curves as _plot_iv_func

# Create wrapper methods that properly handle 'self'
def _plot_wrapper(self, result, plot_type='all'):
    """Plot measurement results (wrapper for backward compatibility)."""
    return _plot_func(result, plot_type)

def _save_wrapper(self, result, filename, format='csv'):
    """Save measurement results (wrapper for backward compatibility)."""
    return _save_func(result, filename, format)

# Create wrapper method for IV curves
def _plot_iv_wrapper(self, result, save_plot=False, filename=None):
    """Plot detailed IV curve analysis."""
    return _plot_iv_func(result, save_plot=save_plot, filename=filename)

# Attach the wrapper methods (not the original functions)
PulseMeasurement.plot_measurement_results = _plot_wrapper
PulseMeasurement.save_results = _save_wrapper
IVCurveMeasurement.plot_measurement_results = _plot_wrapper
IVCurveMeasurement.save_results = _save_wrapper
IVCurveMeasurement.plot_iv_curves = _plot_iv_wrapper
# LTPLTDMeasurement.save_results = _save_wrapper

