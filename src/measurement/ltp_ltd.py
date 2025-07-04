# LEGACY CODE: Entire file replaced by relative pulse sequencing approach
# now uses:
# - pulse_patterns.create_ltp_ltd_sequence_new() 
# - PulseMeasurement.run_relative_sequence()
# Instead of this LTPLTDMeasurement class


# from typing import Dict, List, Optional, Union, Tuple, Any
# from dataclasses import dataclass, field
# from datetime import datetime
# import numpy as np
# import matplotlib.pyplot as plt
# import os
       

# from .base import BaseMeasurement, MeasurementResult
# from ..analysis.plotting import plot_pulse_sequence
# from ..instruments.pulse_patterns import create_ltp_ltd_sequence_enhanced as create_ltp_ltd_sequence
# from ..instruments.pulse_sequence import PulseSequence
# from .pulse import PulseMeasurement

# @dataclass
# class LTPLTDConfiguration:
#     """Configuration for LTP/LTD protocols."""
#     row_pin: int
#     column_pin: int
#     vread: float = 0.2
#     period: int = 1000000  # 1ms in ns
    
#     # LTP parameters
#     vwrite_ltp_start: float = 1.0
#     vwrite_ltp_end: float = 2.0
#     twrite_ltp_start: int = 5000  
#     twrite_ltp_end: int = 5000    
#     nb_ltp: int = 10
    
#     # LTD parameters
#     vwrite_ltd_start: float = -1.0
#     vwrite_ltd_end: float = -2.0
#     twrite_ltd_start: int = 5000
#     twrite_ltd_end: int = 5000 
#     nb_ltd: int = 10
    
#     compliance_current: float = 1e-3
#     read_pulse_offset: int = 340000
#     metadata: Dict[str, Any] = field(default_factory=dict)

# class LTPLTDMeasurement(BaseMeasurement):
#     """
#     Class for LTP/LTD measurements using ArC TWO.
#     Uses existing sequence generation module and PulseMeasurement for execution.
#     """
    
#     def __init__(self, instrument):
#         """Initialize with an ArC Two instrument instance."""
#         super().__init__(instrument)
    
#     def configure_sequence(self, 
#                          config: Union[LTPLTDConfiguration, Dict],
#                          debug: bool = False) -> Dict[str, Any]:
#         """
#         Configure the LTP/LTD sequence based on provided parameters.
        
#         Parameters:
#         -----------
#         config : LTPLTDConfiguration or Dict
#             Configuration for LTP/LTD protocol
#         debug : bool
#             Whether to print debug information
            
#         Returns:
#         --------
#         Dict[str, Any]
#             Configured pulse sequence dictionary
#         """
#         # Handle dictionary input for backward compatibility
#         if isinstance(config, dict):
#             config = LTPLTDConfiguration(
#                 row_pin=config.get('row_pin'),
#                 column_pin=config.get('column_pin'),
#                 vread=config.get('vread', 0.2),
#                 period=config.get('period', 1000000),
#                 vwrite_ltp_start=config.get('vwrite_ltp_start', 1.0),
#                 vwrite_ltp_end=config.get('vwrite_ltp_end', 2.0),
#                 twrite_ltp_start=config.get('twrite_ltp_start', 100000),
#                 twrite_ltp_end=config.get('twrite_ltp_end', 200000),
#                 nb_ltp=config.get('nb_ltp', 10),
#                 vwrite_ltd_start=config.get('vwrite_ltd_start', -1.0),
#                 vwrite_ltd_end=config.get('vwrite_ltd_end', -2.0),
#                 twrite_ltd_start=config.get('twrite_ltd_start', 100000),
#                 twrite_ltd_end=config.get('twrite_ltd_end', 200000),
#                 nb_ltd=config.get('nb_ltd', 10),
#                 compliance_current=config.get('compliance_current', 1e-3),
#                 read_pulse_offset=config.get('read_pulse_offset', 340000),
#                 metadata=config
#             )
        
#     # Use the original create_ltp_ltd_sequence for visualization compatibility
#         pulse_sequence = create_ltp_ltd_sequence(
#             nb_ltp=config.nb_ltp, 
#             nb_ltd=config.nb_ltd,
#             vwrite_ltp_start=config.vwrite_ltp_start,
#             vwrite_ltp_end=config.vwrite_ltp_end,
#             twrite_ltp_start=config.twrite_ltp_start,
#             twrite_ltp_end=config.twrite_ltp_end,
#             vwrite_ltd_start=config.vwrite_ltd_start,
#             vwrite_ltd_end=config.vwrite_ltd_end,
#             twrite_ltd_start=config.twrite_ltd_start,
#             twrite_ltd_end=config.twrite_ltd_end
#         )
        
#         # Create sequence dictionary with period
#         sequence = {
#             'sequence': pulse_sequence['sequence'],  # Extract the inner sequence directly
#             'period': config.period,
#             'read_voltage': config.vread,
#             'row_pin': config.row_pin,
#             'column_pin': config.column_pin
#         }
        
#         # Also create enhanced sequence and attach it for execution
#         from ..instruments.pulse_patterns import create_ltp_ltd_sequence_enhanced
#         enhanced_sequence = create_ltp_ltd_sequence_enhanced(
#             nb_ltp=config.nb_ltp, 
#             nb_ltd=config.nb_ltd,
#             vwrite_ltp_start=config.vwrite_ltp_start,
#             vwrite_ltp_end=config.vwrite_ltp_end,
#             twrite_ltp_start=config.twrite_ltp_start,
#             twrite_ltp_end=config.twrite_ltp_end,
#             vwrite_ltd_start=config.vwrite_ltd_start,
#             vwrite_ltd_end=config.vwrite_ltd_end,
#             twrite_ltd_start=config.twrite_ltd_start,
#             twrite_ltd_end=config.twrite_ltd_end,
#             period=config.period,
#             read_voltage=config.vread,
#             read_pulse_offset=config.read_pulse_offset
#         )
        
#         # Store both formats for compatibility
#         sequence['enhanced_sequence'] = enhanced_sequence
        
#         if debug:
#             print(f"Generated LTP/LTD sequence with {len(pulse_sequence)} pulses")
#             print(f"LTP: {config.nb_ltp} pulses from {config.vwrite_ltp_start}V/{config.twrite_ltp_start}ns to {config.vwrite_ltp_end}V/{config.twrite_ltp_end}ns")
#             print(f"LTD: {config.nb_ltd} pulses from {config.vwrite_ltd_start}V/{config.twrite_ltd_start}ns to {config.vwrite_ltd_end}V/{config.twrite_ltd_end}ns")
#             print(f"Period: {config.period}ns, Read voltage: {config.vread}V")
        
#         return sequence
    
#     def visualize_ltp_ltd_sequence(self, sequence_obj):
#         """
#         Visualize the LTP/LTD sequence for verification.
        
#         Parameters:
#         -----------
#         sequence_obj : Dict or PulseSequence
#             Pulse sequence to visualize
#         """
#         from ..analysis.plotting import plot_pulse_sequence
#         # If we're using the enhanced_sequence PulseSequence object, use it directly
#         if 'enhanced_sequence' in sequence_obj and hasattr(sequence_obj['enhanced_sequence'], 'to_dict'):
#             pulse_seq = sequence_obj['enhanced_sequence']
#             sequence_obj = pulse_seq
#         elif isinstance(sequence_obj, dict) and 'sequence' in sequence_obj and len(sequence_obj['sequence']) > 0:
#             # If sequence contains pulse data directly, create a compatible format
#             all_pulses = sequence_obj['sequence']
#             sequence_obj = {
#                 'all_pulses': all_pulses,
#                 'write_pulses': [p for p in all_pulses if not p.get('is_read', False)],
#                 'read_pulses': [p for p in all_pulses if p.get('is_read', False)]
#             }
        
#         # Determine total duration for visualization
#         total_duration_ms = 10  # Default fallback
#         if isinstance(sequence_obj, dict):
#             if 'all_pulses' in sequence_obj and sequence_obj['all_pulses']:
#                 pulses = sequence_obj['all_pulses']
#                 if pulses and 'time_offset' in pulses[-1]:
#                     last_time = pulses[-1]['time_offset']
#                     last_width = pulses[-1].get('pulse_width', 0)
#                     total_duration_ms = (last_time + last_width + 500000) / 1_000_000
#             elif 'sequence' in sequence_obj and sequence_obj['sequence']:
#                 pulses = sequence_obj['sequence']
#                 if pulses and 'time_offset' in pulses[-1]:
#                     last_time = pulses[-1]['time_offset']
#                     last_width = pulses[-1].get('pulse_width', 0)
#                     total_duration_ms = (last_time + last_width + 500000) / 1_000_000

#         print(f"Visualizing sequence with duration: {total_duration_ms} ms")
        
#         plot_pulse_sequence(
#             sequence_obj,
#             title="LTP/LTD Pulse Sequence Visualization",
#             show=True,
#             time_range=[0, total_duration_ms]
#         )
        
#     def run_ltp_ltd_protocol(self,
#                         sequence_obj,  # Now required
#                         compliance_current=1e-3,
#                         visualize=False,
#                         debug=False) -> MeasurementResult:
#         """
#         Run LTP/LTD protocol with a pre-configured sequence.
        
#         Parameters:
#         -----------
#         sequence_obj : Dict or PulseSequence
#             Pre-configured sequence object from configure_sequence()
#         compliance_current : float
#             Maximum allowed current
#         visualize : bool
#             Whether to visualize the sequence before executing
#         debug : bool
#             Whether to print debug information
                
#         Returns:
#         --------
#         MeasurementResult
#             Container with measurement results
#         """
#         if sequence_obj is None:
#             raise ValueError("sequence_obj is required. Call configure_sequence() first.")
        
#         # Use the enhanced sequence if available
#         execution_sequence = sequence_obj.get('enhanced_sequence', sequence_obj)
        
#         # Visualize if requested
#         if visualize:
#             self.visualize_ltp_ltd_sequence(sequence_obj)
        
#         # Get pins from sequence_obj
#         row_pin = sequence_obj.get('row_pin')
#         column_pin = sequence_obj.get('column_pin')
        
#         # Execute using the enhanced sequence
#         from .pulse import PulseMeasurement
#         pulse_measurement = PulseMeasurement(self.instrument)
        
#         result = pulse_measurement.apply_pulse_sequence_object(
#             low_pin=row_pin, 
#             high_pin=column_pin,
#             sequence_obj=execution_sequence,  # Use the enhanced sequence
#             compliance_current=compliance_current,
#             debug=debug
#         )
        
#         # Add minimal metadata
#         result.metadata.update({
#             "measurement_type": "ltp_ltd"
#         })
        
#         return result
    
#     def run_custom_pulse_sequence(self,
#                                 pulse_sequence: List[Dict],
#                                 row_pin: int,
#                                 column_pin: int,
#                                 vread: float = 0.2,
#                                 period: int = 1000000,
#                                 read_pulse_offset: int = 340000,
#                                 compliance_current: float = 1e-3,
#                                 visualize: bool = False,
#                                 debug: bool = False) -> MeasurementResult:
#         """
#         Run a custom pulse sequence for LTP/LTD measurements.
        
#         Parameters:
#         -----------
#         pulse_sequence : List[Dict]
#             List of pulse dictionaries, each containing at least 'voltage' and 'pulse_width' keys
#         row_pin : int
#             The row pin to apply voltage to
#         column_pin : int
#             The column pin to measure current from
#         vread : float
#             Read voltage
#         period : int
#             Time between pulses in nanoseconds
#         read_pulse_offset : int
#             Delay between pulse and read operation in nanoseconds
#         compliance_current : float
#             Maximum allowed current
#         visualize : bool
#             Whether to visualize the sequence before executing
#         debug : bool
#             Whether to print debug information
            
#         Returns:
#         --------
#         MeasurementResult
#             Container with measurement results
#         """
#         # Create a PulseSequence object
#         pulse_seq = PulseSequence(period=period)

#         # Add pulses to the sequence
#         for pulse in pulse_sequence:
#             if isinstance(pulse, dict):
#                 if 'voltage' in pulse and 'pulse_width' in pulse:
#                     # Add write pulse with delay until read
#                     write_pulse_width = pulse['pulse_width']
#                     delay_after_write = max(0, read_pulse_offset - write_pulse_width)
                    
#                     pulse_seq.add_write_pulse(
#                         voltage=pulse['voltage'], 
#                         pulse_width=write_pulse_width,
#                         delay_after=delay_after_write
#                     )
                    
#                     # Add read pulse with delay until next cycle
#                     read_pulse_width = pulse_seq.default_read_pulse_width
#                     delay_after_read = max(0, period - read_pulse_offset - read_pulse_width)
                    
#                     pulse_seq.add_read_pulse(
#                         voltage=vread,
#                         pulse_width=read_pulse_width,
#                         delay_after=delay_after_read
#                     )
#                 else:
#                     raise ValueError(f"Pulse {pulse} is missing required 'voltage' or 'pulse_width' keys")
#             elif isinstance(pulse, tuple) and len(pulse) == 2:
#                 # Add write pulse with delay until read
#                 write_pulse_width = pulse[1]
#                 delay_after_write = max(0, read_pulse_offset - write_pulse_width)
                
#                 pulse_seq.add_write_pulse(
#                     voltage=pulse[0], 
#                     pulse_width=write_pulse_width,
#                     delay_after=delay_after_write
#                 )
                
#                 # Add read pulse with delay until next cycle
#                 read_pulse_width = pulse_seq.default_read_pulse_width
#                 delay_after_read = max(0, period - read_pulse_offset - read_pulse_width)
                
#                 pulse_seq.add_read_pulse(
#                     voltage=vread,
#                     pulse_width=read_pulse_width,
#                     delay_after=delay_after_read
#                 )
#             else:
#                 raise ValueError(f"Unsupported pulse format: {pulse}")

#         # Execute the sequence using apply_pulse_sequence_object
#         pulse_measurement = PulseMeasurement(self.instrument)

#         result = pulse_measurement.apply_pulse_sequence_object(
#             low_pin=row_pin,
#             high_pin=column_pin,
#             sequence_obj=pulse_seq,
#             compliance_current=compliance_current,
#             debug=debug
#         )
        
#         # Add metadata
#         result.metadata.update({
#             "measurement_type": "custom_ltp_ltd",
#             "sequence_length": len(pulse_sequence)
#         })
        
#         return result
    
#     def plot_ltp_ltd_results(self, result: MeasurementResult, save_plot = False, filename = None, save_dir = "figures", show_plot: bool = True) -> None:
#         """
#         Plot the results from an LTP/LTD measurement with detailed analysis.
        
#         Parameters:
#         -----------
#         result : MeasurementResult
#             Measurement result from LTP/LTD experiment
#         show_plot : bool
#             Whether to display the plot
#         """
#         if not result:
#             print("No data to plot")
#             return
        
#         # Extract parameters
#         nb_ltp = result.metadata.get('nb_ltp', 0)
#         nb_ltd = result.metadata.get('nb_ltd', 0)
#         protocol_type = result.metadata.get('protocol_type', 'unknown')

#         # Get write voltages and pulse widths from metadata
#         write_voltages = self.get_write_voltages(result)
#         write_pulse_widths = self.get_write_pulse_widths(result)
        
#         # Create pulse numbers
#         pulse_numbers = np.arange(1, len(write_voltages) + 1)
        

#         # Plot write voltages
#         # Create figure with subplots
#         fig = plt.figure(figsize=(12, 10))
#         gs = fig.add_gridspec(3, 2)
        
#         # Plot 1: Voltage vs Pulse Number (top row)
#         ax1 = fig.add_subplot(gs[0, :])
#         ax1.plot(pulse_numbers, write_voltages, 'o-', color='blue')
#         ax1.set_title('Applied Voltage vs Pulse Number')
#         ax1.set_xlabel('Pulse Number')
#         ax1.set_ylabel('Voltage (V)')
#         ax1.axvspan(0.5, nb_ltp + 0.5, alpha=0.2, color='red', label='LTP')
#         ax1.axvspan(nb_ltp + 0.5, nb_ltp + nb_ltd + 0.5, alpha=0.2, color='blue', label='LTD')
#         ax1.grid(True)
#         ax1.legend()
        
#         # Plot 2: Current vs Pulse Number
#         ax2 = fig.add_subplot(gs[1, 0])
#         ax2.plot(pulse_numbers, result.currents, 'o-', color='red')
#         ax2.set_title('Current vs Pulse Number')
#         ax2.set_xlabel('Pulse Number')
#         ax2.set_ylabel('Current (A)')
#         ax2.axvspan(0.5, nb_ltp + 0.5, alpha=0.2, color='red')
#         ax2.axvspan(nb_ltp + 0.5, nb_ltp + nb_ltd + 0.5, alpha=0.2, color='blue')
#         ax2.grid(True)
        
#         # Plot 3: Pulse Width vs Pulse Number (if applicable)
#         ax3 = fig.add_subplot(gs[1, 1])
#         ax3.plot(pulse_numbers, np.array(write_pulse_widths)/1000, 'o-', color='purple')
#         ax3.set_title('Pulse Width vs Pulse Number')
#         ax3.set_xlabel('Pulse Number')
#         ax3.set_ylabel('Pulse Width (μs)')
#         ax3.axvspan(0.5, nb_ltp + 0.5, alpha=0.2, color='red')
#         ax3.axvspan(nb_ltp + 0.5, nb_ltp + nb_ltd + 0.5, alpha=0.2, color='blue')
#         ax3.grid(True)
        
#         # Plot 4: Resistance (log scale) vs Pulse Number
#         ax4 = fig.add_subplot(gs[2, 0])
#         ax4.plot(pulse_numbers, np.abs(result.resistances), 'o-', color='green')
#         ax4.set_title('Resistance vs Pulse Number')
#         ax4.set_xlabel('Pulse Number')
#         ax4.set_ylabel('Resistance (Ω)')
#         ax4.set_yscale('log')
#         ax4.axvspan(0.5, nb_ltp + 0.5, alpha=0.2, color='red')
#         ax4.axvspan(nb_ltp + 0.5, nb_ltp + nb_ltd + 0.5, alpha=0.2, color='blue')
#         ax4.grid(True)
        
#         # # Plot 5: Resistance vs Voltage
#         # ax5 = fig.add_subplot(gs[2, 1])
        
#         # # Separate LTP and LTD data
#         # ltp_voltages = write_voltages[:nb_ltp] if nb_ltp > 0 else []
#         # ltd_voltages = write_voltages[nb_ltp:nb_ltp+nb_ltd] if nb_ltd > 0 else []
#         # ltp_resistances = [np.abs(result.resistances[i]) for i in range(nb_ltp)] if nb_ltp > 0 else []
#         # ltd_resistances = [np.abs(result.resistances[i+nb_ltp]) for i in range(nb_ltd)] if nb_ltd > 0 else []
        
#         # # if len(ltp_voltages) > 0:
#         # ax5.plot(ltp_voltages, ltp_resistances, 'o-', color='red', label='LTP')
#         # # if len(ltd_voltages) > 0:
#         # ax5.plot(ltd_voltages, ltd_resistances, 'o-', color='blue', label='LTD')
            
#         # ax5.set_title('Resistance vs Voltage')
#         # ax5.set_xlabel('Voltage (V)')
#         # ax5.set_ylabel('Resistance (Ω)')
#         # ax5.set_yscale('log')
#         # ax5.grid(True)
#         # ax5.legend()
        
#         plt.tight_layout()
        
#         # Save the plot if requested
#         if save_plot:
#             # Create figures directory if it doesn't exist
#             os.makedirs(save_dir, exist_ok=True)
            
#             # Generate filename with timestamp if not provided
#             if filename is None:
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 filename = f"ltp_ltd_{protocol_type}_{timestamp}"
            
#             # Add extension if not present
#             if not filename.endswith('.png') and not filename.endswith('.jpg') and not filename.endswith('.pdf'):
#                 full_path = os.path.join(save_dir, f"{filename}.png")
#             else:
#                 full_path = os.path.join(save_dir, filename)
            
#             # Save the figure
#             plt.savefig(full_path, dpi=300, bbox_inches='tight')
#             print(f"Plot saved to {full_path}")
    
#         # Show plot if requested
#         if show_plot:
#             plt.show()
