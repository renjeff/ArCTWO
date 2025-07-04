import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
from pyarc2 import DataMode

from .base import BaseMeasurement, MeasurementResult
from ..utils.hardware_utils import get_cluster_timing
from src.instruments.pulse_sequence import PulseSequence

@dataclass
class PulseOperation:
    """Single pulse operation with all parameters needed for execution."""
    low_pin: int
    high_pin: int
    voltage: float
    pulse_width: int
    compensation: float = 1.33
    read_after: bool = False
    read_voltage: float = 0.1
    differential: bool = True

@dataclass
class PulseConfiguration:
    """Complete configuration for a sequence of pulse operations."""
    operations: List[PulseOperation]
    period: int
    write_read_delay: int = 340000
    inter_pulse_delay: int = 0
    read_at_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class PulseMeasurement(BaseMeasurement):
    """
    Class for executing pulse sequences and measuring results.
    """
    
    @staticmethod
    def _get_pulse_type(pulse):
        if 'type' in pulse:
            return pulse['type'].lower()
        elif 'is_read' in pulse:
            return 'read' if pulse['is_read'] else 'write'
        else:
            return 'write'
        
    def compute_offsets_with_transition_overheads(self, pulses, measured_overheads_us_dict):
        """
        Compute pulse start times (offsets) in ns using transition-specific measured overheads and programmed delays.
        - pulses: list of pulse dicts (must have 'type' or 'is_read', 'pulse_width', and 'delay' keys)
        - measured_overheads_us_dict: dict mapping (prev_type, curr_type) to overhead in us
        Returns: list of offsets in ns
        """
        offsets_ns = [0]
        for i in range(1, len(pulses)):
            prev_type = self._get_pulse_type(pulses[i-1])
            curr_type = self._get_pulse_type(pulses[i])
            prev_width = pulses[i-1].get('pulse_width', 530_000)
            delay_ns = pulses[i-1].get('delay_after', 0)
            overhead_us = measured_overheads_us_dict.get((prev_type, curr_type), 0)
            overhead_ns = int(overhead_us * 1000)
            next_start = offsets_ns[-1] + prev_width + overhead_ns + delay_ns
            offsets_ns.append(next_start)
        return offsets_ns
    
    def visualize_pulse_sequence(self, sequence_obj):
        """
        Visualize a pulse sequence for verification.
        
        Parameters:
        -----------
        sequence_obj : PulseSequence
            Pulse sequence to visualize
        """
        try:
            from ..analysis.plotting import plot_pulse_sequence
            
            measured_overheads_us = {
                ('write', 'read'): 240,
                ('read', 'write'): 228,
                ('write', 'write'): 66,
                ('read', 'read'): 274,
            }
            # Convert sequence object to dictionary format if needed
            if hasattr(sequence_obj, 'to_dict'):
                sequence_data = sequence_obj.to_dict()
            else:
                sequence_data = sequence_obj
            
            print("DEBUG: Sequence data keys:", list(sequence_data.keys()) if isinstance(sequence_data, dict) else "Not a dict")
            if isinstance(sequence_data, dict) and 'all_pulses' in sequence_data:
                print(f"DEBUG: Found {len(sequence_data['all_pulses'])} pulses")
            
            all_pulses = sequence_data.get('all_pulses', [])
            for pulse in all_pulses:
                if (pulse.get('type', '').lower() == 'read') or pulse.get('is_read', False):
                    pulse['pulse_width'] = 530_000 # Enforce hardware read pulse width

            #  Compute offsets using transition overheads and programmed delays
            offsets_ns = self.compute_offsets_with_transition_overheads(all_pulses, measured_overheads_us)
            for pulse, t in zip(all_pulses, offsets_ns):
                pulse['time_offset'] = t

            # 4. Calculate total duration for visualization
            max_end_time = 0
            for pulse in all_pulses:
                pulse_width = pulse.get('pulse_width', 0)
                if pulse_width is None:
                    pulse_width = sequence_data.get('default_read_pulse_width', 530_000)
                pulse_end = pulse.get('time_offset', 0) + pulse_width
                max_end_time = max(max_end_time, pulse_end)
            buffer_ns = max_end_time * 0.1 
            total_duration_ns = max_end_time + buffer_ns
            total_duration_ms = total_duration_ns / 1_000_000
                
            print(f"DEBUG: Max pulse end time: {max_end_time/1_000_000:.2f} ms")
            print(f"DEBUG: Total duration with buffer: {total_duration_ms:.2f} ms")                 
            print(f"DEBUG: Calculated total duration: {total_duration_ms:.2f} ms")
            
            
            # Call the plotting function with proper error handling
            try:
                fig, ax = plot_pulse_sequence(
                    sequence_data, 
                    title="Custom Pulse Sequence Visualization", 
                    show=True,
                    time_range=[0, total_duration_ms]
                )
                return fig, ax
            except Exception as plot_error:
                print(f"Error in plot_pulse_sequence: {plot_error}")
                # Print more details about the sequence structure
                print("Sequence structure:")
                if isinstance(sequence_data, dict):
                    for key, value in sequence_data.items():
                        if isinstance(value, list):
                            print(f"  {key}: list with {len(value)} items")
                            if value and len(value) > 0:
                                print(f"    First item: {value[0]}")
                        else:
                            print(f"  {key}: {type(value).__name__}")
                raise plot_error
                
        except ImportError as e:
            print(f"Could not import plot_pulse_sequence: {e}")
            print("Using fallback visualization...")
            self._fallback_visualization(sequence_obj)
        except Exception as e:
            print(f"Error in visualize_pulse_sequence: {e}")
            import traceback
            traceback.print_exc()


    
    def _execute_read_pulse(self, high_pin, voltage, pulse, read_pulses, debug):
        self.instrument.config_channels([(high_pin, voltage)])
        self.instrument.read_slice_open_deferred([high_pin], True)
        read_pulses.append(pulse)
        
        if debug:
            print(f"read pulse:{voltage}V")

    def _execute_write_pulse(self, low_pin, high_pin, voltage, pulse_width, differential, debug):
        if differential:
            self._execute_differential_write(low_pin, high_pin, voltage, pulse_width, debug)
        else:
            self._execute_single_ended_write(low_pin, high_pin, voltage, pulse_width, debug)

    def _execute_differential_write(self, low_pin, high_pin, voltage, pulse_width, debug):
        if voltage < 0:
            self.instrument.pulse_one(high_pin, low_pin, abs(voltage), pulse_width)
        else:
            self.instrument.pulse_one(low_pin, high_pin, voltage, pulse_width)

    def _execute_single_ended_write(self, low_pin, high_pin, voltage, pulse_width, debug):
        if voltage < 0:
            self.instrument.config_channels([(high_pin, 0.0)])
            abs_voltage = abs(voltage)
            cluster_timings = get_cluster_timing([low_pin], pulse_width)
            self.instrument.pulse_slice_fast_open(
                [(low_pin, abs_voltage, 0.0)],
                cluster_timings,
                True
            )
        else:
            self.instrument.config_channels([(low_pin, 0.0)])
            cluster_timings = get_cluster_timing([high_pin], pulse_width)
            self.instrument.pulse_slice_fast_open(
                [(high_pin, voltage, 0.0)],
                cluster_timings,
                True
            )

    def run_relative_sequence(self, low_pin, high_pin, seq, differential=True, debug=False):
        """
        Execute a PulseSequence object (or dict) built with build_relative_pulse_sequence,
        applying each pulse in order with its delay.
        """
        ARC2_DELAY_OVERHEAD = 244_000
        ARC2_READ_OVERHEAD = 100_000

        measured_gaps_us = [66, 240, 228, 274]

        # voltage amplitude compensation constants
        tau = 3100
        initial_fraction = 0.732
        delta_fraction = 1 - initial_fraction
        max_compensation = 2.0
        short_pulse_threshold = 2000
    
        def linear_compensation(pulse_width):
            m = -0.000133
            b = 1.2766
            return m * pulse_width + b
        
        base_time = time.time()
        accumulated_time_ns = 0
        timestamps = []
        pulse_timestamps = {}
    
        # Convert to dict if needed
        sequence_data = seq.to_dict() if hasattr(seq, 'to_dict') else seq
        pulses = sequence_data.get('all_pulses', [])
        read_pulses = []

        times_us = [0]
        for i in range(1, len(pulses)):
            prev_width_us = (pulses[i-1]['pulse_width'] or sequence_data.get('default_read_pulse_width', 530_000)) / 1000
            gap_us = measured_gaps_us[i-1] if i-1 < len(measured_gaps_us) else 0
            next_start = times_us[-1] + prev_width_us + gap_us
            times_us.append(next_start)
        # Store timestamps (convert us to seconds)
        for idx, t_us in enumerate(times_us):
            pulse_timestamps[idx] = base_time + (t_us / 1e6)

        accumulated_time_ns = 0

        for idx, pulse in enumerate(pulses):
            pulse_timestamps[idx] = base_time +(accumulated_time_ns / 1e9)
            if 'type' in pulse:
                typ = pulse['type'].lower()
            elif 'is_read' in pulse:
                typ = 'read' if pulse['is_read'] else 'write'
            else:
                raise KeyError("Pulse must have either 'type' or 'is_read' key")
            voltage = pulse['voltage']
            pulse_width = pulse['pulse_width']
            delay = pulse.get('delay_after', 0)
            if typ == 'write':
                        # Apply voltage compensation
                        if pulse_width < short_pulse_threshold:
                            compensation = linear_compensation(pulse_width)
                        else:
                            exp_factor = 1 - np.exp(-pulse_width / tau)
                            compensation = initial_fraction + delta_fraction / exp_factor if exp_factor > 0 else 1.0
                        accumulated_time_ns += pulse_width
                        compensation = min(compensation, max_compensation)
                        compensated_voltage = voltage * compensation
                        
                        if debug:
                            print(f"Write: {voltage:.3f}V → {compensated_voltage:.3f}V (comp={compensation:.2f})")
                            
                        self._execute_write_pulse(low_pin, high_pin, compensated_voltage, pulse_width, differential, debug)
            elif typ == 'read':
                self._execute_read_pulse(high_pin, voltage, pulse, read_pulses, debug)
                accumulated_time_ns += sequence_data.get('default_read_pulse_width', 530_000)
            else:
                raise ValueError(f"Unknown pulse type: {typ}")
            if delay and delay > 0:
   
                self.instrument.delay(delay)
                actual_delay = max(0, delay)
                accumulated_time_ns += actual_delay

        self.instrument.execute()
        self.instrument.wait()

        read_voltages = []
        read_currents = []
        read_resistances = []
        read_pulse_widths = []
        read_timestamps = []

        if debug:
            print(f"Collecting {len(read_pulses)} readings...")

        try:
            reading_data = list(self.instrument.get_iter(DataMode.All))
            if debug:
                print(f"Retrieved {len(reading_data)} readings from hardware")
                if reading_data:
                    print(f"First reading data structure: {reading_data[0]}")
            for i, data in enumerate(reading_data):
                if i >= len(read_pulses):
                    break
                read_pulse = read_pulses[i]
                read_voltage = read_pulse['voltage']
                try:
                    current = None
                    if isinstance(data, dict) and 'current' in data:
                        current = data['current']
                    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                        current = data[0].get(high_pin)
                    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                        current = data[0][high_pin]
                    else:
                        if debug:
                            print(f"  Data structure: {data}")
                        current = data[0][high_pin]
                    read_voltages.append(read_voltage)
                    read_currents.append(current)
                    read_resistances.append(read_voltage / current if abs(current) > 1e-12 else float('inf'))
                    read_pulse_widths.append(read_pulse['pulse_width'])
                    read_timestamps.append(time.time())
                    if debug:
                        print(f"  Reading {i+1}/{len(read_pulses)}: V={read_voltage}V, I={current:.3e}A, R={read_resistances[-1]:.2e}Ω")
                except Exception as e:
                    if debug:
                        print(f"  Error processing reading {i+1}: {e}")
                        print(f"  Data structure: {data}")
        except Exception as e:
            if debug:
                print(f"Error collecting readings: {e}")

        # Optionally, also collect write pulse info for metadata
        write_pulses = [p for p in pulses if (('type' in p and p['type'].lower() == 'write') or (p.get('is_read') is False))]
        write_voltages = [p['voltage'] for p in write_pulses]
        write_pulse_widths = [p['pulse_width'] for p in write_pulses]

        write_indices = [i for i, p in enumerate(pulses) if (('type' in p and p['type'].lower() == 'write') or (p.get('is_read') is False))]
        read_indices = [i for i, p in enumerate(pulses) if (('type' in p and p['type'].lower() == 'read') or (p.get('is_read') is True))]
        
        write_timestamps = [pulse_timestamps[i] for i in write_indices]
        read_timestamps = [pulse_timestamps[i] for i in read_indices[:len(read_currents)]] 

        # Combined arrays for MeasurementResult
        all_voltages = write_voltages + read_voltages
        all_pulse_widths = write_pulse_widths + read_pulse_widths
        all_timestamps = write_timestamps + read_timestamps
        all_is_read = [False] * len(write_voltages) + [True] * len(read_voltages)

        result = MeasurementResult(
            voltages=all_voltages,
            currents=read_currents,
            resistances=read_resistances,
            pulse_widths=all_pulse_widths,
            timestamps=all_timestamps,
            is_read=all_is_read,
            metadata={
                'low_pin': low_pin,
                'high_pin': high_pin,
                'sequence_length': len(pulses),
                'read_pulses': len(read_pulses),
                'write_pulses': len(write_pulses),
            }
        )
        result.metadata['write_voltages'] = write_voltages
        result.metadata['write_pulse_widths'] = write_pulse_widths
        return result
   
    ####### Legacy Code - kept for reference, but not used in current implementation #######
   
    # def create_pulse_configuration(self, 
    #                              low_pin: int,
    #                              high_pin: int,
    #                              sequence: Dict[str, Any],
    #                              read_voltage: float = 0.2,
    #                              read_pulse_offset: int = 340000,
    #                              reading_mode: str = "after_each", 
    #                              read_indices: List[int] = None,
    #                              read_interval: int = 1,
    #                              differential: bool = True,
    #                              debug: bool = False) -> PulseConfiguration:
    #     """Create a pulse configuration without executing it."""
    #     # Extract sequence and period from dictionary
    #     pulse_sequence = sequence['sequence']
    #     period = sequence['period']
        
    #     if not pulse_sequence:
    #         raise ValueError("Pulse sequence is empty")
        
    #     # Calculate pulse timing parameters
    #     max_width = max(width for _, width in pulse_sequence)
    #     write_read_delay = read_pulse_offset

    #     inter_pulse_delay = period - max_width - write_read_delay
    #     if inter_pulse_delay < 0:
    #         inter_pulse_delay = 0
    #         if debug:
    #             print("Warning: Period is too short. Setting inter_pulse_delay to 0.")
        
    #     # Create a list of indices where we should take readings
    #     read_at_indices = []
    #     if reading_mode == "after_each":
    #         read_at_indices = list(range(len(pulse_sequence)))
    #     elif reading_mode == "specific_indices":
    #         if read_indices:
    #             read_at_indices = [i for i in read_indices if 0 <= i < len(pulse_sequence)]
    #         else:
    #             if debug:
    #                 print("Warning: No read_indices provided for 'specific_indices' mode. No readings will be taken.")
    #     elif reading_mode == "interval":
    #         read_at_indices = list(range(0, len(pulse_sequence), read_interval))
    #     # For "none" mode, leave read_at_indices empty
        
    #     if debug:
    #         print(f"Creating configuration for {len(pulse_sequence)} pulses")
    #         print(f"Period: {period} ns")
    #         print(f"Max pulse width: {max_width} ns")
    #         print(f"Read pulse offset: {read_pulse_offset} ns")
    #         print(f"Inter-pulse delay: {inter_pulse_delay} ns")
    #         print(f"Reading mode: {reading_mode}")
    #         print(f"Taking readings at {len(read_at_indices)} positions: {read_at_indices}")
        
    #     # Create pulse operations for each pulse in the sequence
    #     operations = []
    #     for idx, (voltage, pulse_width) in enumerate(pulse_sequence):
    #         # hard coded compensation due to long settling time of pulse signal, here estimated for 1us pulses
    #         compensation = 1.33  # Base compensation factor
    #         abs_v = abs(voltage)
    #         if abs_v < 1.0:
    #             compensation += 0.4 * (1.0 - abs_v)  
    #         else:
    #             compensation -= 0.08 * (abs_v - 1.0)  
            
    #         # Create a pulse operation
    #         operation = PulseOperation(
    #             low_pin=low_pin,
    #             high_pin=high_pin,
    #             voltage=voltage,
    #             pulse_width=pulse_width,
    #             compensation=compensation,
    #             read_after=(idx in read_at_indices),
    #             read_voltage=read_voltage,
    #             differential=differential,
    #         )
    #         operations.append(operation)
        
    #     # Create and return the full configuration
    #     config = PulseConfiguration(
    #         operations=operations,
    #         period=period,
    #         write_read_delay=write_read_delay,
    #         inter_pulse_delay=inter_pulse_delay,
    #         read_at_indices=read_at_indices,
    #         metadata={
    #             'low_pin': low_pin,
    #             'high_pin': high_pin,
    #             'read_voltage': read_voltage,
    #             'sequence_length': len(pulse_sequence),
    #             'reading_mode': reading_mode
    #         }
    #     )
        
    #     return config
    
    # def execute_pulse_configuration(self, 
    #                              config: PulseConfiguration, 
    #                              compliance_current: float = 1e-3,
    #                              debug: bool = False) -> MeasurementResult:
    #     """Execute a pulse configuration and return measurement results."""
    #     if not config.operations:
    #         raise ValueError("Pulse configuration is empty")
        
    #     if debug:
    #         print(f"Executing configuration with {len(config.operations)} pulse operations")
    #         print(f"Taking readings at {len(config.read_at_indices)} positions")
        
    #     # Get pins from the first operation for consistency check
    #     low_pin = config.operations[0].low_pin
    #     high_pin = config.operations[0].high_pin
        
    #     # Determine cluster indices for the channels
    #     low_cluster = low_pin // 8
    #     high_cluster = high_pin // 8
        
    #     # If no readings are requested, use simpler execution path
    #     if not config.read_at_indices:
    #         if debug:
    #             print("No readings requested. Executing pulses only.")
                
    #         # For each pulse in the sequence
    #         for idx, op in enumerate(config.operations):
    #             if op.differential:
    #             # Configure channels for differential pulse
    #                 chan_configs = [
    #                     (op.low_pin, -op.voltage*op.compensation/2, 0.0),
    #                     (op.high_pin, op.voltage*op.compensation/2, 0.0)
    #                 ]
    #             else:
    #                 chan_configs = [
    #                     (op.low_pin, 0.0, 0.0),
    #                     (op.high_pin, op.voltage*op.compensation, 0.0)
    #                 ]

    #             # Configure cluster timings
    #             cluster_timings = [None] * 8
    #             cluster_timings[low_cluster] = op.pulse_width
    #             cluster_timings[high_cluster] = op.pulse_width
                
    #             # Apply pulse
    #             self.instrument.pulse_slice_fast_open(chan_configs, cluster_timings, True)
                
    #             # Add delay between pulses
    #             if idx < len(config.operations) - 1:
    #                 self.instrument.delay(config.period)
            
    #         # Execute all operations
    #         self.instrument.execute()
    #         self.instrument.wait()
            
    #         # Return empty result set
    #         return MeasurementResult(
    #             voltages=[],
    #             currents=[],
    #             resistances=[],
    #             pulse_widths=[],
    #             timestamps=[],
    #             metadata=config.metadata
    #         )
        
    #     # For configurations with readings
    #     pulse_indices = []  # Track which pulse indices will have readings
        
    #     # Calculate total pulse cycle time (write + read + inter-pulse)
    #     total_cycle_time = config.period

    #     # For each pulse, queue operations
    #     for idx, op in enumerate(config.operations):
    #         if op.differential:
    #         # Configure channels for differential pulse
    #             chan_configs = [
    #                 (op.low_pin, -op.voltage*op.compensation/2, 0.0),
    #                 (op.high_pin, op.voltage*op.compensation/2, 0.0)
    #             ]
    #         else:
    #             chan_configs = [
    #                 (op.low_pin, 0.0, 0.0),
    #                 (op.high_pin, op.voltage*op.compensation, 0.0)
    #             ]
                    
    #         # Configure cluster timings
    #         cluster_timings = [None] * 8
    #         cluster_timings[low_cluster] = op.pulse_width
    #         cluster_timings[high_cluster] = op.pulse_width
            
    #         # Apply pulse
    #         self.instrument.pulse_slice_fast_open(chan_configs, cluster_timings, True)
            
    #         # Take a reading if required for this pulse
    #         if op.read_after:
    #             # Add delay before reading
    #             self.instrument.delay(config.write_read_delay)
                
    #             # Read resistance
    #             self.instrument.generate_read_train(
    #                 lows=[op.low_pin],
    #                 highs=[op.high_pin],
    #                 vread=op.read_voltage,
    #                 nreads=1,
    #                 inter_nanos=0,
    #                 ground=True
    #             )
                
    #             # Store metadata for processing
    #             pulse_indices.append(idx)

    #             # Calculate remaining time in cycle to maintain equidistant timing
    #             # This ensures the time from write start to next write start is exactly period
    #             remaining_time = total_cycle_time - op.pulse_width - config.write_read_delay       
                
    #             # Add inter-pulse delay if not the last pulse
    #             if idx < len(config.operations) - 1 and remaining_time > 0:
    #                 self.instrument.delay(remaining_time)
    #         else:
    #             # Use full period delay if we didn't take a reading
    #             if idx < len(config.operations) - 1:
    #                 self.instrument.delay(config.period)
        
    #     # Execute all operations
    #     self.instrument.execute()
    #     self.instrument.wait()
        
    #     # Process results
    #     return self._process_measurement_results(
    #         config=config, 
    #         pulse_indices=pulse_indices, 
    #         compliance_current=compliance_current,
    #         debug=debug
    #     )

    # def _process_measurement_results(self, 
    #                              config: PulseConfiguration,
    #                              pulse_indices: List[int],
    #                              compliance_current: float = 1e-3,
    #                              debug: bool = False) -> MeasurementResult:
    #     if not pulse_indices:
    #         return MeasurementResult(
    #             voltages=[],
    #             currents=[],
    #             resistances=[],
    #             pulse_widths=[],
    #             timestamps=[],
    #             metadata=config.metadata
    #         )
        
    #     voltages = []
    #     pulse_widths = []
    #     currents = []
    #     resistances = []
    #     timestamps = []
        
    #     first_op = config.operations[0]
    #     high_pin = first_op.high_pin
        
    #     # Process each reading result
    #     read_count = 0
    #     for data in self.instrument.get_iter(DataMode.All):
    #         if read_count >= len(pulse_indices):
    #             break
                
    #         pulse_idx = pulse_indices[read_count]   # Get the pulse index where this reading happened

    #         read_count += 1
            
    #         op = config.operations[pulse_idx]
            
    #         curr_time = time.time()
    #         timestamps.append(curr_time)
            
    #         voltages.append(op.voltage)
    #         pulse_widths.append(op.pulse_width)
        
    #         current = data[0][high_pin]
    #         currents.append(current)
            
    #         # Calculate resistance
    #         if abs(current) > 1e-12:
    #             resistance = abs(op.read_voltage / current) # negative resistances makes no sense
    #         else:
    #             resistance = float('inf')
    #         resistances.append(resistance)
            
    #         if debug and (read_count == 1 or read_count == len(pulse_indices) or read_count % 10 == 0):
    #             print(f"Reading {read_count}/{len(pulse_indices)} (pulse #{pulse_idx+1}): "
    #                   f"V={op.voltage:.3f}V, I={current:.3e}A, R={resistance:.2e}Ω")
            
    #         # Check for compliance
    #         if abs(current) > compliance_current:
    #             print(f"Compliance current exceeded ({abs(current):.3e}A > {compliance_current:.3e}A). Stopping.")
    #             break
        
    #     # Create result object
    #     result = MeasurementResult(
    #         voltages=voltages,
    #         currents=currents,
    #         resistances=resistances,
    #         pulse_widths=pulse_widths,
    #         timestamps=timestamps,
    #         metadata={
    #             **config.metadata,
    #             'compliance_current': compliance_current,
    #             'pulse_indices': pulse_indices,
    #             'readings_processed': read_count
    #         },
    #         is_read=True,
    #     )
        
    #     return result

    # def apply_pulse_sequence(self, 
    #                        low_pin: int,
    #                        high_pin: int,
    #                        sequence: Dict[str, Any],
    #                        read_voltage: float = 0.2,
    #                        read_pulse_offset: int = 340000,
    #                        compliance_current: float = 1e-3,
    #                        reading_mode: str = "after_each", 
    #                        read_indices: List[int] = None,
    #                        read_interval: int = 1,
    #                        differential: bool = True,
    #                        debug: bool = False) -> MeasurementResult:
    #     """Apply a pulse sequence and take measurements."""
    #     # Create the configuration
    #     config = self.create_pulse_configuration(
    #         low_pin=low_pin,
    #         high_pin=high_pin,
    #         sequence=sequence,
    #         read_voltage=read_voltage,
    #         read_pulse_offset=read_pulse_offset,
    #         reading_mode=reading_mode,
    #         read_indices=read_indices,
    #         read_interval=read_interval,
    #         differential=differential,
    #         debug=debug
    #     )
        
    #     # Execute the configuration
    #     return self.execute_pulse_configuration(
    #         config=config,
    #         compliance_current=compliance_current,
    #         debug=debug
    #     )

    # def apply_pulse_sequence_object(self, low_pin, high_pin, sequence_obj, compliance_current=1e-3, debug=False):
    #     """LEGACY: Process PulseSequence with arbitrary read/write operations and voltages"""
    #     # calibration constants since read and delay operations have overhead, measured on the osci
    #     ARC2_DELAY_OVERHEAD = 244_000  
    #     ARC2_READ_OVERHEAD = 100_000  
    
    #     sequence_data = sequence_obj.to_dict() if hasattr(sequence_obj, 'to_dict') else sequence_obj
        
    #     all_pulses = sequence_data.get('all_pulses', [])
    #     if not all_pulses:
    #         raise ValueError("Pulse sequence has no pulses")
        
    #     sorted_pulses = sorted(all_pulses, key=lambda p: p['time_offset'])
        
    #     if debug:
    #         print(f"Configuring sequence with {len(all_pulses)} pulses")
    #         write_pulses = [p for p in all_pulses if not p.get('is_read', False)]
    #         read_pulses = [p for p in all_pulses if p.get('is_read', False)]
    #         print(f"Write pulses: {len(write_pulses)}")
    #         print(f"Read pulses: {len(read_pulses)}")
        
    #     # Track read pulses and current time
    #     read_pulses = []
    #     current_time = 0
    #     differential = sequence_data.get('differential', True)
        
    #     # dynamicv Compensation logic due to long settling time, first-oder RC circuit V(t)=V_start + (V_0-V_start)*(1-exp(-t/tau))
    #     # linear fit for initial ramp based on measurements
    #     tau = 3100  # estimated from oscilloscope measurements in ns
    #     initial_fraction = 0.732
    #     delta_fraction = 1 - initial_fraction
    #     max_compensation = 2.0
    #     short_pulse_threshold = 2000 

    #     def linear_compensation(pulse_width):
    #         m = -0.000133
    #         b = 1.2766
    #         return m * pulse_width + b

    #     # order all operatoins using time_offset
    #     for pulse in sorted_pulses:
    #         is_read = pulse.get('is_read', False)
    #         pulse_width = int(pulse['pulse_width'])
    #         if not is_read:
    #             if pulse_width < short_pulse_threshold:
    #                 compensation = linear_compensation(pulse_width)
    #             else:
    #                 exp_factor = 1 - np.exp(-pulse_width / tau)
    #                 if exp_factor > 0:
    #                     compensation = initial_fraction + delta_fraction / exp_factor
    #                 else:
    #                     compensation = 1.0
    #             compensation = min(compensation, max_compensation)
    #             pulse['voltage'] = pulse['voltage'] * compensation
        
    #     for i, pulse in enumerate(sorted_pulses):
    #         pulse_time = pulse['time_offset']
            
    #         if pulse_time > current_time:
    #             delay_needed = pulse_time - current_time
                
    #             calibrated_delay = delay_needed - ARC2_DELAY_OVERHEAD
                
    #             if pulse.get('is_read', False):
    #                 calibrated_delay -= ARC2_READ_OVERHEAD
                
    #             # handle negative delays
    #             if calibrated_delay <= 0:
    #                 calibrated_delay = 0
                
    #             if debug:
    #                 print(f"Current time: {current_time/1000:.1f} μs")
    #                 print(f"Target time: {pulse_time/1000:.1f} μs") 
    #                 print(f"Raw delay needed: {delay_needed/1000:.1f} μs")
    #                 print(f"Calibrated delay: {calibrated_delay/1000:.1f} μs")
                
    #             if calibrated_delay > 0:
    #                 self.instrument.delay(calibrated_delay)

    #             actual_delay_applied = calibrated_delay + ARC2_DELAY_OVERHEAD
    #             if pulse.get('is_read', False):
    #                 actual_delay_applied += ARC2_READ_OVERHEAD
    #             current_time = current_time + actual_delay_applied
    #         else:
    #             current_time = pulse_time
            
    #         is_read = pulse.get('is_read', False)
    #         voltage = pulse['voltage']
    #         pulse_width = int(pulse['pulse_width'])
    #         compensation = pulse.get('compensation', 1.0)

    #         if debug:
    #             print(f"Processing pulse at {pulse_time/1000:.1f} μs: is_read={is_read}, voltage={voltage}V, width={pulse_width/1000:.1f} μs")
            
    #         # Skip 0V write pulses, add delay centering the read operation
    #         if not is_read and voltage == 0:
    #             if debug:
    #                 print(f"  Skipping 0V write pulse - calculating delay to center read")
                
    #             period = sequence_data.get('period')
    #             read_pulse_width = sequence_data.get('default_read_pulse_width',530_000)
                
    #             centering_delay = (period - read_pulse_width) // 2
                
    #             if debug:
    #                 print(f"  Period: {period/1000:.1f} μs")
    #                 print(f"  Read pulse width: {read_pulse_width/1000:.1f} μs") 
    #                 print(f"  Centering delay: {centering_delay/1000:.1f} μs")
                
    #             if centering_delay > 0:
    #                 self.instrument.delay(centering_delay)
    #                 current_time += centering_delay
    #             else:
    #                 current_time = pulse_time
    #             continue

    #         if is_read:
    #             self._execute_read_pulse(high_pin, voltage, pulse, read_pulses, debug)
    #         else:
    #             self._execute_write_pulse(low_pin, high_pin, voltage, pulse_width, differential, debug)
            

    #         current_time += pulse_width
    #         if debug:
    #             print(f"After pulse, current time: {current_time/1000:.1f} μs")



    #     # Execute all configured operations
    #     if debug:
    #         print(f"Executing batch of {len(sorted_pulses)} operations...")
        
    #     self.instrument.execute()
    #     self.instrument.wait()
        
    #     # Collect measurements from read operations
    #     read_voltages = []
    #     read_currents = []
    #     read_resistances = []
    #     read_pulse_widths = []
    #     read_timestamps = []
        
    #     # Get all readings from execution
    #     if debug:
    #         print(f"Collecting {len(read_pulses)} readings...")

    #     try:
    #         # First check if there are any readings at all
    #         reading_data = list(self.instrument.get_iter(DataMode.All))
            
    #         if debug:
    #             print(f"Retrieved {len(reading_data)} readings from hardware")
    #             if reading_data:
    #                 print(f"First reading data structure: {reading_data[0]}")
            
    #         for i, data in enumerate(reading_data):
    #             if i >= len(read_pulses):
    #                 break
                
    #             # Get the corresponding read pulse
    #             read_pulse = read_pulses[i]
    #             read_voltage = read_pulse['voltage']
                
    #             try:
    #                 # Try accessing the current with more robust error handling
    #                 current = None
                    
    #                 # Try different ways to access the data
    #                 if isinstance(data, dict) and 'current' in data:
    #                     current = data['current']
    #                 elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
    #                     current = data[0].get(high_pin)
    #                 elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
    #                     current = data[0][high_pin]
    #                 else:
    #                     # Try to inspect the data
    #                     if debug:
    #                         print(f"  Data structure: {data}")
                            
    #                     # Use a fallback method
    #                     current = data[0][high_pin]
                    
    #                 # Process reading
    #                 read_voltages.append(read_voltage)
    #                 read_currents.append(current)
    #                 read_resistances.append(read_voltage / current if abs(current) > 1e-12 else float('inf'))
    #                 read_pulse_widths.append(read_pulse['pulse_width'])
    #                 read_timestamps.append(time.time())
                    
    #                 if debug:
    #                     print(f"  Reading {i+1}/{len(read_pulses)}: V={read_voltage}V, I={current:.3e}A, R={read_resistances[-1]:.2e}Ω")
                
    #             except Exception as e:
    #                 if debug:
    #                     print(f"  Error processing reading {i+1}: {e}")
    #                     print(f"  Data structure: {data}")

    #     except Exception as e:
    #         if debug:
    #             print(f"Error collecting readings: {e}")
        
    #     # Now create combined arrays including both write and read pulses
    #     # Extract write pulses for the arrays
    #     write_pulses = [p for p in sorted_pulses if not p.get('is_read', False)]
    #     write_voltages = [p['voltage'] for p in write_pulses]
    #     write_pulse_widths = [p['pulse_width'] for p in write_pulses]
    #     write_timestamps = [time.time() - (len(read_timestamps) * 0.001) for _ in range(len(write_pulses))]
        
    #     # Combined arrays for MeasurementResult
    #     all_voltages = write_voltages + read_voltages
    #     all_pulse_widths = write_pulse_widths + read_pulse_widths
    #     all_timestamps = write_timestamps + read_timestamps
    #     all_is_read = [False] * len(write_voltages) + [True] * len(read_voltages)
        
    #     # Create measurement result with both read and write operations
    #     result = MeasurementResult(
    #         voltages=all_voltages,
    #         currents=read_currents,  # Only read operations have current measurements
    #         resistances=read_resistances,  # Only read operations have resistance measurements
    #         pulse_widths=all_pulse_widths,
    #         timestamps=all_timestamps,
    #         is_read=all_is_read,
    #         metadata={
    #             'low_pin': low_pin,
    #             'high_pin': high_pin,
    #             'sequence_length': len(all_pulses),
    #             'read_pulses': len(read_pulses),
    #             'write_pulses': len(write_pulses),
    #             'compliance_current': compliance_current
    #         }
    #     )
        
    #     # Store write voltage and pulse width arrays separately for backward compatibility
    #     result.metadata['write_voltages'] = write_voltages
    #     result.metadata['write_pulse_widths'] = write_pulse_widths
        
    #     # Also store pulse metadata by type
    #     if 'write' in sequence_data:
    #         result.metadata['write_metadata'] = sequence_data.get('write', {})
    #     if 'read' in sequence_data:
    #         result.metadata['read_metadata'] = sequence_data.get('read', {})
        
    #     return result
    


        
    # def configure_custom_sequence(self, config):
    #     """
    #     Configure a custom pulse sequence with arbitrary pulses and timings using period
    #     """
    #     voltages = config.get('voltages', [])
    #     pulse_widths = config.get('pulse_widths', [])
    #     read_after = config.get('read_after', [True] * len(voltages))
    #     period = config.get('period', 1_000_000)
    #     read_pulse_width = 530_000  # Fixed hardware constraint
    #     read_voltage = config.get('read_voltage', 0.2)
    #     differential = config.get('differential', True)
  
    #     seq = PulseSequence(
    #         period=period,
    #         default_read_voltage=read_voltage,
    #         default_read_pulse_width=read_pulse_width,
    #         default_read_pulse_offset=340_000,
    #         differential= differential
    #     )
        
    #     # Build the sequence using simple approach - let PulseSequence handle timing
    #     for i, (voltage, width) in enumerate(zip(voltages, pulse_widths)):
    #         # Just add the write pulse
    #         seq.add_write_pulse(voltage=voltage, pulse_width=width)
            
    #         # Add read pulse if needed
    #         if i < len(read_after) and read_after[i]:
    #             seq.add_read_pulse(voltage=read_voltage)
        
    #     return seq