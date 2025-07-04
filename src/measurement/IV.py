from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import numpy as np
import traceback
from datetime import datetime
from pyarc2 import DataMode

from .base import BaseMeasurement, MeasurementResult

@dataclass
class IVSweepConfiguration:
    """Configuration for IV curve sweep measurements."""
    row_pin: int
    column_pin: int
    voltage_range: Tuple[float, float] = (-1.0, 1.0)
    steps: int = 200
    sweep_type: str = 'full'  # 'full', 'hysteresis', 'bipolar', 'full_double'
    reset_voltage: Optional[float] = None
    read_delay_ns: int = 0
    compliance_current: float = 1e-3
    metadata: Dict[str, Any] = field(default_factory=dict)

class IVCurveMeasurement(BaseMeasurement):
    """
    Class for IV curve measurements using ArC TWO.
    """
    
    def __init__(self, instrument):
        """Initialize with an ArC Two instrument instance."""
        super().__init__(instrument)
    
    def measure_iv_curve(self, 
                      config: Union[IVSweepConfiguration, Dict],
                      debug: bool = False) -> MeasurementResult:
        """
        Perform an IV curve measurement using the gapless staircase method.
        
        Parameters:
        -----------
        config : IVSweepConfiguration or Dict
            Configuration for the IV sweep
        debug : bool
            Whether to print debug information
            
        Returns:
        --------
        MeasurementResult
            Container with measurement results
        """
        # Handle dictionary input for backward compatibility
        if isinstance(config, dict):
            config = IVSweepConfiguration(
                row_pin=config.get('row_pin'),
                column_pin=config.get('column_pin'),
                voltage_range=config.get('voltage_range', (-1.0, 1.0)),
                steps=config.get('steps', 200),
                sweep_type=config.get('sweep_type', 'full'),
                reset_voltage=config.get('reset_voltage'),
                read_delay_ns=config.get('read_delay_ns', 1000),
                compliance_current=config.get('compliance_current', 1e-3),
                metadata=config
            )
        
        min_v, max_v = config.voltage_range
        low_chan = config.column_pin  # Low channel is column
        high_chan = config.row_pin    # High channel is row
        
        # Define segment sequences based on sweep_type
        segments = []
        num_segments = 0

        if config.sweep_type == 'full':
            num_segments = 4
        elif config.sweep_type == 'full_double':
            num_segments = 8
        elif config.sweep_type == 'hysteresis':
            num_segments = 2
        elif config.sweep_type == 'bipolar':
            num_segments = 4
        else:
            num_segments = 4
        
        steps_per_segment = max(2, int(np.ceil(config.steps / num_segments)))

        if config.sweep_type == 'full':
            # Full characterization cycle: 0 → max → 0 → min → 0
            segments.append(np.linspace(0, max_v, steps_per_segment))         # 0 → max
            segments.append(np.linspace(max_v, 0, steps_per_segment)[1:])      # max → 0
            segments.append(np.linspace(0, min_v, steps_per_segment)[1:])      # 0 → min
            segments.append(np.linspace(min_v, 0, steps_per_segment)[1:])      # min → 0
            
        elif config.sweep_type == 'full_double':
            # Double full characterization cycle
            segments.append(np.linspace(0, max_v, steps_per_segment))         # 0 → max
            segments.append(np.linspace(max_v, 0, steps_per_segment)[1:])      # max → 0
            segments.append(np.linspace(0, min_v, steps_per_segment)[1:])      # 0 → min
            segments.append(np.linspace(min_v, 0, steps_per_segment)[1:])      # min → 0
            segments.append(np.linspace(0, max_v, steps_per_segment)[1:])      # 0 → max
            segments.append(np.linspace(max_v, 0, steps_per_segment)[1:])      # max → 0
            segments.append(np.linspace(0, min_v, steps_per_segment)[1:])      # 0 → min
            segments.append(np.linspace(min_v, 0, steps_per_segment)[1:])      # min → 0

        elif config.sweep_type == 'hysteresis':
            # Traditional hysteresis loop: min → max → min
            segments.append(np.linspace(min_v, max_v, steps_per_segment))     # min → max
            segments.append(np.linspace(max_v, min_v, steps_per_segment)[1:]) # max → min
            
        elif config.sweep_type == 'bipolar':
            # Bipolar sweep: min → 0 → max → 0 → min
            segments.append(np.linspace(min_v, 0, steps_per_segment))         # min → 0
            segments.append(np.linspace(0, max_v, steps_per_segment)[1:])      # 0 → max
            segments.append(np.linspace(max_v, 0, steps_per_segment)[1:])      # max → 0
            segments.append(np.linspace(0, min_v, steps_per_segment)[1:])      # 0 → min
        
        else:
            if debug:
                print(f"Unknown sweep type '{config.sweep_type}'. Using default full pattern.")
            segments.append(np.linspace(0, max_v, steps_per_segment))         # 0 → max
            segments.append(np.linspace(max_v, 0, steps_per_segment)[1:])      # max → 0
            segments.append(np.linspace(0, min_v, steps_per_segment)[1:])      # 0 → min
            segments.append(np.linspace(min_v, 0, steps_per_segment)[1:])      # min → 0
        
        if debug:
            # Display configuration summary
            print("\n--- IV Sweep Configuration Summary ---")
            print(f"Sweep type: {config.sweep_type}")
            print(f"Voltage range: {min_v:.3f}V to {max_v:.3f}V")
            print(f"Total steps: {config.steps} per segment")
            print(f"Number of segments: {len(segments)}")
            for i, segment in enumerate(segments):
                print(f"  Segment {i+1}: {segment[0]:.3f}V → {segment[-1]:.3f}V ({len(segment)} points)")
            print(f"Read delay: {config.read_delay_ns} ns")
            print(f"Compliance current: {config.compliance_current:.3e} A")
            if config.reset_voltage is not None:
                print(f"Reset voltage: {config.reset_voltage} V")
            print(f"Channels: Low={low_chan}, High={high_chan}")
        
        # Result containers
        voltages = []
        currents = []
        resistances = []
        timestamps = []
        pulse_widths = []  # For compatibility with MeasurementResult
        
        arc = self.instrument
        
        try:
            # Ground all pins for safety
            arc.ground_all()
            arc.execute()
            arc.wait()
            
            # # Apply reset voltage if specified
            # if config.reset_voltage is not None:
            #     if debug:
            #         print(f"Applying reset voltage: {config.reset_voltage} V")
            #     try:
            #         if config.reset_voltage >= 0:
            #             arc.pulse_one(low_chan, high_chan, config.reset_voltage, 100000).execute()  # 100ms pulse
            #         else:
            #             arc.pulse_one(high_chan, low_chan, abs(config.reset_voltage), 100000).execute()
            #         # Wait for stabilization
            #         time.sleep(0.1)
            #         arc.ground_all().execute()
            #         time.sleep(0.1)
            #     except Exception as reset_error:
            #         if debug:
            #             print(f"Error applying reset voltage: {reset_error}")
            
            if debug:
                print(f"Starting sweep using {config.sweep_type} pattern with {len(segments)} segments")
            
            # Process each segment
            for segment_idx, segment_voltages in enumerate(segments):
                if debug:
                    print(f"Processing segment {segment_idx+1}/{len(segments)}: {segment_voltages[0]:.3f}V → {segment_voltages[-1]:.3f}V")
                
                start_time = time.time()
                
                # Determine if this segment has mixed polarities
                has_pos = any(v > 0 for v in segment_voltages)
                has_neg = any(v < 0 for v in segment_voltages)
                mixed_polarity = has_pos and has_neg
                
                # Process voltages one by one instead of batching if mixed polarity
                if mixed_polarity:
                    # Process each voltage individually for mixed polarity segments
                    for v_idx, v in enumerate(segment_voltages):
                        # Ground all pins between polarities 
                        # arc.ground_all()  # grounding after each voltage gives better results for single resistor
                
                        # Configure for this single voltage
                        if v >= 0:
                            arc.config_channels([(low_chan, v)], None)
                            arc.read_slice_open_deferred([high_chan], True)
                        else:
                            arc.config_channels([(high_chan, -v)], None)
                            arc.read_slice_open_deferred([low_chan], True)
                        
                        # Optional tiny delay
                        if config.read_delay_ns > 0:
                            arc.delay(config.read_delay_ns)
                        
                        # Execute for single voltage
                        arc.execute()
                        
                        # Get the single reading
                        data_iter = list(arc.get_iter(DataMode.All))
                        if data_iter:
                            data = data_iter[0]  
                        
                        voltage_value = v
                        voltages.append(voltage_value)
                        timestamps.append(time.time())
                        pulse_widths.append(0)
                        
                        # Get current for the appropriate channel
                        if v >= 0:
                            current_value = data[0][high_chan]
                        else:
                            current_value = -data[0][low_chan]
                        
                        currents.append(current_value)
                        
                        # Calculate resistance
                        if abs(current_value) > 1e-12:
                            resistance_value = abs(voltage_value / current_value)
                        else:
                            resistance_value = float('inf')
                        resistances.append(resistance_value)
                        
                        if debug:
                            print(f"V={voltage_value:.3f}V, I={current_value:.3e}A, R={resistance_value:.2e}Ω")
                else:
                    # For uniform polarity, batch commands
                    # Queue all operations without intermediate grounding for this segment
                    for v in segment_voltages:
                        # Configure channels for the voltage (direction depends on sign)
                        if v >= 0:
                            arc.config_channels([(low_chan, v)], None)
                            arc.read_slice_open_deferred([high_chan], True)
                        else:
                            arc.config_channels([(high_chan, -v)], None)
                            arc.read_slice_open_deferred([low_chan], True)
                        
                        # Optional tiny delay between reads if needed for stability
                        if config.read_delay_ns > 0:
                            arc.delay(config.read_delay_ns)
                    
                    # Execute the batched commands for this segment
                    arc.execute()
         
                    # Use get_iter to retrieve results
                    for idx, data in enumerate(arc.get_iter(DataMode.All)):
                        if idx < len(segment_voltages):
                            voltage_value = segment_voltages[idx]
                            voltages.append(voltage_value)
                            timestamps.append(time.time())
                            pulse_widths.append(0)
                            
                            # Get current for the appropriate channel
                            if segment_voltages[idx] >= 0:
                                current_value = data[0][high_chan]
                            else:
                                current_value = -data[0][low_chan]
                                
                            currents.append(current_value)
                            
                            # Calculate resistance
                            if abs(current_value) > 1e-12:
                                resistance_value = abs(voltage_value / current_value)
                            else:
                                resistance_value = float('inf')
                            resistances.append(resistance_value)
                    
                            if debug:
                                print(f"V={voltage_value:.3f}V, I={current_value:.3e}A, R={resistance_value:.2e}Ω")
                
                if debug:
                    segment_time = time.time() - start_time
                    print(f"Segment {segment_idx+1} completed in {segment_time:.3f}s")
            
        except Exception as e:
            if debug:
                print(f"Unexpected error during IV sweep: {e}")
                print(traceback.format_exc())
            raise
        
        finally:
            # Ensure device is grounded after measurement
            try:
                arc.ground_all().execute()
                if debug:
                    print("Device grounded after measurement")
            except Exception as ground_error:
                if debug:
                    print(f"Error grounding device: {ground_error}")
        
        # Create metadata dictionary
        metadata = {
            "row_pin": high_chan,
            "column_pin": low_chan,
            "sweep_type": config.sweep_type,
            "voltage_range": config.voltage_range,
            "steps": config.steps,
            "read_delay_ns": config.read_delay_ns,
            "compliance_current": config.compliance_current,
            "reset_voltage": config.reset_voltage,
            "measurement_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "measurement_type": "iv_curve",
            **config.metadata
        }
        
        # Return measurement results
        return MeasurementResult(
            voltages=voltages,
            currents=currents,
            resistances=resistances,
            pulse_widths=pulse_widths,
            timestamps=timestamps,
            metadata=metadata,
            is_read=True 
        )
    
    def sweep(self, 
            row_pin: int, 
            column_pin: int,
            sweep_type: str = 'full',
            voltage_range: Optional[Tuple[float, float]] = None,
            max_voltage: Optional[float] = None,
            steps: int = 21,
            compliance_current: float = 1e-3,
            read_delay_ns: int = 1000,
            debug: bool = False) -> MeasurementResult:
        """
        Perform an IV sweep with configurable pattern.
        
        Parameters:
        -----------
        sweep_type: str
            'full', 'full_double', 'hysteresis', or 'bipolar'
        voltage_range: Optional[Tuple[float, float]]
            Min and max voltage (required for 'full' and 'bipolar')
        max_voltage: Optional[float]
            Maximum voltage amplitude for symmetric sweeps (required for 'hysteresis')
        """
        # Handle different voltage specifications based on sweep type
        if sweep_type == 'hysteresis' and max_voltage is not None:
            v_range = (-abs(max_voltage), abs(max_voltage))
        elif voltage_range is not None:
            v_range = voltage_range
        else:
            raise ValueError("Must provide either voltage_range or max_voltage based on sweep type")

        # Create configuration
        config = IVSweepConfiguration(
            row_pin=row_pin,
            column_pin=column_pin,
            voltage_range=v_range,
            steps = steps,
            sweep_type=sweep_type,
            compliance_current=compliance_current,
            read_delay_ns=read_delay_ns
        )
        
        # Perform measurement
        return self.measure_iv_curve(config, debug=debug)