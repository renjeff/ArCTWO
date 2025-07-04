from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from .pulse_sequence import PulseSequence

def create_ltp_ltd_sequence_new(
    read_voltage=0.1,
    ltp_voltages=None,
    ltd_voltages=None,
    pulse_width_ns=10000,
    read_delay_ns=100000,
    write_delay_ns=50000,
    inter_phase_delay_ns=500000
):
    """
    Create a potentiation/depression (LTP/LTD) sequence
    
    Parameters
    ----------
    read_voltage : float
        Voltage used for read pulses
    ltp_voltages : array-like
        Array of voltage levels for potentiation phase
    ltd_voltages : array-like
        Array of voltage levels for depression phase
    pulse_width_ns : int
        Duration of write pulses in nanoseconds
    read_delay_ns : int
        Delay after read pulses in nanoseconds
    write_delay_ns : int
        Delay after write pulses in nanoseconds
    inter_phase_delay_ns : int
        Delay between LTP and LTD phases in nanoseconds
        
    Returns
    -------
    voltages, types, pulse_widths, delays : tuple of lists
        Lists ready for build_relative_pulse_sequence
    """
    # Default voltage arrays if not provided
    if ltp_voltages is None:
        ltp_voltages = np.linspace(0.8, 2.0, 10)
    if ltd_voltages is None:
        ltd_voltages = np.linspace(-0.8, -2.0, 10)
    
    # Create sequence arrays
    voltages = []
    types = []
    pulse_widths = []
    delays = []

    # Initial read
    voltages.append(read_voltage)
    types.append('read')
    pulse_widths.append(None)
    delays.append(read_delay_ns)

    # LTP (potentiation) sequence
    for v in ltp_voltages:
        # Write pulse
        voltages.append(v)
        types.append('write')
        pulse_widths.append(pulse_width_ns)
        delays.append(write_delay_ns)
        
        # Read pulse
        voltages.append(read_voltage)
        types.append('read')
        pulse_widths.append(None)
        delays.append(read_delay_ns)

    # Wait between LTP and LTD
    if len(delays) > 0:
        delays[-1] = inter_phase_delay_ns

    # LTD (depression) sequence
    for v in ltd_voltages:
        # Write pulse
        voltages.append(v)
        types.append('write')
        pulse_widths.append(pulse_width_ns)
        delays.append(write_delay_ns)
        
        # Read pulse
        voltages.append(read_voltage)
        types.append('read')
        pulse_widths.append(None)
        delays.append(read_delay_ns)

    return voltages, types, pulse_widths, delays


def create_retention_sequence(
    set_voltage=1.8,
    read_voltage=0.1,
    set_pulse_width_ns=10000,
    read_times_sec=None,
    read_pulse_width_ns=350_000,
):
    """
    Create a retention test sequence
    
    Parameters
    ----------
    set_voltage : float
        Voltage used to SET the device to its target state
    read_voltage : float
        Voltage used for read pulses
    set_pulse_width_ns : int
        Duration of SET pulse in nanoseconds
    read_times_sec : array-like
        Array of times (in seconds) at which to read the device state
    read_pulse_width_ns : int
        Duration of read pulses in nanoseconds (default 100,000 ns)
        
    Returns
    -------
    voltages, types, pulse_widths, delays : tuple of lists
        Lists ready for build_relative_pulse_sequence
    """
    # Default read times if not provided
    if read_times_sec is None:
        read_times_sec = [0.001, 0.01, 0.1, 1, 5, 10, 30, 60]
    
    # Convert read times to nanoseconds
    read_times_ns = [int(t * 1e9) for t in read_times_sec]
    
    # Create sequence arrays
    voltages = []
    types = []
    pulse_widths = []
    delays = []

    # Initial SET pulse
    voltages.append(set_voltage)
    types.append('write')
    pulse_widths.append(set_pulse_width_ns)
    
    # First delay (after SET pulse to first read)
    first_read_delay = read_times_ns[0] - set_pulse_width_ns
    delays.append(max(0, first_read_delay))
    
    # Add reads at specified times
    for i, read_time_ns in enumerate(read_times_ns):
        # Add read pulse - use explicit width instead of None
        voltages.append(read_voltage)
        types.append('read')
        pulse_widths.append(read_pulse_width_ns)  # Explicit width instead of None
        
        # Calculate delay to next read (if any)
        if i < len(read_times_ns) - 1:
            # Subtract read pulse width to maintain correct timing
            next_delay = read_times_ns[i+1] - read_times_ns[i] - read_pulse_width_ns
            delays.append(max(0, next_delay))
        else:
            # Final delay
            delays.append(0)

    return voltages, types, pulse_widths, delays

def create_endurance_sequence(
    num_cycles=100,
    set_voltage=1.6,
    reset_voltage=-1.8,
    read_voltage=0.1,
    pulse_width_ns=10000,
    std_delay_ns=50000,
    read_interval=10  # Read every N cycles
):
    """
    Create an endurance test sequence
    
    Parameters
    ----------
    num_cycles : int
        Number of SET/RESET cycles to perform
    set_voltage : float
        Voltage used for SET operations
    reset_voltage : float
        Voltage used for RESET operations
    read_voltage : float
        Voltage used for read operations
    pulse_width_ns : int
        Duration of write pulses in nanoseconds
    std_delay_ns : int
        Standard delay between operations in nanoseconds
    read_interval : int
        Read every N cycles (to reduce test time)
        
    Returns
    -------
    voltages, types, pulse_widths, delays : tuple of lists
        Lists ready for build_relative_pulse_sequence
    """
    # Create sequence arrays and add inital read
    voltages = [read_voltage]
    types = ['read'] 
    pulse_widths = [None]
    delays = [std_delay_ns]

    # Build endurance cycling sequence
    for cycle in range(num_cycles):
        # Only read on specific cycles to reduce test time
        read_this_cycle = (cycle < 5 or 
                        cycle % read_interval == 0 or 
                        cycle >= num_cycles-5)
        
        # SET operation
        voltages.append(set_voltage)
        types.append('write')
        pulse_widths.append(pulse_width_ns)
        delays.append(std_delay_ns)
        
        # Read after SET (optional)
        if read_this_cycle:
            voltages.append(read_voltage)
            types.append('read')
            pulse_widths.append(None)
            delays.append(std_delay_ns)
        
        # RESET operation
        voltages.append(reset_voltage)
        types.append('write')
        pulse_widths.append(pulse_width_ns)
        delays.append(std_delay_ns)
        
        # Read after RESET (optional)
        if read_this_cycle:
            voltages.append(read_voltage)
            types.append('read')
            pulse_widths.append(None)
            delays.append(std_delay_ns)

    return voltages, types, pulse_widths, delays

# def create_periodic_sequence(
#     write_voltage: float,
#     write_pulse_width: int,
#     write_count: int,
#     write_period_off: int = 1000000,
#     read_pattern: str = "after_each",  # Options: "after_each", "custom_times", "before_after", "none"
#     read_times: list = None,           # For custom times
#     read_indices: list = None,         # For specific indices
#     read_interval: int = 1,            # For interval pattern
#     read_voltage: float = 0.2,
#     read_pulse_width: int = 100000,
#     read_pulse_offset: int = 340000,
#     differential: bool = True
# ) -> PulseSequence:
#     """
#     Create a sequence with periodic writes and flexible read patterns.
    
#     Args:
#         write_voltage: Pulse voltage
#         write_pulse_width: Pulse width in nanoseconds
#         write_count: Number of pulses to generate
#         write_period: Time between pulses in nanoseconds
#         read_pattern: When to take readings ("after_each", "interval", "custom_times", "specific_indices", "none")
#         read_times: List of specific times for reads (for "custom_times" mode)
#         read_indices: List of specific pulse indices to read after (for "specific_indices" mode)
#         read_interval: For "interval" mode, take reading every N pulses
#         read_voltage: Voltage used for read operations
#         read_pulse_width: Width of read pulses in nanoseconds
#         read_pulse_offset: Offset from write pulse to read pulse in nanoseconds
#         differential: Whether to use differential signaling for pulses
        
#     Returns:
#         PulseSequence object
#     """
#     sequence = PulseSequence(
#         period=write_period_off,
#         default_read_voltage=read_voltage,
#         default_read_pulse_width=read_pulse_width,
#         default_read_pulse_offset=read_pulse_offset,
#         differential=differential
#     )
    
#     write_period = write_period_off - write_pulse_width 

#     # Add write pulses
#     for i in range(write_count):
#         time_offset = i * write_period
#         sequence.add_write_pulse(time_offset, write_voltage, write_pulse_width)
        
#     # Add read pulses according to pattern
#     if read_pattern == "after_each":
#         for i in range(write_count):
#             time_offset = i * write_period + read_pulse_offset
#             sequence.add_read_pulse(time_offset)
            
#     elif read_pattern == "interval":
#         for i in range(0, write_count, read_interval):
#             time_offset = i * write_period + read_pulse_offset
#             sequence.add_read_pulse(time_offset)
            
#     elif read_pattern == "custom_times":
#         if read_times:
#             for read_time in read_times:
#                 sequence.add_read_pulse(read_time)
                
#     elif read_pattern == "specific_indices":
#         if read_indices:
#             for idx in read_indices:
#                 if 0 <= idx < write_count:
#                     time_offset = idx * write_period + read_pulse_offset
#                     sequence.add_read_pulse(time_offset)
                
#     elif read_pattern == "before_after":
#         # Add read before the first write
#         sequence.add_read_pulse(0)
        
#         # Add reads after each write
#         for i in range(write_count):
#             time_offset = i * write_period + read_pulse_offset
#             sequence.add_read_pulse(time_offset)
            
#         # Add read after the last write plus an additional period
#         sequence.add_read_pulse(write_count * write_period + read_pulse_offset)
    
#     return sequence

# def create_ltp_ltd_sequence_enhanced(
#     nb_ltp: int,
#     nb_ltd: int,
#     vwrite_ltp_start: float,
#     vwrite_ltp_end: float,
#     vwrite_ltd_start: float,
#     vwrite_ltd_end: float,
#     twrite_ltp_start: int,
#     twrite_ltp_end: int,
#     twrite_ltd_start: int,
#     twrite_ltd_end: int,
#     period: int = 1000000,
#     read_voltage: float = 0.2,
#     read_pulse_width: int = 100000,
#     read_pulse_offset: int = 340000,
#     differential: bool = True
# ) -> PulseSequence:
#     """
#     LEGACY CODE
#     Create an LTP/LTD pulse sequence with integrated read operations.
#     Enhanced version using the PulseSequence class.
    
#     Args:
#         nb_ltp: Number of LTP pulses
#         nb_ltd: Number of LTD pulses
#         vwrite_ltp_start: Starting voltage for LTP pulses
#         vwrite_ltp_end: Ending voltage for LTP pulses
#         vwrite_ltd_start: Starting voltage for LTD pulses
#         vwrite_ltd_end: Ending voltage for LTD pulses
#         twrite_ltp_start: Starting pulse width for LTP pulses
#         twrite_ltp_end: Ending pulse width for LTP pulses
#         twrite_ltd_start: Starting pulse width for LTD pulses
#         twrite_ltd_end: Ending pulse width for LTD pulses
#         period: Time between pulses in nanoseconds
#         read_voltage: Voltage used for read operations
#         read_pulse_width: Width of read pulses in nanoseconds
#         read_pulse_offset: Offset from write pulse to read pulse
#         differential: Whether to use differential signaling
    
#     Returns:
#         PulseSequence object
#     """
#     sequence = PulseSequence(
#         period=period,
#         default_read_voltage=read_voltage,
#         default_read_pulse_width=read_pulse_width,
#         default_read_pulse_offset=read_pulse_offset,
#         differential=differential
#     )
    
#     # Calculate voltage and pulse width steps
#     vwrite_ltp_step = (vwrite_ltp_end - vwrite_ltp_start) / max(1, nb_ltp - 1) if nb_ltp > 1 else 0
#     vwrite_ltd_step = (vwrite_ltd_end - vwrite_ltd_start) / max(1, nb_ltd - 1) if nb_ltd > 1 else 0
#     twrite_ltp_step = (twrite_ltp_end - twrite_ltp_start) / max(1, nb_ltp - 1) if nb_ltp > 1 else 0
#     twrite_ltd_step = (twrite_ltd_end - twrite_ltd_start) / max(1, nb_ltd - 1) if nb_ltd > 1 else 0
    

#     # Add LTP pulses with reads
#     for i in range(nb_ltp):
#         vwrite = vwrite_ltp_start + i * vwrite_ltp_step
#         twrite = int(twrite_ltp_start + i * twrite_ltp_step)
#         sequence.add_write_with_read(vwrite, twrite, read_voltage)
        
#     # Fix the LTD pulses section 
#     ltp_duration = nb_ltp * period
#     for i in range(nb_ltd):
#         vwrite = vwrite_ltd_start + i * vwrite_ltd_step
#         twrite = int(twrite_ltd_start + i * twrite_ltd_step)
#         sequence.add_write_with_read(vwrite, twrite, read_voltage)
        
#     return sequence