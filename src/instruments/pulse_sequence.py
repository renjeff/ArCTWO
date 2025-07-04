from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from ..utils.hardware_utils import get_pin_cluster, get_cluster_timing

class PulseSequence:
    """Base class for pulse sequences """
    
    def __init__(self, 
                 period: int = 1000000,
                 default_read_voltage: float = 0.2,
                 default_read_pulse_width: int = 525_000,
                 default_read_pulse_offset: int = 340000,
                 differential: bool = True):
        """
        Args:
            period: Default delay between consecutive pulses in nanoseconds
            default_read_voltage: Default voltage used for read operations
            default_read_pulse_width: Default width of read pulses in nanoseconds
            default_read_pulse_offset: Default offset from write pulse to read
            differential: Whether to use differential signaling for pulses
        """
        self.period = period
        self.default_read_voltage = default_read_voltage
        self.default_read_pulse_width = default_read_pulse_width
        self.default_read_pulse_offset = default_read_pulse_offset
        self.differential = differential
        
        # Storage for sequence elements with relative delays
        self.pulses = []
        self.next_id = 0
    
    @staticmethod
    def build_relative_pulse_sequence(voltages, types, pulse_widths, delays, read_voltage=0.2, differential=True, default_read_pulse_width=530_000):
        """
        Build a PulseSequence using parallel lists and the add_write_pulse/add_read_pulse methods.
        """
        
        seq = PulseSequence(
            period=0,  # Not used, required by constructor
            default_read_voltage=read_voltage,
            differential=differential
        )
        current_time = 0
        for v, t, w, d in zip(voltages, types, pulse_widths, delays):
            if t.lower() == 'write':
                pulse_id = seq.add_write_pulse(voltage=v, pulse_width=w, delay_after=d, differential=differential)
                width = w
            elif t.lower() == 'read':
                read_width = w if w not in (None, 0) else default_read_pulse_width
                pulse_id = seq.add_read_pulse(voltage=v, pulse_width=read_width, delay_after=d, differential=differential)
                width = read_width
            else:
                raise ValueError(f"Unknown pulse type: {t}")

            seq.pulses[-1]['time_offset'] = current_time
            current_time += width + d
        return seq
    
    def add_write_pulse(self, 
                        voltage: float, 
                        pulse_width: int,
                        delay_after: int = None,
                        differential: bool = None):    
        """
        Add a write pulse with a delay afterwards.
        
        Args:
            voltage: Pulse voltage
            pulse_width: Pulse width in nanoseconds
            delay_after: Delay after this pulse before the next one (default: use class period - pulse_width)
            differential: Whether to use differential signaling
        
        Returns:
            int: ID of the added pulse
        """
        if differential is None:
            differential = self.differential
        if delay_after is None:
            delay_after = max(0, self.period - pulse_width)
            
        pulse_id = self.next_id
        self.next_id += 1
        
        self.pulses.append({
            'id': pulse_id,
            'voltage': voltage,
            'pulse_width': pulse_width,
            'delay_after': delay_after,
            'is_read': False,
            'differential': differential
        })
        return pulse_id
            
    def add_read_pulse(self, 
                      voltage: float = None, 
                      pulse_width: int = None,
                      delay_after: int = None,
                      differential: bool = None): 
        """
        Add a read pulse with a delay afterwards.
        
        Args:
            voltage: Read voltage (default: use class default)
            pulse_width: Read pulse width (default: use class default)
            delay_after: Delay after this pulse before the next one (default: use class period - pulse_width)
            differential: Whether to use differential signaling
            
        Returns:
            int: ID of the added pulse
        """
        if voltage is None:
            voltage = self.default_read_voltage
        if pulse_width is None:
            pulse_width = self.default_read_pulse_width
        if differential is None:
            differential = self.differential
        if delay_after is None:
            delay_after = max(0, self.period - pulse_width)
            
        pulse_id = self.next_id
        self.next_id += 1
        
        self.pulses.append({
            'id': pulse_id,
            'voltage': voltage,
            'pulse_width': pulse_width,
            'delay_after': delay_after,
            'is_read': True,
            'differential': differential
        })
        return pulse_id
    
    def add_write_with_read(self, 
                           write_voltage: float,
                           write_pulse_width: int,
                           read_voltage: float = None, 
                           read_delay: int = None,
                           delay_after_read: int = None,
                           differential: bool = None):
        """
        Add a write pulse followed by a read pulse.
        
        Args:
            write_voltage: Write pulse voltage
            write_pulse_width: Write pulse width in nanoseconds
            read_voltage: Read pulse voltage (default: use class default)
            read_delay: Delay between end of write and start of read (default: use class default)
            delay_after_read: Delay after read before next pulse (default: use class period - read_pulse_width)
            differential: Whether to use differential signaling
            
        Returns:
            tuple: (write_pulse_id, read_pulse_id)
        """
        if read_voltage is None:
            read_voltage = self.default_read_voltage
        if read_delay is None:
            read_delay = self.default_read_pulse_offset - write_pulse_width
        if read_delay < 0:
            read_delay = 0
        if differential is None:
            differential = self.differential
        
        # Add write pulse with delay before read
        write_id = self.add_write_pulse(
            voltage=write_voltage,
            pulse_width=write_pulse_width,
            delay_after=read_delay,
            differential=differential
        )
        
        # Add read pulse with default delay after
        read_id = self.add_read_pulse(
            voltage=read_voltage,
            delay_after=delay_after_read,
            differential=differential
        )
        
        return write_id, read_id
    
    def calculate_absolute_times(self):
        """Calculate absolute time offsets with write pulses at exact period intervals"""
        sorted_pulses = sorted(self.pulses, key=lambda p: p['id'])
        absolute_times = {}
        
        # Find write-read pairs to understand the intended structure
        write_pulse_periods = {}  # Map write pulse id to its intended period
        current_period = 0
        
        for pulse in sorted_pulses:
            if not pulse['is_read']:
                # Each write pulse gets its own period
                write_pulse_periods[pulse['id']] = current_period
                absolute_times[pulse['id']] = current_period * self.period
                current_period += 1
            else:
                # Read pulses: find the most recent write pulse and position read equidistantly
                most_recent_write_id = None
                most_recent_write_time = None
                most_recent_write_width = None
                
                # Find the write pulse that comes just before this read
                for prev_pulse in sorted_pulses:
                    if (not prev_pulse['is_read'] and 
                        prev_pulse['id'] < pulse['id'] and
                        prev_pulse['id'] in absolute_times):
                        most_recent_write_id = prev_pulse['id']
                        most_recent_write_time = absolute_times[prev_pulse['id']]
                        most_recent_write_width = prev_pulse['pulse_width']
                
                if most_recent_write_time is not None and most_recent_write_width is not None:
                    # Calculate equidistant read placement within the same period
                    write_end_time = most_recent_write_time + most_recent_write_width
                    next_write_period = write_pulse_periods[most_recent_write_id] + 1
                    next_write_start_time = next_write_period * self.period
                    available_time = next_write_start_time - write_end_time
                    
                    # Position read pulse so edge-to-edge distances are equal
                    gap_before_read = (available_time - pulse['pulse_width']) // 2
                    absolute_times[pulse['id']] = write_end_time + gap_before_read
                else:
                    # Fallback: just use sequential timing
                    current_time = max(absolute_times.values()) if absolute_times else 0
                    absolute_times[pulse['id']] = current_time + 100_000  # Small gap
        
        return absolute_times
        
    def get_sequence_duration(self):
        """Get total duration of the pulse sequence"""
        if not self.pulses:
            return 0
            
        absolute_times = self.calculate_absolute_times()
        durations = []
        
        for pulse in self.pulses:
            pulse_end = absolute_times[pulse['id']] + pulse['pulse_width'] + pulse['delay_after']
            durations.append(pulse_end)
            
        return max(durations) if durations else 0
        
    def to_dict(self):
        """Convert the pulse sequence to a dictionary format using corrected absolute times."""
        all_pulses = []
        
        if self.period == 0:
            # relative-based sequences
            for pulse in self.pulses:
                pulse_dict = pulse.copy()
                pulse_dict['time_offset'] = pulse.get('time_offset', 0)
                all_pulses.append(pulse_dict)
        else:
            # period-based sequences
            absolute_times = self.calculate_absolute_times()
            for pulse in self.pulses:
                pulse_dict = pulse.copy()
                pulse_dict['time_offset'] = absolute_times.get(pulse['id'], 0)
                all_pulses.append(pulse_dict)
            all_pulses.sort(key=lambda p: p['time_offset'])

        
        # Separate write and read pulses
        write_pulses = [p for p in all_pulses if not p['is_read']]
        read_pulses = [p for p in all_pulses if p['is_read']]
        
        # Calculate total duration
        if all_pulses:
            max_end_time = max([p['time_offset'] + p['pulse_width'] for p in all_pulses])
            duration = max_end_time
        else:
            duration = 0
        
        return {
            'write': {'voltages': [p['voltage'] for p in write_pulses],
                    'pulse_widths': [p['pulse_width'] for p in write_pulses],
                    'time_offsets': [p['time_offset'] for p in write_pulses]},
            'read': {'voltages': [p['voltage'] for p in read_pulses],
                    'pulse_widths': [p['pulse_width'] for p in read_pulses], 
                    'time_offsets': [p['time_offset'] for p in read_pulses]},
            'all_pulses': all_pulses,
            'original_pulses': self.pulses,
            'write_pulses': write_pulses,
            'read_pulses': read_pulses,
            'read_indices': [i for i, p in enumerate(all_pulses) if p['is_read']],
            'sequence': all_pulses,
            'period': self.period,
            'read_voltage': self.default_read_voltage,
            'read_pulse_width': self.default_read_pulse_width,
            'read_pulse_offset': self.default_read_pulse_offset,
            'write_read_delay': getattr(self, 'write_read_delay', 0),
            'inter_pulse_delay': getattr(self, 'inter_pulse_delay', 0),
            'differential': self.differential,
            'duration': duration
        }
    
    def calculate_cluster_timings(self, low_pin, high_pin):
        """
        Pre-calculate cluster timings for all pulses.
        
        Args:
            low_pin (int): Low pin number
            high_pin (int): High pin number
        """
        for pulse in self.pulses:
            pulse_width = pulse['pulse_width']
            differential = pulse.get('differential', True)
            
            if pulse['is_read']:
                # For read pulses, only high pin is used
                pulse['cluster_timings'] = get_cluster_timing([high_pin], pulse_width)
            else:
                # For write pulses
                if differential:
                    # Both pins are used
                    pulse['cluster_timings'] = get_cluster_timing([low_pin, high_pin], pulse_width)
                else:
                    # Only high pin is used
                    pulse['cluster_timings'] = get_cluster_timing([high_pin], pulse_width)
                    
        return self
    

    
        # def _format_for_existing_api(self):
    #     """Format pulses into the existing API format"""
    #     # Calculate absolute times
    #     absolute_times = self.calculate_absolute_times()
        
    #     # Create pulses with absolute times
    #     pulses_with_abs_times = []
    #     for pulse in self.pulses:
    #         pulse_copy = pulse.copy()
    #         pulse_copy['time_offset'] = absolute_times[pulse['id']]
    #         pulses_with_abs_times.append(pulse_copy)
        
    #     # Extract write pulses only, sort by absolute time
    #     write_pulses = sorted(
    #         [p for p in pulses_with_abs_times if not p['is_read']], 
    #         key=lambda p: p['time_offset']
    #     )
        
    #     # Format as (voltage, pulse_width) tuples for compatibility with existing code
    #     sequence = [(p['voltage'], p['pulse_width']) for p in write_pulses]
        
    #     # Identify which indices should have reads after them
    #     read_indices = []
    #     for read_pulse in [p for p in pulses_with_abs_times if p['is_read']]:
    #         # Find the closest write pulse before this read
    #         read_time = read_pulse['time_offset']
    #         closest_idx = None
    #         min_diff = float('inf')
            
    #         for idx, write_pulse in enumerate(write_pulses):
    #             write_time = write_pulse['time_offset']
    #             # If this write comes before the read and is closer than any found so far
    #             if write_time < read_time and (read_time - write_time) < min_diff:
    #                 min_diff = read_time - write_time
    #                 closest_idx = idx
            
    #         if closest_idx is not None and closest_idx not in read_indices:
    #             read_indices.append(closest_idx)
        
    #     return sequence, read_indices, pulses_with_abs_times