import os
import numpy as np
import matplotlib.pyplot as plt
from ..measurement.base import MeasurementResult
from matplotlib.colors import LinearSegmentedColormap
import traceback

def plot_pulse_sequence(sequence, title='Pulse Sequence Waveform', show=True, fig=None, ax=None, time_range=None):
    """Enhanced visualization with proper support for read and write pulses"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert PulseSequence to dict if needed
    if hasattr(sequence, 'to_dict'):
        sequence = sequence.to_dict()
    
    # Extract all pulses
    all_pulses = sequence.get('all_pulses', [])
    if not all_pulses:
        print("No pulses found in sequence")
        return
    
    # Create figure if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Separate read and write pulses
    read_pulses = [p for p in all_pulses if p.get('is_read', False)]
    write_pulses = [p for p in all_pulses if not p.get('is_read', False)]
    
    # Find time range for plot
    max_time = max([p.get('time_offset', 0) for p in all_pulses]) + 2000000  # Add buffer
    time_ms = np.linspace(0, max_time/1000000, 1000)  # Convert ns to ms
    
    # Plot write pulses
    for pulse in write_pulses:
        t_start = pulse.get('time_offset', 0) / 1000000  # ns to ms
        width = pulse.get('pulse_width', 100000) / 1000000  # ns to ms
        voltage = pulse.get('voltage', 0)
        
        # Plot rectangle for pulse
        rect = plt.Rectangle((t_start, 0), width, voltage, 
                             color='blue', alpha=0.7, label='_Write')
        ax.add_patch(rect)
        
        # # Add voltage label
        # ax.text(t_start + width/2, voltage*1.05, f"{voltage:.2f}V", 
        #         ha='center', va='bottom' if voltage > 0 else 'top')
    
    # Plot read pulses
    for pulse in read_pulses:
        t_start = pulse.get('time_offset', 0) / 1000000  # ns to ms
        width = pulse.get('pulse_width', 100000) / 1000000  # ns to ms
        voltage = pulse.get('voltage', 0)
        
        # Plot rectangle for pulse with different color
        rect = plt.Rectangle((t_start, 0), width, voltage, 
                             color='red', alpha=0.4, label='_Read')
        ax.add_patch(rect)
        
        # # Add voltage label
        # ax.text(t_start + width/2, voltage*1.05, f"{voltage:.2f}V", 
        #         ha='center', va='bottom', fontsize=8, color='darkred')
    
    # Create manual legend
    write_patch = plt.Rectangle((0,0), 1, 1, color='blue', alpha=0.7, label='Write Pulse')
    read_patch = plt.Rectangle((0,0), 1, 1, color='red', alpha=0.4, label='Read Pulse')
    ax.legend(handles=[write_patch, read_patch])
    
    # Set plot parameters
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title(title)
    
    # scale y axis dynamically for better readability
    unique_voltages = set()
    for p in all_pulses:
        voltage = p.get('voltage', 0)
        if abs(voltage) > 0.001:  # Skip values very close to zero
            unique_voltages.add(round(voltage, 2))
    
    # Adjust y limits to show all pulses with appropriate padding
    v_max = max([p.get('voltage', 0) for p in all_pulses])
    v_min = min([p.get('voltage', 0) for p in all_pulses])
    padding = max(0.5, (v_max - v_min) * 0.2)
    y_min = v_min - padding if v_min < 0 else -padding
    y_max = v_max + padding
    ax.set_ylim(y_min, y_max)
    
    # Set specific y-ticks based on actual voltage levels used
    voltage_ticks = sorted(list(unique_voltages))
    # Add zero if not already there
    if 0 not in voltage_ticks:
        voltage_ticks.append(0)
    voltage_ticks = sorted(voltage_ticks)
    
    # Ensure y-ticks include all voltage levels
    ax.set_yticks(voltage_ticks)
    
    # Format y-tick labels to show voltage values
    ax.set_yticklabels([f"{v:.2f}V" for v in voltage_ticks])
    
    # Add more prominent grid for easier reading
    ax.grid(True, which='both', alpha=0.6, linestyle='-')
    ax.grid(True, which='major', alpha=0.8, linestyle='-')
    
    # Add horizontal lines at key voltage levels for easier reading
    for voltage in voltage_ticks:
        ax.axhline(y=voltage, color='gray', linestyle='--', alpha=0.6)
    
    if time_range is not None:
        ax.set_xlim(time_range[0], time_range[1])
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return fig, ax

def plot_measurement_results(result, plot_type: str = 'all',save_plot: bool = False, filename: str = None):
    """
    Plot measurement results.
    
    Parameters:
    -----------
    result : MeasurementResult
        Measurement results to plot
    plot_type : str
        Type of plot: 'all', 'voltage', 'current', 'resistance'
    save_plot : bool
        Whether to save the plot as png file
    filename : str, optional
        Filename to save the plot to (without extension)
    """
    if len(result.voltages) == 0:
        print("No data to plot")
        return

    # Check if we have new is_read attribute
    if hasattr(result, 'is_read'):
        # New plotting with write/read separation
        write_indices = [i for i, is_read in enumerate(result.is_read) if not is_read]
        read_indices = [i for i, is_read in enumerate(result.is_read) if is_read]
        
        # Create mapping: which write operation does each read correspond to?
        write_to_read_mapping = {}  # write_operation_number -> read_measurement_index
        
        # Try to use metadata if available, otherwise create simple mapping
        if (hasattr(result, 'metadata') and result.metadata and 
            'write_metadata' in result.metadata and 'read_metadata' in result.metadata):
            
            write_times = result.metadata['write_metadata']['time_offsets']
            read_times = result.metadata['read_metadata']['time_offsets']
            
            # Map each read to the write period it belongs to
            read_idx = 0
            for write_idx, write_time in enumerate(write_times):
                # Define the period for this write (until next write or +period)
                if write_idx < len(write_times) - 1:
                    next_write_time = write_times[write_idx + 1]
                else:
                    next_write_time = write_time + 1000000  # period from config
                
                # Check if there's a read in this write's period
                if read_idx < len(read_times):
                    read_time = read_times[read_idx]
                    if write_time <= read_time < next_write_time:
                        write_to_read_mapping[write_idx] = read_idx
                        read_idx += 1
        else:
            print("Using simple mapping for loaded data")
            # Simple mapping: assume alternating write/read pattern
            # This works for most loaded LTP/LTD data
            write_to_read_mapping = {}
            read_counter = 0
            for i, write_idx in enumerate(write_indices):
                if read_counter < len(read_indices):
                    write_to_read_mapping[i] = read_counter
                    read_counter += 1
        
        print(f"Debug: write_to_read_mapping = {write_to_read_mapping}")
        print(f"Debug: number of writes = {len(write_indices)}")
        print(f"Debug: number of reads = {len(read_indices)}")
        print(f"Debug: number of current measurements = {len(result.currents)}")
        
        # Plot with the mapping we have
        if plot_type == 'all':
            fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=False)

            def get_tick_spacing(n_points):
                # Choose a reasonable number of ticks based on data size
                if n_points <= 10:
                    return 1  # Show all ticks for small datasets
                elif n_points <= 20:
                    return 2  # Every 2nd tick
                elif n_points <= 50:
                    return 5  # Every 5th tick
                elif n_points <= 100:
                    return 10  # Every 10th tick
                else:
                    return n_points // 10

            # 1. Voltage plot - all writes
            if write_indices:
                write_voltages = [result.voltages[i] for i in write_indices]
                axs[0].plot(np.arange(len(write_voltages)), write_voltages, 'o-b', markersize=8)
                axs[0].set_ylabel('Voltage (V)')
                axs[0].set_title('Applied Write Voltages')
                spacing = get_tick_spacing(len(write_voltages))
                tick_positions = np.arange(0, len(write_voltages), spacing)
                tick_labels = [f'{i}' for i in range(0, len(write_voltages), spacing)]
                axs[0].set_xticks(tick_positions)
                axs[0].set_xticklabels(tick_labels)
            else:
                axs[0].text(0.5, 0.5, 'No write operations', ha='center', va='center', transform=axs[0].transAxes)
            axs[0].grid(True)

            # 2. Current plot - all reads
            if read_indices and len(result.currents) >= len(read_indices):
                read_currents = [-result.currents[i] * 1e6 for i in range(len(read_indices))]
                axs[1].plot(np.arange(len(read_currents)), read_currents, 'or', markersize=8)
                axs[1].set_ylabel('Current (μA)')
                axs[1].set_title('Measured Read Currents')
                spacing = get_tick_spacing(len(read_currents))
                tick_positions = np.arange(0, len(read_currents), spacing)
                tick_labels = [f'{i}' for i in range(0, len(read_currents), spacing)]
                axs[1].set_xticks(tick_positions)
                axs[1].set_xticklabels(tick_labels)
            else:
                axs[1].text(0.5, 0.5, 'No read operations', ha='center', va='center', transform=axs[1].transAxes)
            axs[1].grid(True)

            # 3. Resistance plot - all reads
            if read_indices and len(result.resistances) >= len(read_indices):
                read_resistances = [abs(result.resistances[i]) for i in range(len(read_indices))]
                axs[2].semilogy(np.arange(len(read_resistances)), read_resistances, 'og', markersize=8)
                axs[2].set_ylabel('Resistance (Ω)')
                axs[2].set_title('Calculated Read Resistances')
                spacing = get_tick_spacing(len(read_resistances))
                tick_positions = np.arange(0, len(read_resistances), spacing)
                tick_labels = [f'{i}' for i in range(0, len(read_resistances), spacing)]
                axs[2].set_xticks(tick_positions)
                axs[2].set_xticklabels(tick_labels)
            else:
                axs[2].text(0.5, 0.5, 'No resistance measurements', ha='center', va='center', transform=axs[2].transAxes)
            axs[2].grid(True)
            
            # 4. Conductance plot - all reads  
            if read_indices and len(result.resistances) >= len(read_indices):
                read_conductances = []
                for i in range(len(read_indices)):
                    r = result.resistances[i]
                    if abs(r) > 1e-12:  # Avoid division by zero or very small values
                        read_conductances.append(1.0 / abs(r) * 1e6)  # Convert to μS
                    else:
                        read_conductances.append(0)  # Set to zero for infinite resistance
                
                axs[3].semilogy(np.arange(len(read_conductances)), read_conductances, 'om', markersize=8)
                axs[3].set_xlabel('number of Pulses') 
                axs[3].set_ylabel('Conductance (μS)')
                axs[3].set_title('Calculated Read Conductances')
                spacing = get_tick_spacing(len(read_conductances))
                tick_positions = np.arange(0, len(read_conductances), spacing)
                tick_labels = [f'{i}' for i in range(0, len(read_conductances), spacing)]
                axs[3].set_xticks(tick_positions)
                axs[3].set_xticklabels(tick_labels)
            else:
                axs[3].text(0.5, 0.5, 'No conductance measurements', ha='center', va='center', transform=axs[3].transAxes)
            axs[3].grid(True)

            plt.tight_layout()
            if save_plot:
                from datetime import datetime
                figures_dir = 'figures'
                os.makedirs(figures_dir, exist_ok=True)
                if filename is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"measurement_results_{timestamp}"
                if not filename.endswith('.png'):
                    full_path = os.path.join(figures_dir, f"{filename}.png")
                else:
                    full_path = os.path.join(figures_dir, filename)
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {full_path}")
            plt.show()
            return
        
        if plot_type == 'all':
            fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
            
            # 1. Voltage plot - show all write voltages
            if write_indices:
                write_voltages = [result.voltages[i] for i in write_indices]
                x_writes = np.arange(len(write_voltages))
                axs[0].plot(x_writes, write_voltages, 'o-b', markersize=8)
                axs[0].set_ylabel('Voltage (V)')
                axs[0].set_title('Applied Write Voltage vs. Write Operation')
                axs[0].set_xticks(x_writes)
                axs[0].set_xticklabels([f'W{i}' for i in range(len(write_voltages))])
            else:
                axs[0].text(0.5, 0.5, 'No write operations', ha='center', va='center', transform=axs[0].transAxes)
            axs[0].grid(True)
            
            # 2. Current plot - show all write positions, but only plot points where reads exist
            if write_indices:
                x_writes = np.arange(len(write_voltages))
                
                # Collect data for line plotting
                line_x = []
                line_y = []
                
                # Plot points only where reads exist
                for write_idx, read_idx in write_to_read_mapping.items():
                    if read_idx < len(result.currents):
                        current_value = -result.currents[read_idx] * 1e6  # Convert to μA
                        axs[1].plot(write_idx, current_value, 'or', markersize=8)
                        line_x.append(write_idx)
                        line_y.append(current_value)
                        print(f"Debug: Plotting current at write {write_idx} (read {read_idx}): {current_value} μA")
                
                # Connect consecutive points with lines
                if len(line_x) > 1:
                    axs[1].plot(line_x, line_y, 'r-', alpha=0.6, linewidth=1.5)
                
                axs[1].set_ylabel('Current (μA)')
                axs[1].set_title('Measured Current vs. Write Operation (only reads shown)')
                axs[1].set_xticks(x_writes)
                axs[1].set_xticklabels([f'W{i}' for i in range(len(write_voltages))])
            else:
                axs[1].text(0.5, 0.5, 'No write operations', ha='center', va='center', transform=axs[1].transAxes)
            axs[1].grid(True)
            
            # 3. Resistance plot
            if write_indices:
                x_writes = np.arange(len(write_voltages))
                
                # Collect data for line plotting
                line_x = []
                line_y = []
                
                # Plot points only where reads exist
                for write_idx, read_idx in write_to_read_mapping.items():
                    if read_idx < len(result.resistances):
                        resistance_value = abs(result.resistances[read_idx])
                        axs[2].semilogy(write_idx, resistance_value, 'og', markersize=8)
                        line_x.append(write_idx)
                        line_y.append(resistance_value)
                        print(f"Debug: Plotting resistance at write {write_idx} (read {read_idx}): {resistance_value} Ω")
                
                # Connect consecutive points with lines
                if len(line_x) > 1:
                    axs[2].semilogy(line_x, line_y, 'g-', alpha=0.6, linewidth=1.5)
                
                axs[2].set_ylabel('|Resistance| (Ω)')
                axs[2].set_title('Calculated Resistance vs. Write Operation (only reads shown)')
                axs[2].set_xticks(x_writes)
                axs[2].set_xticklabels([f'W{i}' for i in range(len(write_voltages))])
            else:
                axs[2].text(0.5, 0.5, 'No write operations', ha='center', va='center', transform=axs[2].transAxes)
            axs[2].grid(True)

            # 4. Conductance plot
            if write_indices:
                x_writes = np.arange(len(write_voltages))
                
                # Collect data for line plotting
                line_x = []
                line_y = []
                
                # Plot points only where reads exist
                for write_idx, read_idx in write_to_read_mapping.items():
                    if read_idx < len(result.resistances):
                        # Calculate conductance (1/R) in μS
                        if abs(result.resistances[read_idx]) > 1e-12:  # Avoid division by zero
                            conductance_value = 1.0 / abs(result.resistances[read_idx]) * 1e6  # Convert to μS
                        else:
                            conductance_value = 0  # Set to zero for infinite resistance
                        
                        axs[3].semilogy(write_idx, conductance_value, 'om', markersize=8)
                        line_x.append(write_idx)
                        line_y.append(conductance_value)
                        print(f"Debug: Plotting conductance at write {write_idx} (read {read_idx}): {conductance_value} μS")
                
                # Connect consecutive points with lines
                if len(line_x) > 1:
                    axs[3].semilogy(line_x, line_y, 'm-', alpha=0.6, linewidth=1.5)
                
                axs[3].set_xlabel('Write Operation')  # Only bottom plot needs x-label
                axs[3].set_ylabel('Conductance (μS)')
                axs[3].set_title('Calculated Conductance vs. Write Operation (only reads shown)')
                axs[3].set_xticks(x_writes)
                axs[3].set_xticklabels([f'W{i}' for i in range(len(write_voltages))])
            else:
                axs[3].text(0.5, 0.5, 'No write operations', ha='center', va='center', transform=axs[3].transAxes)
            axs[3].grid(True)
            
            plt.tight_layout()
            
            # Save plot if requested
            if save_plot:
                from datetime import datetime
                figures_dir = 'figures'
                os.makedirs(figures_dir, exist_ok=True)
                
                if filename is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Include device info if available
                    device_info = ""
                    if hasattr(result, 'metadata') and result.metadata:
                        row_pin = result.metadata.get('high_pin', result.metadata.get('row_pin', ''))
                        col_pin = result.metadata.get('low_pin', result.metadata.get('column_pin', ''))
                        if row_pin and col_pin:
                            device_info = f"_pins{col_pin}-{row_pin}"
                    filename = f"custom_pulse_measurement_results{device_info}_{timestamp}"
                
                # Add extension if not present
                if not filename.endswith('.png'):
                    full_path = os.path.join(figures_dir, f"{filename}.png")
                else:
                    full_path = os.path.join(figures_dir, filename)
                
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {full_path}")
           
            plt.show()
        
        elif plot_type == 'voltage':
            plt.figure(figsize=(10, 6))
            write_indices = [i for i, is_read in enumerate(result.is_read) if not is_read]
            if write_indices:
                write_voltages = [result.voltages[i] for i in write_indices]
                x_writes = np.arange(len(write_voltages))
                plt.plot(x_writes, write_voltages, 'o-b', markersize=8)
                plt.xlabel('Write Operation')
                plt.title('Applied Write Voltage vs. Write Operation')
                plt.xticks(x_writes, [f'W{i}' for i in range(len(write_voltages))])
            else:
                plt.text(0.5, 0.5, 'No write operations', ha='center', va='center', transform=plt.gca().transAxes)
            plt.ylabel('Voltage (V)')
            plt.grid(True)
            plt.show()
        
        elif plot_type == 'current':
            if len(result.currents) > 0 and write_indices:
                plt.figure(figsize=(10, 6))
                write_voltages = [result.voltages[i] for i in write_indices]
                x_writes = np.arange(len(write_voltages))
                
                # Plot only points with measurements
                for write_idx, read_idx in write_to_read_mapping.items():
                    if read_idx < len(result.currents):
                        current_value = result.currents[read_idx] * 1e6
                        plt.plot(write_idx, current_value, 'or', markersize=8)
                
                plt.xlabel('Write Operation')
                plt.ylabel('Current (μA)')
                plt.title('Measured Current vs. Write Operation')
                plt.xticks(x_writes, [f'W{i}' for i in range(len(write_voltages))])
                plt.grid(True)
                plt.show()
            else:
                print("No current measurements to plot")
        
        elif plot_type == 'resistance':
            if len(result.resistances) > 0 and write_indices:
                plt.figure(figsize=(10, 6))
                write_voltages = [result.voltages[i] for i in write_indices]
                x_writes = np.arange(len(write_voltages))
                
                # Plot only points with measurements
                for write_idx, read_idx in write_to_read_mapping.items():
                    if read_idx < len(result.resistances):
                        resistance_value = abs(result.resistances[read_idx])
                        plt.semilogy(write_idx, resistance_value, 'og', markersize=8)
                
                plt.xlabel('Write Operation')
                plt.ylabel('|Resistance| (Ω)')
                plt.title('Resistance vs. Write Operation')
                plt.xticks(x_writes, [f'W{i}' for i in range(len(write_voltages))])
                plt.grid(True)
                plt.show()
            else:
                print("No resistance measurements to plot")

        elif plot_type == 'conductance':
            if len(result.resistances) > 0 and write_indices:
                plt.figure(figsize=(10, 6))
                write_voltages = [result.voltages[i] for i in write_indices]
                x_writes = np.arange(len(write_voltages))
                
                # Plot only points with measurements
                for write_idx, read_idx in write_to_read_mapping.items():
                    if read_idx < len(result.resistances):
                        if abs(result.resistances[read_idx]) > 1e-12:  # Avoid division by zero
                            conductance_value = 1.0 / abs(result.resistances[read_idx]) * 1e6  # Convert to μS
                        else:
                            conductance_value = 0  # Set to zero for infinite resistance
                        
                        plt.semilogy(write_idx, conductance_value, 'om', markersize=8)
                
                plt.xlabel('Write Operation')
                plt.ylabel('Conductance (μS)')
                plt.title('Conductance vs. Write Operation')
                plt.xticks(x_writes, [f'W{i}' for i in range(len(write_voltages))])
                plt.grid(True)
                plt.show()
            else:
                print("No conductance measurements to plot")
            
    else:
        # Backwards compatibility mode - old behavior for scripts without is_read
        if plot_type == 'all':
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            
            x = np.arange(1, len(result.voltages) + 1)
            
            # Voltage plot - all voltages
            axs[0].plot(x, result.voltages, 'o-b')
            axs[0].set_ylabel('Voltage (V)')
            axs[0].set_title('Applied Voltage vs. Operation Number')
            axs[0].grid(True)
            
            # Current plot - only available currents
            if len(result.currents) > 0:
                x_curr = np.arange(1, len(result.currents) + 1)
                axs[1].plot(x_curr, -np.array(result.currents) * 1e6, 'o-r', markersize=8)
                axs[1].set_ylabel('Current (μA)')
                axs[1].set_title('Measured Current')
            else:
                axs[1].text(0.5, 0.5, 'No current measurements', ha='center', va='center', transform=axs[1].transAxes)
            axs[1].grid(True)
            
            # Resistance plot - only available resistances
            if len(result.resistances) > 0:
                x_res = np.arange(1, len(result.resistances) + 1)
                axs[2].semilogy(x_res, np.abs(result.resistances), 'o-g', markersize=8)
                axs[2].set_xlabel('Operation Number')
                axs[2].set_ylabel('|Resistance| (Ω)')
                axs[2].set_title('Calculated Resistance')
            else:
                axs[2].text(0.5, 0.5, 'No resistance measurements', ha='center', va='center', transform=axs[2].transAxes)
            axs[2].grid(True)
            
            plt.tight_layout()
            plt.show()
        
        elif plot_type == 'voltage':
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(1, len(result.voltages) + 1), result.voltages, 'o-b')
            plt.xlabel('Operation Number')
            plt.ylabel('Voltage (V)')
            plt.title('Applied Voltage vs. Operation Number')
            plt.grid(True)
            plt.show()
        
        elif plot_type == 'current':
            if len(result.currents) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(np.arange(1, len(result.currents) + 1), 
                        -np.array(result.currents) * 1e6, 'o-r', markersize=8)
                plt.xlabel('Measurement Number')
                plt.ylabel('Current (μA)')
                plt.title('Measured Current')
                plt.grid(True)
                plt.show()
            else:
                print("No current measurements to plot")
        
        elif plot_type == 'resistance':
            if len(result.resistances) > 0:
                plt.figure(figsize=(10, 6))
                plt.semilogy(np.arange(1, len(result.resistances) + 1), 
                            np.abs(result.resistances), 'o-g', markersize=8)
                plt.xlabel('Measurement Number')
                plt.ylabel('|Resistance| (Ω)')
                plt.title('Resistance')
                plt.grid(True)
                plt.show()
            else:
                print("No resistance measurements to plot")

def visualize_iv_sweep(max_voltage, steps, sweep_type, title=None, fig=None, ax=None, show=True):
    """
    Visualize an IV sweep voltage pattern before execution.
    
    Parameters:
    -----------
    max_voltage : float
        Maximum voltage magnitude in volts
    steps : int
        Number of voltage steps in the sweep
    sweep_type : str
        Type of sweep: "hysteresis", "full", "positive", or "negative"
    title : str, optional
        Custom title for the plot, default is generated based on sweep_type
    fig : matplotlib.figure.Figure, optional
        Figure to plot on; creates new figure if None
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; creates new axes if None
    show : bool, optional
        Whether to display the plot immediately (default: True)
    
    Returns:
    --------
    fig, ax : tuple
        Figure and axes objects for further customization
    voltages : numpy.ndarray
        Array of voltage points in the sweep
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Generate voltage points based on sweep type
    if sweep_type == "hysteresis":
        forward = np.linspace(0, max_voltage, steps + 1)
        backward = np.linspace(max_voltage, 0, steps + 1)[1:]
        voltages = np.concatenate([forward, backward])
    elif sweep_type == "full":
        step_quarter = max(steps, 1)
        pos_forward = np.linspace(0, max_voltage, step_quarter + 1)
        pos_backward = np.linspace(max_voltage, 0, step_quarter + 1)[1:]
        neg_forward = np.linspace(0, -max_voltage, step_quarter + 1)[1:]
        neg_backward = np.linspace(-max_voltage, 0, step_quarter + 1)[1:]
        voltages = np.concatenate([pos_forward, pos_backward, neg_forward, neg_backward])
    elif sweep_type == "positive":
        voltages = np.linspace(0, max_voltage, steps)
    elif sweep_type == "negative":
        voltages = np.linspace(0, -max_voltage, steps)
    else:
        voltages = np.array([])
        print(f"Unknown sweep type: {sweep_type}")

    if len(voltages) > 0:
        ax.plot(range(len(voltages)), voltages, 'b-', lw=2)
        ax.scatter(range(len(voltages)), voltages, c=range(len(voltages)), cmap='viridis', s=30, zorder=5)

        # Set plot limits with a bit of padding
        v_max = np.max(voltages)
        v_min = np.min(voltages)
        abs_max = max(abs(v_max), abs(v_min), 1)
        y_padding = max(0.05, abs_max * 0.15)
        ax.set_ylim(v_min - y_padding, v_max + y_padding)

        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Sweep Step')
        ax.set_ylabel('Voltage (V)')

        if title is None:
            title = f'{sweep_type.title()} IV Sweep visualization ({len(voltages)} points, {max_voltage:.2f}V max)'
        ax.set_title(title)

        plt.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0, len(voltages)-1), cmap='viridis'),
            ax=ax, label='Measurement sequence'
        )

        plt.tight_layout()
        if show:
            plt.show()

    return fig, ax, voltages

def plot_iv_curves(result, save_plot=False, filename=None, save_dir='figures'):
    """
    Plot comprehensive I-V curve analysis with multiple perspectives.
    
    Parameters:
    -----------
    result : MeasurementResult
        Container with measurement results
    save_plot : bool
        Whether to save the plot to file
    filename : str, optional
        Filename to save the plot to
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    
    if result is None or len(result.voltages) == 0:
        print("No valid data to plot")
        return
    
    try:
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. I-V Characteristic curve
        ax1 = fig.add_subplot(2, 2, 1)
        
        # Use color gradient to show measurement sequence
        voltage = np.array(result.voltages)
        current = np.array(result.currents) * 1e6  # Convert to μA
        n_points = len(voltage)
        
        # Create a color gradient to show the sequence
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))
        
        # Plot points with color gradient and connect with lines
        for i in range(n_points-1):
            ax1.plot(voltage[i:i+2], current[i:i+2], '-', color=colors[i], linewidth=1.5)
        
        # Plot scatter points on top
        scatter = ax1.scatter(voltage, current, c=np.arange(n_points), cmap='viridis', 
                              s=10, zorder=10, edgecolor='black', linewidth=0.5)
        
        # Add colorbar to show sequence direction
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Measurement Sequence')
        
        # Add arrows to show sweep direction at key points
        arrow_indices = []
        # Choose arrow positions based on sweep type
        if 'sweep_type' in result.metadata:
            sweep_type = result.metadata['sweep_type']
            if sweep_type == 'full':
                # For the full cycle, place arrows at key transitions
                steps_per_segment = n_points // 4
                arrow_indices = [steps_per_segment//2, steps_per_segment+steps_per_segment//2, 
                                2*steps_per_segment+steps_per_segment//2, 3*steps_per_segment+steps_per_segment//2]
            else:
                # Default approach for other sweep types
                arrow_indices = [int(n_points/8), int(3*n_points/8), int(5*n_points/8), int(7*n_points/8)]
        
        for i in arrow_indices:
            if i+3 < n_points:
                ax1.annotate('',
                            xy=(voltage[i+3], current[i+3]),
                            xytext=(voltage[i], current[i]),
                            arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.7))
        
        ax1.set_xlabel('Voltage (V)')
        ax1.set_ylabel('Current (μA)')
        ax1.set_title('I-V Characteristic')
        ax1.grid(True)
        
        # 2. R-V Characteristic curve (log scale)
        ax2 = fig.add_subplot(2, 2, 2)
        resistance_data = np.array(result.resistances)
        # Convert to absolute values before filtering
        abs_resistance = np.abs(resistance_data)
        # Filter out non-finite or very large values
        valid_indices = np.isfinite(abs_resistance) & (abs_resistance > 0) & (abs_resistance < 1e9) & (np.abs(voltage)>0.1)

        if np.any(valid_indices):
            scatter_r = ax2.scatter(
                np.array(voltage)[valid_indices],
                abs_resistance[valid_indices],
                c=np.arange(n_points)[valid_indices],
                cmap='viridis',
                s=10,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Connect points with lines
            ax2.semilogy(
                np.array(voltage)[valid_indices],
                abs_resistance[valid_indices],
                '-',
                linewidth=1,
                alpha=0.7,
                color='gray'
            )
            # Add colorbar
            cbar_r = plt.colorbar(scatter_r, ax=ax2)
            cbar_r.set_label('Measurement Sequence')
            
        ax2.set_xlabel('Voltage (V)')
        ax2.set_ylabel('Resistance (Ω)')
        ax2.set_title('R-V Characteristic')
        ax2.grid(True)
        
        # Create relative time axis in seconds
        time_values = np.array(result.timestamps) - result.timestamps[0]
        
        # 3. Voltage vs Time plot
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(time_values, voltage, '-', linewidth=1.5, color='green')
        ax3.scatter(time_values, voltage, c=np.arange(n_points), cmap='viridis', 
                   s=10, edgecolor='black', linewidth=0.5)
        
        # Mark important transitions in the voltage pattern
        if 'sweep_type' in result.metadata:
            sweep_type = result.metadata['sweep_type']
            if sweep_type == 'full' or sweep_type == 'set-reset':
                # Mark the 0→max→0→min→0 transitions
                segments = n_points // 4
                transition_points = [0, segments, 2*segments, 3*segments, n_points-1]
                for idx in transition_points:
                    ax3.plot(time_values[idx], voltage[idx], 'ro', markersize=8, fillstyle='none')
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Voltage (V)')
        ax3.set_title('Voltage vs Time')
        ax3.grid(True)
        
        # 4. Current vs Time plot
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(time_values, current, '-', linewidth=1.5, color='purple')
        scatter_i = ax4.scatter(time_values, current, c=np.arange(n_points), cmap='viridis', 
                              s=10, edgecolor='black', linewidth=0.5)
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Current (μA)')
        ax4.set_title('Current vs Time')
        ax4.grid(True)
        
        # Add metadata as text annotation
        row_pin = result.metadata.get('row_pin', 'N/A')
        col_pin = result.metadata.get('column_pin', 'N/A')
        sweep_type = result.metadata.get('sweep_type', 'standard')
        meas_time = result.metadata.get('measurement_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        plt.figtext(0.5, 0.01, 
                  f"Device: Channels {col_pin}-{row_pin} | " +
                  f"Sweep: {sweep_type} | " +
                  f"{meas_time}", 
                  ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        if save_plot:
            # Create the figures directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'iv_curve_analysis_{timestamp}'
            
            # Add extension if not present
            if not filename.endswith('.png') and not filename.endswith('.jpg') and not filename.endswith('.pdf'):
                full_path = os.path.join(save_dir, f"{filename}.png")
            else:
                full_path = os.path.join(save_dir, filename)
                
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {full_path}")
        
        plt.show()
        return fig
        
    except Exception as e:
        print(f"Error plotting characteristics: {e}")
        print(traceback.format_exc())

def visualize_vmm_config(vmm_config, show=True):
    """
    Visualize a VMM configuration.
    
    Parameters:
    -----------
    vmm_config : Dict
        VMM configuration with input_vector and voltage_vector
    show : bool
        If True, display the plot. Otherwise, return the figure.
    
    Returns:
    --------
    tuple
        Figure and axes objects if show=False, None otherwise
    """
    input_vector = vmm_config['input_vector']
    voltage_vector = vmm_config['voltage_vector']
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot input vector
    ax1.bar(range(len(input_vector)), input_vector)
    ax1.set_title('Input Vector')
    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Input Value')
    ax1.grid(True, alpha=0.3)
    
    # Plot voltage vector
    ax2.bar(range(len(voltage_vector)), voltage_vector)
    ax2.set_title('Applied Voltages')
    ax2.set_xlabel('Column Index')
    ax2.set_ylabel('Voltage (V)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"VMM Operation Summary:")
    print(f"  Input vector range: [{np.min(input_vector):.3f}, {np.max(input_vector):.3f}]")
    print(f"  Voltage vector range: [{np.min(voltage_vector):.3f}V, {np.max(voltage_vector):.3f}V]")
    print(f"  Active columns: {len(vmm_config['active_cols'])}")
    print(f"  Grounded columns: {len(vmm_config['grounded_cols'])}")
    
    if show:
        plt.show()
        return None
    return fig, (ax1, ax2)

def visualize_crossbar(conductance_matrix, row_pins=None, col_pins=None, 
                     value_type='conductance', cmap='viridis', log_scale=True, 
                     read_voltage=0.1, figsize=(10, 8), show=True):
    """
    Visualize the crossbar array using a heatmap with value annotations.
    
    Parameters:
    -----------
    conductance_matrix : numpy.ndarray
        2D conductance matrix of the crossbar
    row_pins : list
        List of row pin numbers for annotation
    col_pins : list
        List of column pin numbers for annotation
    value_type : str
        Type of values to display: 'conductance', 'resistance', or 'current'
    cmap : str
        Matplotlib colormap to use
    log_scale : bool
        Whether to use logarithmic scale for values
    read_voltage : float
        Read voltage used for current calculation
    figsize : tuple
        Figure size (width, height) in inches
    show : bool
        Whether to display the plot immediately
        
    Returns:
    --------
    tuple
        (fig, ax) - Matplotlib figure and axis objects
    """
    # Create figure with proper spacing for title
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data based on value type
    if value_type == 'conductance':
        data = conductance_matrix
        title = 'Conductance (S)'
        format_str = '{:.2e} S'
    elif value_type == 'resistance':
        # Convert conductance to resistance
        data = np.zeros_like(conductance_matrix)
        mask = conductance_matrix > 1e-12
        data[mask] = 1.0 / conductance_matrix[mask]
        data[~mask] = float('inf')  # Use infinity for zeros
        title = 'Resistance (Ω)'
        format_str = '{:.2e} Ω'
    elif value_type == 'current':
        data = conductance_matrix * read_voltage
        title = f'Current at {read_voltage}V (A)'
        format_str = '{:.2e} A'
    else:
        raise ValueError(f"Unknown value_type: {value_type}")
    
    rows, cols = data.shape
    
    # Create a copy of data for visualization (handle infinities)
    plot_data = np.copy(data)
    # Replace infinity with very large value for visualization
    max_finite = np.max(plot_data[np.isfinite(plot_data)]) if np.any(np.isfinite(plot_data)) else 1e10
    plot_data[~np.isfinite(plot_data)] = max_finite * 10
    
    # Use log scale if requested and data is suitable
    if log_scale and np.any(plot_data > 0):
        # Replace zeros with minimum positive value
        min_positive = np.min(plot_data[plot_data > 0])
        data_log = np.copy(plot_data)
        data_log[data_log <= 0] = min_positive / 10
        plot_data = np.log10(data_log)
        norm = plt.Normalize(np.min(plot_data), np.max(plot_data))
        
        # Create the colormap
        im = ax.imshow(plot_data, cmap=cmap, norm=norm)
        title = f'Log10({title})'
        
        # Create colorbar with actual (non-log) values
        cbar = fig.colorbar(im, ax=ax)
        # Generate tick positions in log space
        tick_positions = np.linspace(np.min(plot_data), np.max(plot_data), 5)
        # Convert to actual values
        tick_labels = [f'{10**pos:.2e}' for pos in tick_positions]
        # Set ticks and labels
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
    else:
        # Linear scale
        im = ax.imshow(plot_data, cmap=cmap)
        fig.colorbar(im, ax=ax)
    
    # Add labels and title with padding
    ax.set_title(title, pad=20)  # Add padding to the title
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Add pin number annotations and values if not too large
    if rows <= 10 and cols <= 10:
        for i in range(rows):
            for j in range(cols):
                # Format the value based on type
                if np.isinf(data[i, j]):
                    value_text = "∞"
                else:
                    value_text = format_str.format(data[i, j])
                
                # Add pin numbers if available
                pin_text = ""
                if row_pins is not None and col_pins is not None:
                    pin_text = f'{row_pins[i]},{col_pins[j]}'
                    
                # Put pin numbers and values on separate lines
                text = f"{pin_text}\n{value_text}" if pin_text else value_text
                
                ax.text(j, i, text, 
                    ha='center', va='center', color='white',
                    fontsize=7)  # Smaller font size for more text
    
    # Adjust layout to make room for the title and annotations
    plt.tight_layout(rect=[0, 0, 1, 0.92]) 
    
    if show:
        plt.show()
        
    return fig, ax