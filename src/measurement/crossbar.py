import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import List, Dict, Tuple, Optional, Any, Union

from ..instruments import connect_arc_two
from .pulse import PulseMeasurement, PulseOperation, PulseConfiguration
from .base import MeasurementResult
from ..instruments.pulse_sequence import PulseSequence

from ..analysis import plotting
from ..utils import data_io

class Crossbar:
    def __init__(self, instrument, rows, cols, row_pins=None, col_pins=None, 
                 read_voltage=0.1, max_current=1e-3, 
                 sneak_path_mitigation="v_2"):
        """
        Initialize a crossbar array controller.
        
        Parameters:
        -----------
        instrument : Instrument
            Initialized ArC2 instrument
        rows : int
            Number of rows in the crossbar
        cols : int
            Number of columns in the crossbar
        row_pins : list
            ArC2 pins connected to rows (bitlines). If None, defaults to sequential pins.
        col_pins : list
            ArC2 pins connected to columns (wordlines). If None, defaults to sequential pins.
        read_voltage : float
            Default voltage used for reading devices
        max_current : float
            Compliance current for device protection
        sneak_path_mitigation : str
            Method to reduce sneak path effects: "ground_unused", "v_3", or "none"
        """
        # Core properties
        self.instrument = instrument
        self.rows = rows
        self.cols = cols
        
        # Pin assignment
        self.row_pins = row_pins if row_pins is not None else list(range(rows))
        self.col_pins = col_pins if col_pins is not None else list(range(16, 16 + cols))
        
        # Operation parameters
        self.read_voltage = read_voltage
        self.max_current = max_current
        self.sneak_path_mitigation = sneak_path_mitigation
        
        # Data storage
        self.conductance_matrix = np.zeros((rows, cols))
        self.last_vmm_result = None
        
        # Create measurement handler
        self.pulse_measurement = PulseMeasurement(instrument)
        
        # Verify pin configuration
        self._validate_pins()
        
    def _validate_pins(self):
        """Validate that pins are correctly configured for the crossbar."""
        # Check for pin count mismatch
        if len(self.row_pins) != self.rows:
            raise ValueError(f"Row pin count ({len(self.row_pins)}) doesn't match rows ({self.rows})")
        if len(self.col_pins) != self.cols:
            raise ValueError(f"Column pin count ({len(self.col_pins)}) doesn't match columns ({self.cols})")
        
        # Check for pin overlap
        all_pins = self.row_pins + self.col_pins
        if len(all_pins) != len(set(all_pins)):
            raise ValueError("Row and column pins must be unique")
            
        # Check that pins are in valid ArC2 range
        if max(all_pins) >= 64 or min(all_pins) < 0:
            raise ValueError("Pins must be in range 0-63")
        
    def from_config_file(instrument, config_file=None, rows=None, cols=None, 
                        row_pins=None, col_pins=None, read_voltage=0.1, max_current=1e-3):
        """
        Create a crossbar instsance from a TOML configuration file with fallback to manual configuration.
        
        Parameters:
        -----------
        instrument : pyarc2.ARCTWO
            ArC TWO instrument
        config_file : str, optional
            Path to the TOML configuration file (if None, uses manual config)
        rows : int, optional
            Number of rows (used if TOML not provided or fails)
        cols : int, optional
            Number of columns (used if TOML not provided or fails)
        row_pins : list, optional
            List of row pins (used if TOML not provided or fails)
        col_pins : list, optional
            List of column pins (used if TOML not provided or fails)
        read_voltage : float
            Voltage used for reading (V)
        max_current : float
            Maximum current (A)
            
        Returns:
        --------
        Crossbar
            Initialized crossbar instance
        """
        config = {}
        
        # Try loading from TOML if file is provided
        if config_file:
            try:
                from ..utils import data_io
                config = data_io.load_crossbar_config(config_file)
                print(f"Loaded crossbar configuration: {config['name']}")
            except Exception as e:
                print(f"Warning: Failed to load TOML configuration: {e}")
                print("Falling back to manual configuration")
        
        # Use provided parameters as fallbacks
        crossbar_rows = config.get('rows', rows)
        crossbar_cols = config.get('cols', cols)
        crossbar_row_pins = config.get('row_pins', row_pins)
        crossbar_col_pins = config.get('col_pins', col_pins)
        
        # Verify we have all required parameters
        if not all([crossbar_rows, crossbar_cols, crossbar_row_pins, crossbar_col_pins]):
            raise ValueError("Insufficient configuration parameters. Either provide a valid TOML file or complete manual configuration.")
        
        # Create the crossbar instance
        crossbar = Crossbar(
            instrument=instrument,
            rows=crossbar_rows,
            cols=crossbar_cols,
            row_pins=crossbar_row_pins,
            col_pins=crossbar_col_pins,
            read_voltage=read_voltage,
            max_current=max_current
        )
        
        # Add name as an attribute if it was in the config
        if 'name' in config:
            crossbar.name = config['name']
    
        return crossbar

    def create_vmm_config(self, input_vector, read_duration=1000, ground_unused=True, 
                        differential=True, debug=False):
        """
        Create a configuration for Vector-Matrix Multiplication operation.
        
        Parameters:
        -----------
        input_vector : numpy.ndarray
            1D input vector to multiply with the conductance matrix
        read_duration : int
            Duration of read pulse in nanoseconds
        ground_unused : bool
            Whether to ground unused columns to mitigate sneak paths
        differential : bool
            If True, use differential pulsing
        debug : bool
            Enable debug output
            
        Returns:
        --------
        Dict
            Contains VMM configuration with keys:
            - 'input_vector': Normalized input vector
            - 'voltage_vector': Vector of voltages to apply
            - 'active_cols': List of active column pins
            - 'grounded_cols': List of grounded column pins
            - 'read_configs': Dict mapping row indices to pulse configurations
            - 'max_input': Maximum value in input vector (for normalization)
        """
        if len(input_vector) != self.cols:
            raise ValueError(f"Input vector length ({len(input_vector)}) doesn't match columns ({self.cols})")
        
        # Scale input vector to voltage values
        voltage_vector = np.array(input_vector)
        max_input = np.max(np.abs(input_vector))

        max_allowed = 2  # max voltage which is safe for devices
        if np.any(np.abs(voltage_vector) > max_allowed):
            raise ValueError(f"Input vector contains voltages exceeding safe limit of {max_allowed} V.")
        
        # Determine active and grounded columns
        active_cols = []
        active_voltages = []
        grounded_cols = []
        
        for j, voltage in enumerate(voltage_vector):
            col_pin = self.col_pins[j]
            if abs(voltage) > 1e-6:  # Only set non-zero voltages
                active_cols.append(col_pin)
                active_voltages.append(voltage)
            elif ground_unused and self.sneak_path_mitigation == "ground_unused":
                grounded_cols.append(col_pin)
        
        # Create read configurations for each row
        read_configs = {}
        
        for i, row_pin in enumerate(self.row_pins):
            # Create a sequence for reading current from this row
            # We'll apply the voltages to columns and read from this row
            read_sequence = {'sequence': [(self.read_voltage, read_duration)], 'period': read_duration * 2}
            
            # Create configuration for reading this row
            config = self.pulse_measurement.create_pulse_configuration(
                low_pin=row_pin,
                high_pin=None,  # Special placeholder for VMM operation
                sequence=read_sequence,
                read_voltage=self.read_voltage,
                differential=differential,
                reading_mode="after_each",
                debug=debug
            )
            
            # Store extra metadata for VMM execution
            config.metadata.update({
                'active_cols': active_cols,
                'active_voltages': active_voltages,
                'grounded_cols': grounded_cols,
                'is_vmm_row': True
            })
            
            read_configs[i] = config
        
        # Create the complete VMM configuration
        vmm_config = {
            'input_vector': input_vector,
            'voltage_vector': voltage_vector,
            'active_cols': active_cols,
            'grounded_cols': grounded_cols,
            'read_configs': read_configs,
            'max_input': max_input,
            'differential': differential
        }
        
        if debug:
            print(f"Created VMM configuration:")
            print(f"  Input vector shape: {input_vector.shape}")
            print(f"  Maximum input value: {max_input}")
            print(f"  Active columns: {len(active_cols)}")
            print(f"  Grounded columns: {len(grounded_cols)}")
        
        return vmm_config
    
    def visualize_vmm_config(self, vmm_config):
        """
        Visualize the VMM configuration.
        
        Parameters:
        -----------
        vmm_config : Dict
            VMM configuration from create_vmm_config()
        """
        plotting.visualize_vmm_config(vmm_config)

    def _read_device(self, row_pin, col_pin, read_voltage=None, differential=True, debug=False):
        """Helper method to read a single device using pulse sequence object"""
        if read_voltage is None:
            read_voltage = self.read_voltage
            
        from ..instruments.pulse_sequence import PulseSequence
        
        # Create a pulse sequence
        seq = PulseSequence(differential=differential)
        seq.add_read_pulse(
            voltage=read_voltage,
            differential=differential
        )
        
        # Apply the sequence
        result = self.pulse_measurement.apply_pulse_sequence_object(
            low_pin=row_pin,
            high_pin=col_pin,
            sequence_obj=seq,
            debug=debug
        )
        
        # Process and return results
        if result.currents and len(result.currents) > 0:
            current = result.currents[0]
            resistance = result.resistances[0] if result.resistances else abs(read_voltage / current)
            return {
                'current': current,
                'resistance': resistance,
                'conductance': 1/resistance if resistance != 0 else 0
            }
        else:
            return {'current': 0, 'resistance': float('inf'), 'conductance': 0}
    
    def _apply_mitigation_scheme(self, scheme, active_rows, active_cols, read_voltage=None):
        """
        Apply the selected sneak path mitigation scheme.
        
        Parameters:
        -----------
        scheme : str
            Mitigation scheme: "none", "ground_unused", "v_2", "v_3"
        active_rows : list
            List of row pins that are actively used
        active_cols : list
            List of column pins that are actively used
        read_voltage : float
            Read voltage to use (defaults to self.read_voltage)
        """
        if read_voltage is None:
            read_voltage = self.read_voltage
            
        if scheme == "none":
            # Do nothing, leave pins floating
            pass
        elif scheme == "ground_unused":
            self._apply_ground_unused(active_rows, active_cols)
        elif scheme == "v_2":
            self._apply_v2_scheme(active_rows, active_cols, read_voltage)
        elif scheme == "v_3":
            self._apply_v3_scheme(active_rows, active_cols, read_voltage)
        else:
            raise ValueError(f"Unknown mitigation scheme: {scheme}")

    def _apply_ground_unused(self, active_rows, active_cols):
        """Ground all unused pins."""
        for row_pin in self.row_pins:
            if row_pin not in active_rows:
                self.instrument.ground_pin(row_pin)
                
        for col_pin in self.col_pins:
            if col_pin not in active_cols:
                self.instrument.ground_pin(col_pin)

    def _apply_v2_scheme(self, active_rows, active_cols, read_voltage):
        """
        Apply V/2 biasing scheme to minimize sneak paths.
        
        - Selected rows: Will be set to input voltage V later
        - Selected columns: Will be set to ground (0V) later
        - Unselected rows and columns: V/2
        """
        half_voltage = read_voltage / 2
        
        # Set all unselected rows and columns to V/2
        for row_pin in self.row_pins:
            if row_pin not in active_rows:
                self.instrument.write_voltage(row_pin, half_voltage)
                
        for col_pin in self.col_pins:
            if col_pin not in active_cols:
                self.instrument.write_voltage(col_pin, half_voltage)

    def _apply_v3_scheme(self, active_rows, active_cols, read_voltage):
        """
        Apply V/3 biasing scheme for optimal sneak path mitigation.
        
        - Selected rows: Will be set to input voltage V later
        - Selected columns: Will be set to ground (0V) later
        - Unselected rows: 2V/3 
        - Unselected columns: V/3
        """
        v = read_voltage
        
        # Set unselected rows to 2V/3
        for row_pin in self.row_pins:
            if row_pin not in active_rows:
                self.instrument.write_voltage(row_pin, 2*v/3)
                
        # Set unselected columns to V/3
        for col_pin in self.col_pins:
            if col_pin not in active_cols:
                self.instrument.write_voltage(col_pin, v/3)
    
    def execute_vmm_config(self, vmm_config, sneak_path_mitigation=None,debug=False):
        """
        Execute a VMM configuration.
        
        Parameters:
        -----------
        vmm_config : Dict
            VMM configuration from create_vmm_config()
        debug : bool
            Enable debug output
            
        Returns:
        --------
        numpy.ndarray
            Result vector of the VMM operation
        """
        # Extract information from config
        max_input = vmm_config['max_input']
        active_cols = vmm_config['active_cols']
        active_voltages = [v for i, v in enumerate(vmm_config['voltage_vector']) 
                          if abs(v) > 1e-6]
        grounded_cols = vmm_config['grounded_cols']
        read_configs = vmm_config['read_configs']
        differential = vmm_config['differential']
        
        # Use the specified mitigation scheme or fall back to class default
        if sneak_path_mitigation is None:
            sneak_path_mitigation = self.sneak_path_mitigation
            
        if debug:
            print(f"Using sneak path mitigation: {sneak_path_mitigation}")
    
        # Result storage
        vmm_result = np.zeros(self.rows)
        
        if debug:
            print(f"Executing VMM operation with {len(active_cols)} active columns")
        
        # apply the mitigation scheme to set bias voltages
        self._apply_mitigation_scheme(
            scheme=sneak_path_mitigation,
            active_rows=self.row_pins,  # All rows are active in bias setup
            active_cols=active_cols,
            read_voltage=self.read_voltage
    )
        # For each row, perform the read operation
        for i, row_pin in enumerate(self.row_pins):
            if debug:
                print(f"Reading row {i} (pin {row_pin})")
            
            # Create a pulse sequence for this row's VMM operation
            seq = PulseSequence(period=10000, differential=differential)

            # First, add all column voltage configurations as write pulses
            for col_pin, voltage in zip(active_cols, active_voltages):
                # Add each column voltage as a separate write pulse
                seq.add_write_pulse(
                    voltage=voltage, 
                    pulse_width=1000,
                    differential=differential
                )


            # Pre-calculate all cluster timings
            seq.calculate_cluster_timings(row_pin, active_cols[0] if active_cols else self.col_pins[0])
            
            # Next, add the read pulse for this row
            seq.add_read_pulse(
                voltage=self.read_voltage,
                differential=differential
            )
            
            # For grounded columns, add these to sequence metadata
            if grounded_cols:
                seq.grounded_pins = grounded_cols
            
            try:
                # Execute the sequence using the enhanced pulse sequence handler
                result = self.pulse_measurement.apply_pulse_sequence_object(
                    low_pin=row_pin,
                    high_pin=active_cols[0] if active_cols else self.col_pins[0],
                    sequence_obj=seq,
                    debug=debug
                )
                
                # Extract the current reading from the result
                if result.currents and len(result.currents) > 0:
                    vmm_result[i] = abs(result.currents[0])
                else:
                    print(f"Warning: No current reading obtained for row {i}")
                    
            except Exception as e:
                print(f"Error reading from row {i} (pin {row_pin}): {e}")
                
            # Reset all pins after each row read
            self.instrument.ground_all()
            self.instrument.execute()
            self.instrument.wait()
        
        self.last_vmm_result = vmm_result
        
        return vmm_result
    
    def vmm_operation(self, input_vector, ground_unused=True, differential=True, 
                      sneak_path_mitigation=None, visualize=False, debug=False):
        """
        Perform Vector-Matrix Multiplication (convenience wrapper).
        
        Parameters:
        -----------
        input_vector : numpy.ndarray
            1D input vector to multiply with the conductance matrix
        ground_unused : bool
            Whether to ground unused columns to mitigate sneak paths
        differential : bool
            If True, use differential reading mode
        visualize : bool
            If True, visualize the configuration before execution
        debug : bool
            Enable debug output
            
        Returns:
        --------
        numpy.ndarray
            Result of VMM operation
        """
        if ground_unused is not None and sneak_path_mitigation is not None:
            print("Warning: Both ground_unused and sneak_path_mitigation specified. Using sneak_path_mitigation.")
        
        if ground_unused is True and sneak_path_mitigation is None:
            sneak_path_mitigation = "ground_unused"
        
        # Use the class default if not specified
        if sneak_path_mitigation is None:
            sneak_path_mitigation = self.sneak_path_mitigation
        
        # Create the configuration
        vmm_config = self.create_vmm_config(
            input_vector=input_vector,
            ground_unused=False,
            differential=differential,
            debug=debug
        )

        if debug:
            print(f"Using sneak path mitigation: {sneak_path_mitigation}")
        
        result = self.execute_vmm_config(vmm_config, sneak_path_mitigation=sneak_path_mitigation, debug=debug)
        
        # Visualize if requested
        if visualize:
            self.visualize_vmm_config(vmm_config)
        
        # Execute the configuration
        return result
    
    def read_conductance_matrix(self, read_voltage=None, differential=True, debug=False):
        """
        Read the entire conductance matrix of the crossbar.
        
        Parameters:
        -----------
        read_voltage : float
            Voltage to use for reading. If None, uses default read_voltage.
        differential : bool
            If True, use differential reading. If False, apply voltage only to the high pin.
        debug : bool
            Enable debug output
            
        Returns:
        --------
        numpy.ndarray
            2D array of conductance values in Siemens
        """
        if read_voltage is None:
            read_voltage = self.read_voltage
            
        conductance_matrix = np.zeros((self.rows, self.cols))
        
        if debug:
            print(f"Reading {self.rows}x{self.cols} conductance matrix with Vread={read_voltage}V")
            
        # Read each device in the crossbar
        for i, row_pin in enumerate(self.row_pins):
                for j, col_pin in enumerate(self.col_pins):
                    # Create a pulse sequence with specific differential setting
                    seq = PulseSequence(differential=differential)
                    
                    # Add a read pulse with proper configuration
                    seq.add_read_pulse(
                        voltage=read_voltage,
                        differential=differential
                    )
                    
                    # Apply using your enhanced method
                    result = self.pulse_measurement.apply_pulse_sequence_object(
                        low_pin=row_pin,
                        high_pin=col_pin,
                        sequence_obj=seq,
                        debug=(debug and (i*self.cols + j) % 10 == 0)
                    )
                
                    # Get the current measurement
                    if result.currents and len(result.currents) > 0:
                        current = abs(result.currents[0])
                        # Calculate conductance (G = I/V)
                        if abs(current) > 1e-12:  # Avoid division by very small numbers
                            conductance = abs(current / read_voltage)
                        else:
                            conductance = 0
                            
                        conductance_matrix[i, j] = conductance
                    
                if debug and ((i*self.cols + j + 1) % 10 == 0 or (i*self.cols + j + 1) == self.rows*self.cols):
                    print(f"Read progress: {i*self.cols + j + 1}/{self.rows*self.cols} devices")
        
        # Store for later use
        self.conductance_matrix = conductance_matrix
        return conductance_matrix

 

    def read_resistance_matrix(self, read_voltage=None, differential=True, debug=False):
        """
        Read the entire resistance matrix of the crossbar.
        
        Parameters:
        -----------
        read_voltage : float
            Voltage to use for reading. If None, uses default read_voltage.
        differential : bool
            If True, use differential reading. If False, apply voltage only to the high pin.
        debug : bool
            Enable debug output
            
        Returns:
        --------
        numpy.ndarray
            2D array of resistance values in Ohms
        """
        # Use conductance measurements and convert to resistance
        conductance_matrix = self.read_conductance_matrix(read_voltage, differential, debug)
        
        # Convert conductances to resistances
        resistance_matrix = np.zeros_like(conductance_matrix)
        mask = conductance_matrix > 1e-12
        resistance_matrix[mask] = 1.0 / conductance_matrix[mask]
        resistance_matrix[~mask] = float('inf')
        
        return resistance_matrix
    
    def visualize_crossbar(self, value_type='conductance', cmap='viridis', log_scale=True, 
                     figsize=(10, 8)):
        """
        Visualize the crossbar array using a heatmap with value annotations.
        
        Parameters:
        -----------
        value_type : str
            Type of values to display: 'conductance', 'resistance', or 'current'
        cmap : str
            Matplotlib colormap to use
        log_scale : bool
            Whether to use logarithmic scale for values
        figsize : tuple
            Figure size (width, height) in inches
            
        Returns:
        --------
        tuple
            (fig, ax) - Matplotlib figure and axis objects
        """
        # Make sure we have up-to-date measurements
        if np.all(self.conductance_matrix == 0):
            self.read_conductance_matrix()
            
        # Call the plotting function
        return plotting.visualize_crossbar(
            conductance_matrix=self.conductance_matrix,
            row_pins=self.row_pins,
            col_pins=self.col_pins,
            value_type=value_type,
            cmap=cmap,
            log_scale=log_scale,
            read_voltage=self.read_voltage,
            figsize=figsize
        )
    
    def compare_vmm_results(self, result, input_vector, visualize=True):
        """
        Compare measured VMM results with analytical expectations.
        
        Parameters:
        -----------
        result : numpy.ndarray
            Measured VMM result vector
        input_vector : numpy.ndarray
            Input vector used for the VMM operation
        visualize : bool
            If True, generate comparison plot
            
        Returns:
        --------
        dict
            Dictionary containing comparison metrics and arrays
        """
        # Calculate the expected result analytically
        analytical_result = self.conductance_matrix @ input_vector
        
        # Calculate absolute and relative errors
        absolute_error = np.abs(result - analytical_result)
        relative_error = np.zeros_like(absolute_error)
        mask = np.abs(analytical_result) > 1e-10
        relative_error[mask] = 100 * absolute_error[mask] / np.abs(analytical_result[mask])
        
        # Calculate overall metrics
        mean_abs_error = np.mean(absolute_error)
        mean_rel_error = np.mean(relative_error[mask]) if np.any(mask) else 0
        
        # Visualize the comparison if requested
        fig = None
        if visualize:
            fig = plt.figure(figsize=(10, 6))
            bar_width = 0.35
            index = np.arange(len(result))

            plt.bar(index - bar_width/2, analytical_result, bar_width, label='Analytical (Expected)')
            plt.bar(index + bar_width/2, result, bar_width, label='Measured (Actual)')

            plt.xlabel('Output Index')
            plt.ylabel('Value')
            plt.title('VMM Operation: Analytical vs. Measured Results')
            plt.xticks(index, [f'Row {i}' for i in range(len(result))])
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add error values as text on the figure
            for i in range(len(result)):
                plt.text(i, max(analytical_result[i], result[i]) * 1.05, 
                        f"Error: {relative_error[i]:.1f}%", 
                        ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.show()
        
        # Return comparison results as dictionary
        return {
            'analytical_result': analytical_result,
            'measured_result': result,
            'absolute_error': absolute_error,
            'relative_error': relative_error, 
            'mean_absolute_error': mean_abs_error,
            'mean_relative_error': mean_rel_error,
            'figure': fig
        }
    
    def save_conductance_matrix(self, filename=None):
        """Save the current conductance matrix to a file."""
        metadata = {
            'rows': self.rows,
            'cols': self.cols,
            'read_voltage': self.read_voltage,
            'timestamp': datetime.now().isoformat()
        }
        return data_io.save_conductance_matrix(self.conductance_matrix, filename, metadata)
    
    def load_conductance_matrix(self, filename):
        """Load a conductance matrix from a file."""
        conductance_matrix, metadata = data_io.load_conductance_matrix(filename)
        self.conductance_matrix = conductance_matrix
        return metadata