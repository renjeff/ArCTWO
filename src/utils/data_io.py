import os
import csv
import numpy as np
import h5py
import toml
from datetime import datetime


def save_measurement_results(result, filename: str, format: str = 'h5'):
    """
    Save measurement results to a file.
    
    Parameters:
    -----------
    result : MeasurementResult
        Measurement results to save
    filename : str
        Output filename (without extension)
    format : str
        File format: 'csv', 'h5', or 'npz'
    """
    from ..measurement.base import MeasurementResult
    
    # Type check
    if not isinstance(result, MeasurementResult):
        raise TypeError("result must be a MeasurementResult object")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Add timestamp to filename if not already included
    if not any(char.isdigit() for char in filename[-8:]):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_filename = f"results/{filename}_{timestamp}"
    else:
        full_filename = f"results/{filename}"
    
    if format == 'csv':
        with open(f"{full_filename}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['Pulse_Number', 'Voltage(V)', 'Current(A)', 
                           'Resistance(Ohm)', 'Pulse_Width(ns)', 'Timestamp'])
            
            # Write data rows
            for i in range(len(result.voltages)):
                writer.writerow([
                    i+1, 
                    result.voltages[i], 
                    result.currents[i],
                    result.resistances[i], 
                    result.pulse_widths[i],
                    result.timestamps[i]
                ])
            
            # Write metadata
            writer.writerow([])
            writer.writerow(['Metadata:'])
            for key, value in result.metadata.items():
                writer.writerow([key, value])
                
        print(f"Results saved to {full_filename}.csv")
        
    elif format == 'h5':
        try:
            import h5py
            with h5py.File(f"{full_filename}.h5", 'w') as f:
                # Create data group
                data_group = f.create_group('data')
                data_group.create_dataset('voltages', data=np.array(result.voltages))
                data_group.create_dataset('currents', data=np.array(result.currents))
                data_group.create_dataset('resistances', data=np.array(result.resistances))
                data_group.create_dataset('pulse_widths', data=np.array(result.pulse_widths))
                data_group.create_dataset('timestamps', data=np.array(result.timestamps))
                if hasattr(result, 'is_read') and result.is_read is not None:
                    data_group.create_dataset('is_read', data=np.array(result.is_read))
                # Create metadata group
                meta_group = f.create_group('metadata')
                
                # Store metadata with type checking
                for key, value in result.metadata.items():
                    try:
                        # Try to store directly
                        meta_group.attrs[key] = value
                    except (TypeError, ValueError):
                        # If direct storage fails, convert complex objects to strings
                        try:
                            meta_group.attrs[key] = str(value)
                        except Exception as e:
                            print(f"Warning: Could not save metadata '{key}': {e}")
            
            print(f"Results saved to {full_filename}.h5")
        except ImportError:
            print("h5py module not found. Please install it or use a different format.")
            
    elif format == 'npz':
        # Save as NumPy compressed format
        np.savez_compressed(
            full_filename,
            voltages=np.array(result.voltages),
            currents=np.array(result.currents),
            resistances=np.array(result.resistances),
            pulse_widths=np.array(result.pulse_widths),
            timestamps=np.array(result.timestamps),
            metadata=result.metadata
        )
        print(f"Results saved to {full_filename}.npz")
        
    else:
        print(f"Unknown format: {format}")
        
    return full_filename + f".{format}"

def load_measurement_results(filename: str):
    """
    Load measurement results from a file.
    """
    from ..measurement.base import MeasurementResult
    import h5py
    import os
    import numpy as np

    # Add extension if not present
    if not filename.endswith('.h5'):
        filename = f"{filename}.h5"

    # Check if file exists
    if not os.path.exists(filename):
        # Try looking in results directory
        results_dir = os.path.join(os.getcwd(), "results")
        alt_path = os.path.join(results_dir, filename)
        if os.path.exists(alt_path):
            filename = alt_path
        else:
            raise FileNotFoundError(f"File not found: {filename}")

    print(f"Loading measurement data from {filename}")

    # Load the data from h5 file
    with h5py.File(filename, 'r') as f:
        # Load all arrays present in the data group
        data_group = f['data']
        data_arrays = {}
        for key in data_group.keys():
            data_arrays[key] = np.array(data_group[key])

        # Load metadata
        metadata = {}
        if 'metadata' in f:
            meta_group = f['metadata']
            for key, value in meta_group.attrs.items():
                metadata[key] = value

    # Build MeasurementResult using all loaded arrays
    result = MeasurementResult(
        voltages=data_arrays.get('voltages'),
        currents=data_arrays.get('currents'),
        resistances=data_arrays.get('resistances'),
        pulse_widths=data_arrays.get('pulse_widths'),
        timestamps=data_arrays.get('timestamps'),
        is_read=data_arrays.get('is_read', None),
        metadata=metadata
    )
    return result

def load_conductance_matrix(filename):
    """
    Load a conductance matrix from HDF5 format.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file
        
    Returns:
    --------
    tuple
        (conductance_matrix, metadata)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    with h5py.File(filename, 'r') as f:
        conductance_matrix = f['conductance_matrix'][()]
        
        metadata = {}
        if 'metadata' in f:
            meta_group = f['metadata']
            
            # Load attributes
            for key, value in meta_group.attrs.items():
                metadata[key] = value
            
            # Load datasets
            for key in meta_group.keys():
                metadata[key] = meta_group[key][()]
    
    return conductance_matrix, metadata

def save_conductance_matrix(matrix, filename=None, metadata=None):
    """
    Save a conductance matrix to HDF5 format.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        The conductance matrix to save
    filename : str, optional
        Output filename (without extension). If None, generates a timestamped filename.
    metadata : dict, optional
        Additional metadata to save
        
    Returns:
    --------
    str
        Path to the saved file
    """

    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(results_dir, f"conductance_matrix_{timestamp}")
    elif not os.path.isabs(filename):
        # If relative path, prepend results directory
        filename = os.path.join(results_dir, filename)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Add .h5 extension if not present
    if not filename.endswith('.h5'):
        filename = f"{filename}.h5"
        
    with h5py.File(filename, 'w') as f:
        # Save the matrix
        f.create_dataset('conductance_matrix', data=matrix)
        
        # Save metadata if provided
        if metadata:
            meta_group = f.create_group('metadata')
            for key, value in metadata.items():
                try:
                    meta_group.attrs[key] = value
                except (TypeError, ValueError):
                    try:
                        meta_group.attrs[key] = str(value)
                    except Exception as e:
                        print(f"Warning: Could not save metadata '{key}': {e}")
    
    print(f"Conductance matrix saved to {filename}")
    return filename

def load_crossbar_config(config_file: str) -> dict:
    """
    Load crossbar configuration from a TOML file.
    
    Parameters:
    -----------
    config_file : str
        Path to the TOML configuration file
        
    Returns:
    --------
    dict
        Dictionary containing the parsed configuration
    """

    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Crossbar config file not found: {config_file}")
    
    try:
        config = toml.load(config_file)
        
        # Validate required sections
        if 'mapping' not in config:
            raise ValueError("TOML file missing required [mapping] section")
            
        if 'words' not in config['mapping'] or 'bits' not in config['mapping']:
            raise ValueError("TOML [mapping] section must contain 'words' and 'bits' arrays")
        
        # Extract configuration info
        result = {
            'name': config.get('config', {}).get('name', os.path.basename(config_file).split('.')[0]),
            'row_pins': config['mapping']['bits'],  # Note: bits = row pins in our convention
            'col_pins': config['mapping']['words'],  # Note: words = column pins in our convention
            'rows': len(config['mapping']['bits']),
            'cols': len(config['mapping']['words'])
        }
        
        # Get additional configuration if available
        if 'config' in config:
            if 'words' in config['config']:
                result['cols'] = config['config']['words']
            if 'bits' in config['config']:
                result['rows'] = config['config']['bits']
        
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to parse crossbar config: {str(e)}")