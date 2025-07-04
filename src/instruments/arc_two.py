from pyarc2 import Instrument, find_ids, BiasOrder
import os

def connect_arc_two(firmware_path=None, device_id=None):
    """
    Connect to the ArC Two board.

    Parameters:
    -----------
    firmware_path : str, optional
        Path to the firmware file (.bin). If None, will auto-detect.
    device_id : str, optional
        The specific device ID to connect to. If None, will use the first available.

    Returns:
    --------
    Instrument
        An instance of the Instrument class connected to the ArC Two board.
    """
    try:
        # Find available devices
        ids = find_ids()
        if len(ids) == 0:
            print("No ArC Two devices found. Please connect a device and try again.")
            return None
            
        # Use specified device_id or default to first available
        device_id = device_id if device_id is not None else ids[0]
        print(f"Using ArC Two device: {device_id}")
        
        # Set default firmware path if not provided
        if firmware_path is None:
            possible_paths = [
                '/home/abaigol/.local/share/arc2control/firmware/efm03_20240918.bin', # Default path
                os.path.join(os.path.dirname(__file__), "firmware/efm03_20240918.bin") # Relative to this file
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    firmware_path = path
                    break
                    
            if firmware_path is None:
                print("No firmware file found! Please specify the firmware_path parameter.")
                return None
                
        print(f"Using firmware: {firmware_path}")
        
        # Initialize the instrument with device ID and firmware
        instrument = Instrument(device_id, firmware_path)
        
        # Check if connection was successful
        print("Successfully connected to ArC Two board.")
        return instrument
        
    except Exception as e:
        print(f"Failed to connect to ArC Two board: {e}")
        return None