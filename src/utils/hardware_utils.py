def get_pin_cluster(pin):
    """
    Calculate which hardware cluster a pin belongs to.
    
    Args:
        pin (int): Pin number
        
    Returns:
        int: Cluster number (0-7)
    """
    return pin // 8

def get_cluster_timing(pins, pulse_width):
    """
    Create cluster timing array based on pins and pulse width.
    
    Args:
        pins (list): List of pins to activate
        pulse_width (int): Pulse width in nanoseconds
        
    Returns:
        list: Cluster timing array for ArC2 hardware
    """
    cluster_timings = [None] * 8  # ArC2 has 8 clusters
    
    # Set timing for each cluster that contains at least one pin
    for pin in pins:
        cluster = get_pin_cluster(pin)
        cluster_timings[cluster] = pulse_width
    
    return cluster_timings