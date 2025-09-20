import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="heartpy")

import biosppy
import neurokit2 as nk
import heartpy as hp
import wfdb

# Test basic imports and version access
print("=== Biosignal Libraries Test ===")

try:
    print("BioSPPy OK:", biosppy.__version__)
except AttributeError:
    print("BioSPPy OK (version not accessible)")

print("NeuroKit2 OK:", nk.__version__)
print("HeartPy OK:", hp.__version__)
print("WFDB OK:", wfdb.__version__)

# Test pyPPG if available
try:
    import pyPPG
    try:
        print("pyPPG OK:", pyPPG.__version__)
    except AttributeError:
        print("pyPPG OK (version not accessible)")
except ImportError:
    print("pyPPG not installed (optional)")

print("\n=== Basic Functionality Test ===")

# Quick functionality test
try:
    # Test NeuroKit2 PPG simulation
    ppg_signal = nk.ppg_simulate(duration=5, sampling_rate=100)
    print("NeuroKit2 PPG simulation: OK")
    
    # Test BioSPPy import of PPG module
    from biosppy.signals import ppg as biosppy_ppg
    print("BioSPPy PPG module: OK")
    
    # Test HeartPy basic functionality
    import heartpy.datautils
    print("HeartPy datautils: OK")
    
    print("\nAll biosignal libraries are working correctly!")
    
except Exception as e:
    print(f"Functionality test failed: {e}")
