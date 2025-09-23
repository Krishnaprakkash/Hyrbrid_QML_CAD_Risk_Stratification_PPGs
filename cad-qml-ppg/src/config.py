# src/config.py - Central path configuration
from pathlib import Path
import os

class ProjectPaths:
    """Centralized path management for portability"""
    
    def __init__(self):
        # Get project root (wherever the main project folder is)
        self.PROJECT_ROOT = Path(__file__).parent.parent  # Goes up from src/ to project root
        
        # All paths relative to project root
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.MODELS_DIR = self.PROJECT_ROOT / "models"
        self.RESULTS_DIR = self.PROJECT_ROOT / "results"
        self.SRC_DIR = self.PROJECT_ROOT / "src"
        
        # PhysioNet specific paths
        self.PHYSIONET_DIR = self.RAW_DATA_DIR / "physionet_2015"
        self.PHYSIONET_TRAINING = self.PHYSIONET_DIR / "training"
        
        # Create directories if they don't exist
        self.create_directories()
    
    def create_directories(self):
        """Create all necessary directories"""
        for path in [self.DATA_DIR, self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, 
                     self.MODELS_DIR, self.RESULTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_relative_path(self, path: Path) -> str:
        """Convert path to string relative to project root"""
        return str(path.relative_to(self.PROJECT_ROOT))

# Global path instance
PATHS = ProjectPaths()
