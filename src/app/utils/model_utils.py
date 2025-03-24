import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger("ModelUtils")


class ModelManager:
    """Utility class for managing YOLO model files"""

    DEFAULT_MODEL_DIR = "models"
    CONFIG_FILE = "models/model_config.txt"
    DEFAULT_MODEL = "yolov8n.pt"

    @staticmethod
    def get_available_models(model_dir: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Scan the models directory for .pt files
        Returns a list of tuples with (file_path, display_name)
        """
        model_dir = model_dir or ModelManager.DEFAULT_MODEL_DIR
        models = []

        try:
            # Ensure the directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                logger.info(f"Created models directory: {model_dir}")
                return models

            # Scan for .pt files
            for file in os.listdir(model_dir):
                if file.endswith(".pt"):
                    file_path = os.path.join(model_dir, file)
                    # Format display name (remove extension and add size info)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    display_name = f"{file} ({size_mb:.1f} MB)"
                    models.append((file_path, display_name))

            models.sort(key=lambda x: x[0])  # Sort by file path

        except Exception as e:
            logger.error(f"Error scanning model directory: {str(e)}")

        return models

    @staticmethod
    def save_last_model(model_path: str) -> bool:
        """Save the last used model path to config file"""
        try:
            with open(ModelManager.CONFIG_FILE, "w") as f:
                f.write(model_path)
            return True
        except Exception as e:
            logger.error(f"Error saving last model: {str(e)}")
            return False

    @staticmethod
    def get_last_model() -> str:
        """Get the last used model path from config file"""
        # Just return None as we don't want to auto-load any model
        return None
