"""
Compatibility utilities for handling different versions of libraries
"""

import logging
import importlib.metadata

logger = logging.getLogger("Compatibility")


def get_package_version(package_name):
    """Get the installed version of a package"""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Error getting {package_name} version: {str(e)}")
        return None


def log_dependency_versions():
    """Log all relevant dependency versions"""
    dependencies = ["supervision", "ultralytics", "opencv-python", "numpy", "PyQt6"]

    logger.info("Dependency versions:")
    for package in dependencies:
        version = get_package_version(package)
        logger.info(f"  {package}: {version if version else 'Not installed'}")
