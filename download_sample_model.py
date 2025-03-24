import os
import sys
import requests
from tqdm import tqdm
from pathlib import Path


def download_file(url, filename):
    """
    Download a file from a URL with a progress bar
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    # Show download progress
    progress_bar = tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {os.path.basename(filename)}",
    )

    # Write file
    with open(filename, "wb") as file:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Download failed")
        return False

    return True


def main():
    # Base YOLOv8n model URL (using Ultralytics' latest release)
    model_url = (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    )
    model_path = os.path.join("models", "yolov8n.pt")  # Changed from hazard_yolov8n.pt

    print("This script will download a sample YOLOv8n model.")
    print("Note: This is a general-purpose model, not specialized for hazard labels.")
    print(
        "For optimal results, you should train a custom model on GHS hazard label data."
    )
    print(f"The model will be saved to: {model_path}")

    if os.path.exists(model_path):
        overwrite = input(f"Model already exists at {model_path}. Overwrite? (y/n): ")
        if overwrite.lower() != "y":
            print("Download canceled.")
            return

    print("\nDownloading model...")
    success = download_file(model_url, model_path)

    if success:
        print(f"\nModel downloaded successfully to {model_path}")
        print("You can now run the application using:")
        print("  - Windows: Double-click run.bat")
        print("  - All platforms: python src/app.py")
    else:
        print("\nFailed to download the model. Please check your internet connection.")


if __name__ == "__main__":
    main()
