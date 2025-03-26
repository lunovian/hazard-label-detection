import cv2
import time
import platform
import logging
import numpy as np
from enum import Enum, auto
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QMutex, QWaitCondition, QTimer
from typing import Dict, List, Optional, Tuple, Union, Any
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CameraModel")


class CameraBackend(Enum):
    """Camera backend options based on platform"""

    ANY = auto()
    DSHOW = auto()  # Windows
    MSMF = auto()  # Windows
    V4L = auto()  # Linux
    GSTREAMER = auto()  # Linux
    AVFOUNDATION = auto()  # macOS

    @staticmethod
    def get_preferred_backend():
        """Returns the preferred backend for the current platform"""
        system = platform.system().lower()
        if system == "windows":
            return CameraBackend.DSHOW  # DirectShow is preferred on Windows
        elif system == "linux":
            return CameraBackend.V4L  # V4L2 is preferred on Linux
        elif system == "darwin":
            return CameraBackend.ANY  # Default on macOS
        return CameraBackend.ANY  # Default fallback


class CameraInfo:
    """Class to store camera information"""

    def __init__(
        self, id: int, name: str = "", api_name: str = "", is_virtual: bool = False
    ):
        self.id = id
        self.name = name if name else f"Camera {id}"
        self.api_name = api_name
        self.is_virtual = is_virtual

    def __str__(self):
        return f"{self.name} (ID: {self.id}, API: {self.api_name})"


class CameraThread(QThread):
    """Thread for camera capture to avoid blocking the UI"""

    frame_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    connected = pyqtSignal(bool)
    status_message = pyqtSignal(str)
    progress_updated = pyqtSignal(int)  # 0-100 progress value

    # Constants for timeout control
    CONNECT_TIMEOUT = 10.0  # Increased from 5.0 to 10.0 seconds
    FRAME_TIMEOUT = 2.0  # Increased from 0.5 to 2.0 seconds
    MAX_FRAME_RETRIES = 3  # New constant for frame retry attempts

    def __init__(self, camera_id=0, resolution=(640, 480), fps=30, backend=None):
        super().__init__()
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.backend = backend if backend else CameraBackend.get_preferred_backend()
        self.running = False
        self.cap = None
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.retry_count = 0
        self.max_retries = 2  # Reduced from 3 to speed up
        self.retry_delay = 1  # Reduced from 2 to speed up
        self.force_stop = False

        # Timing diagnostics
        self.timing = {}

        # Initialization stages for better progress reporting
        self.init_stages = {
            "prepare": 10,
            "open": 30,
            "configure": 20,
            "first_frame": 40,
        }

    def run(self):
        """Thread main loop for capturing frames"""
        self.retry_count = 0
        self.timing = {}

        while self.retry_count < self.max_retries and not self.force_stop:
            try:
                self.status_message.emit(f"Initializing camera {self.camera_id}...")
                success = self._initialize_camera()

                if not success:
                    raise IOError("Camera initialization timed out")

                # If we get here, camera is successfully initialized
                self.running = True
                self.connected.emit(True)

                # Main capture loop
                self._run_capture_loop()

                # Exit loop if we're stopping intentionally
                if self.force_stop:
                    break

                # If we get here with running still True, there was an error in the capture loop
                if self.running:
                    self.retry_count += 1
                    if self.retry_count < self.max_retries:
                        self.status_message.emit(
                            f"Camera error, retrying ({self.retry_count}/{self.max_retries})..."
                        )
                        self.progress_updated.emit(0)  # Reset progress
                        self.msleep(self.retry_delay * 1000)
                        continue

                break

            except Exception as e:
                self.retry_count += 1
                error_msg = f"Camera error: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Camera timing: {self.timing}")

                if self.retry_count < self.max_retries and not self.force_stop:
                    self.status_message.emit(
                        f"Camera error, retrying ({self.retry_count}/{self.max_retries})..."
                    )
                    self.progress_updated.emit(0)  # Reset progress
                    self.msleep(self.retry_delay * 1000)
                else:
                    self.error.emit(error_msg)
                    break
            finally:
                # Ensure camera is released if an exception occurred
                self._release_camera()

        # Thread ending, make sure we notify if we couldn't connect
        if self.retry_count >= self.max_retries:
            self.connected.emit(False)
            self.error.emit(
                f"Failed to initialize camera after {self.max_retries} attempts"
            )
            # Log timing information
            logger.error(f"Camera initialization timing: {self.timing}")

    def _initialize_camera(self) -> bool:
        """Initialize the camera with timeout control"""
        # Prepare stage - 10%
        self.timing["prepare_start"] = time.time()
        self.progress_updated.emit(0)
        self.status_message.emit(f"Preparing to connect to camera {self.camera_id}...")

        # Try to prepare fast backend options based on platform
        backend_int = self._get_backend_int()
        self.progress_updated.emit(self.init_stages["prepare"])
        self.timing["prepare_end"] = time.time()

        # Open stage - 30%
        self.timing["open_start"] = time.time()
        self.status_message.emit(f"Opening camera {self.camera_id}...")

        # Use a timer to enforce timeout
        timer = threading.Timer(self.CONNECT_TIMEOUT, self._connection_timeout)
        timer.daemon = True
        timer.start()

        try:
            if isinstance(self.camera_id, str):
                # Handle URL/path case
                self.cap = cv2.VideoCapture(self.camera_id)
            else:
                # Hardware camera case with selected backend
                self.cap = cv2.VideoCapture(self.camera_id, backend_int)

            # Check if camera was opened
            if not self.cap.isOpened():
                timer.cancel()
                return False

            self.timing["open_end"] = time.time()
            self.progress_updated.emit(
                self.init_stages["prepare"] + self.init_stages["open"]
            )

            # Configure stage - 20%
            self.timing["configure_start"] = time.time()
            self.status_message.emit(f"Configuring camera {self.camera_id}...")

            # Configure camera properties - limit to essential properties
            # Set resolution (with error handling)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # Only set FPS if not default
            if self.fps != 30:
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.timing["configure_end"] = time.time()
            self.progress_updated.emit(
                self.init_stages["prepare"]
                + self.init_stages["open"]
                + self.init_stages["configure"]
            )

            # First frame - 40%
            self.timing["first_frame_start"] = time.time()
            self.status_message.emit("Getting first frame...")

            # Set a frame timeout to prevent stalling
            frame_timer = threading.Timer(self.FRAME_TIMEOUT, self._frame_timeout)
            frame_timer.daemon = True
            frame_timer.start()

            # Try to get the first frame
            ret, first_frame = self.cap.read()
            frame_timer.cancel()

            if not ret:
                raise IOError("Failed to get first frame from camera")

            # Success - cancel main timer
            timer.cancel()

            self.timing["first_frame_end"] = time.time()
            self.progress_updated.emit(100)  # Full progress

            # Log success
            logger.info(
                f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps"
            )
            self.status_message.emit(
                f"Camera ready: {actual_width}x{actual_height} @ {actual_fps:.1f}fps"
            )

            # Calculate timing totals
            self.timing["total_init_time"] = (
                self.timing["first_frame_end"] - self.timing["prepare_start"]
            )
            logger.info(
                f"Camera initialization completed in {self.timing['total_init_time']:.3f} seconds"
            )

            return True

        except Exception as e:
            timer.cancel()
            if "frame_timer" in locals():
                frame_timer.cancel()
            logger.error(f"Error during camera initialization: {str(e)}")
            return False

    def _get_backend_int(self) -> int:
        """Get the actual CV2 API constant for the selected backend"""
        if self.backend == CameraBackend.DSHOW:
            return cv2.CAP_DSHOW
        elif self.backend == CameraBackend.MSMF:
            return cv2.CAP_MSMF
        elif self.backend == CameraBackend.V4L:
            return cv2.CAP_V4L
        elif self.backend == CameraBackend.GSTREAMER:
            return cv2.CAP_GSTREAMER
        else:
            # Use best default for platform
            system = platform.system().lower()
            if system == "windows":
                return cv2.CAP_DSHOW
            elif system == "linux":
                return cv2.CAP_V4L
            return cv2.CAP_ANY

    def _connection_timeout(self):
        """Handle connection timeout"""
        logger.error("Camera connection timeout")
        self.force_stop = True
        if self.cap:
            self._release_camera()

    def _frame_timeout(self):
        """Handle frame read timeout"""
        logger.error("Frame read timeout")
        # Don't force stop immediately, let retry mechanism handle it
        if self.cap and not self.force_stop:
            # Try to restart the camera capture
            self.cap.release()
            self.cap = cv2.VideoCapture(self.camera_id, self._get_backend_int())

    def _run_capture_loop(self):
        """Run the main frame capture loop"""
        frame_interval = 1.0 / self.fps
        frame_count = 0
        error_count = 0
        last_frame_time = time.time()
        frame_retry_count = 0

        while self.running and not self.force_stop:
            start_time = time.time()

            if not self.cap or not self.cap.isOpened():
                logger.error("Camera disconnected")
                self.error.emit("Camera disconnected")
                break

            try:
                # Set a frame read timeout
                frame_timer = threading.Timer(self.FRAME_TIMEOUT, self._frame_timeout)
                frame_timer.daemon = True
                frame_timer.start()

                ret, frame = self.cap.read()
                frame_timer.cancel()

                if not ret:
                    frame_retry_count += 1
                    error_count += 1
                    logger.warning(
                        f"Failed to read frame (attempt {frame_retry_count}/{self.MAX_FRAME_RETRIES})"
                    )

                    if frame_retry_count >= self.MAX_FRAME_RETRIES:
                        logger.error("Maximum frame retry attempts reached")
                        self.error.emit(
                            "Failed to capture frame after multiple attempts"
                        )
                        break

                    # Short delay before retry
                    time.sleep(0.1)
                    continue

                # Reset counters on successful frame read
                frame_retry_count = 0
                error_count = 0
                frame_count += 1

                # Calculate actual FPS
                current_time = time.time()
                actual_fps = (
                    1.0 / (current_time - last_frame_time)
                    if last_frame_time
                    else self.fps
                )
                last_frame_time = current_time

                # Convert and emit frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(frame_rgb)

                # FPS control
                if self.fps < 30:
                    elapsed = time.time() - start_time
                    sleep_time = max(0, frame_interval - elapsed)
                    if sleep_time > 0:
                        self.mutex.lock()
                        self.condition.wait(self.mutex, int(sleep_time * 1000))
                        self.mutex.unlock()

            except Exception as e:
                logger.error(f"Error during frame capture: {str(e)}")
                error_count += 1
                if error_count > 5:
                    self.error.emit(f"Persistent capture error: {str(e)}")
                    break

                # Add small delay before retry
                time.sleep(0.1)

    def _release_camera(self):
        """Helper method to safely release camera resources"""
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
        except Exception as e:
            logger.error(f"Error releasing camera: {str(e)}")

    def stop(self):
        """Stop the camera thread"""
        self.running = False
        self.force_stop = True

        # Wake up any sleeping wait condition
        self.mutex.lock()
        self.condition.wakeAll()
        self.mutex.unlock()

        # Wait for thread to finish (with timeout)
        self.wait(2000)  # Reduced timeout to 2 seconds

        # If thread is still running, terminate more aggressively
        if self.isRunning():
            logger.warning("Camera thread did not stop gracefully, forcing termination")
            self.terminate()
            self.wait(1000)  # Give it one more second

        # Make sure camera is released
        self._release_camera()


class CameraModel(QObject):
    """Model for managing camera capture"""

    frame_captured = pyqtSignal(np.ndarray)
    camera_error = pyqtSignal(str)
    camera_status = pyqtSignal(str)
    camera_connected = pyqtSignal(bool)
    camera_list_updated = pyqtSignal(list)  # Emits list of CameraInfo objects
    camera_progress = pyqtSignal(int)  # 0-100 progress value

    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.available_cameras = []
        self.current_camera_id = 0
        self.current_resolution = (640, 480)
        self.current_fps = 30
        self.current_backend = CameraBackend.get_preferred_backend()
        self.camera_cache = {}  # Cache of camera properties to speed up init

        # Initialize available cameras list with lower overhead approach
        self._refresh_available_cameras(use_fast_scan=True)

    def _refresh_available_cameras(self, use_fast_scan=False) -> None:
        """Refresh the list of available cameras"""
        self.camera_status.emit("Scanning for cameras...")

        if use_fast_scan:
            # Fast detection method - only check a few indices
            self.available_cameras = self._fast_detect_cameras()
        else:
            # Full detection method - more thorough but slower
            self.available_cameras = self._detect_cameras()

        self.camera_list_updated.emit(self.available_cameras)
        self.camera_status.emit(f"Found {len(self.available_cameras)} cameras")

        # Select a default camera if possible
        if self.available_cameras and self.current_camera_id is None:
            self.current_camera_id = self.available_cameras[0].id

    def _fast_detect_cameras(self) -> List[CameraInfo]:
        """Quick camera detection method - tries common indices only"""
        cameras = []
        system = platform.system().lower()

        # Choose appropriate backend
        backend = None
        if system == "windows":
            backend = cv2.CAP_DSHOW
        elif system == "linux":
            backend = cv2.CAP_V4L

        # Only check the most common camera indices (0,1)
        for i in range(2):
            try:
                # Quick open/close test
                if backend:
                    cap = cv2.VideoCapture(i, backend)
                else:
                    cap = cv2.VideoCapture(i)

                if cap.isOpened():
                    name = f"Camera {i}"
                    api_name = "Auto-detected"
                    cameras.append(CameraInfo(i, name, api_name))

                cap.release()
            except Exception as e:
                pass

        # Ensure at least camera 0 is in the list
        if not cameras:
            cameras.append(CameraInfo(0, "Default Camera", "Fallback"))

        return cameras

    def _detect_cameras(self) -> List[CameraInfo]:
        """Detect available cameras using platform-specific methods"""
        cameras = []

        # First try platform-specific detection
        system = platform.system().lower()

        try:
            if system == "windows":
                cameras = self._detect_cameras_windows()
            elif system == "linux":
                cameras = self._detect_cameras_linux()
            elif system == "darwin":
                cameras = self._detect_cameras_macos()
        except Exception as e:
            logger.error(f"Error in platform-specific camera detection: {str(e)}")

        # Fallback to generic detection if platform-specific failed or didn't find any
        if not cameras:
            cameras = self._detect_cameras_generic()

        # Always ensure camera ID 0 is available as fallback
        has_camera_0 = any(cam.id == 0 for cam in cameras)
        if not has_camera_0:
            cameras.append(CameraInfo(0, "Default Camera", "Fallback"))

        return cameras

    def _detect_cameras_generic(self) -> List[CameraInfo]:
        """Generic camera detection method using OpenCV"""
        cameras = []

        # Check the first few indices (0-9)
        for i in range(10):
            try:
                # Try to open the camera briefly just to check if it exists
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Get some basic info if possible
                    name = f"Camera {i}"
                    api_name = "Unknown"

                    # Try to get more info
                    try:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        name = f"Camera {i} ({width}x{height})"
                    except:
                        pass

                    cameras.append(CameraInfo(i, name, api_name))

                # Always release the capture
                cap.release()

            except Exception as e:
                logger.debug(f"Error checking camera index {i}: {str(e)}")

        return cameras

    def _detect_cameras_windows(self) -> List[CameraInfo]:
        """Windows-specific camera detection using DirectShow"""
        cameras = []

        try:
            # First, try DirectShow backend which often works better on Windows
            for i in range(10):
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        # Get device name if possible
                        name = f"Camera {i}"
                        api_name = "DirectShow"

                        # Get resolution
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        if width > 0 and height > 0:
                            name = f"Camera {i} ({width}x{height})"

                        cameras.append(CameraInfo(i, name, api_name))
                    cap.release()
                except Exception as e:
                    logger.debug(f"Error checking DirectShow camera {i}: {str(e)}")

            # Then try the Media Foundation backend (newer Windows API)
            if not cameras:
                for i in range(10):
                    try:
                        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
                        if cap.isOpened():
                            name = f"Camera {i}"
                            api_name = "Media Foundation"

                            # Get resolution
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            if width > 0 and height > 0:
                                name = f"Camera {i} ({width}x{height})"

                            cameras.append(CameraInfo(i, name, api_name))
                        cap.release()
                    except Exception as e:
                        logger.debug(
                            f"Error checking Media Foundation camera {i}: {str(e)}"
                        )
        except Exception as e:
            logger.error(f"Error in Windows camera detection: {str(e)}")

        return cameras

    def _detect_cameras_linux(self) -> List[CameraInfo]:
        """Linux-specific camera detection using V4L2"""
        cameras = []

        try:
            # Check /dev/video* devices which is standard on Linux
            import glob
            import re

            # Find all video devices
            video_devices = glob.glob("/dev/video*")

            for device in video_devices:
                try:
                    # Extract device number
                    match = re.search(r"/dev/video(\d+)", device)
                    if match:
                        device_id = int(match.group(1))

                        # Try opening with V4L2
                        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L)
                        if cap.isOpened():
                            name = f"V4L2 Camera {device_id}"
                            api_name = "V4L2"

                            # Get resolution
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            if width > 0 and height > 0:
                                name = f"V4L2 Camera {device_id} ({width}x{height})"

                            cameras.append(CameraInfo(device_id, name, api_name))
                        cap.release()
                except Exception as e:
                    logger.debug(f"Error checking V4L2 device {device}: {str(e)}")

        except Exception as e:
            logger.error(f"Error in Linux camera detection: {str(e)}")

        return cameras

    def _detect_cameras_macos(self) -> List[CameraInfo]:
        """macOS-specific camera detection"""
        # macOS typically uses the default AVFoundation back-end in OpenCV
        cameras = []

        try:
            for i in range(10):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        name = f"Camera {i}"
                        api_name = "AVFoundation"

                        # Get resolution
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        if width > 0 and height > 0:
                            name = f"Camera {i} ({width}x{height})"

                        cameras.append(CameraInfo(i, name, api_name))
                    cap.release()
                except Exception as e:
                    logger.debug(f"Error checking AVFoundation camera {i}: {str(e)}")
        except Exception as e:
            logger.error(f"Error in macOS camera detection: {str(e)}")

        return cameras

    def start_camera(
        self,
        camera_id: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None,
        fps: Optional[int] = None,
        backend: Optional[CameraBackend] = None,
    ) -> bool:
        """Start camera capture with specified parameters"""
        # Stop any existing camera thread
        self.stop_camera()

        # Use provided parameters or defaults
        camera_id = camera_id if camera_id is not None else self.current_camera_id
        resolution = resolution if resolution is not None else self.current_resolution
        fps = fps if fps is not None else self.current_fps
        backend = backend if backend is not None else self.current_backend

        if camera_id is None:
            self.camera_error.emit("No camera available")
            return False

        try:
            # Log the attempt
            logger.info(
                f"Starting camera {camera_id} with {resolution[0]}x{resolution[1]} @ {fps}fps"
            )
            self.camera_status.emit(f"Connecting to camera {camera_id}...")
            self.camera_progress.emit(0)  # Reset progress

            # Create and start camera thread with progress monitoring
            self.camera_thread = CameraThread(camera_id, resolution, fps, backend)

            # Connect signals
            self.camera_thread.frame_ready.connect(self.frame_captured)
            self.camera_thread.error.connect(self.camera_error)
            self.camera_thread.status_message.connect(self.camera_status)
            self.camera_thread.connected.connect(self.camera_connected)
            self.camera_thread.progress_updated.connect(self.camera_progress)

            # Setup priority (lower than UI thread) for better responsiveness
            self.camera_thread.setPriority(QThread.Priority.LowPriority)

            # Start the thread
            self.camera_thread.start()

            # Update current settings
            self.current_camera_id = camera_id
            self.current_resolution = resolution
            self.current_fps = fps
            self.current_backend = backend

            return True
        except Exception as e:
            error_msg = f"Error starting camera: {str(e)}"
            logger.error(error_msg)
            self.camera_error.emit(error_msg)
            self.camera_connected.emit(False)
            self.camera_progress.emit(0)  # Reset progress
            return False

    def stop_camera(self):
        """Stop camera capture"""
        if self.camera_thread and self.camera_thread.isRunning():
            logger.info("Stopping camera")
            self.camera_status.emit("Disconnecting from camera...")
            self.camera_thread.stop()
            self.camera_thread = None
            self.camera_status.emit("Camera disconnected")
            self.camera_connected.emit(False)

    def get_available_cameras(self) -> List[CameraInfo]:
        """Get list of available camera devices"""
        return self.available_cameras

    def refresh_cameras(self):
        """Refresh the list of available cameras"""
        self.camera_status.emit("Scanning for cameras...")
        self._refresh_available_cameras()
        self.camera_status.emit(f"Found {len(self.available_cameras)} cameras")

    def set_resolution(self, width: int, height: int):
        """Set camera resolution"""
        self.current_resolution = (width, height)
        if self.camera_thread and self.camera_thread.isRunning():
            # Restart camera with new resolution
            self.start_camera()

    def set_fps(self, fps: int):
        """Set camera FPS"""
        self.current_fps = fps
        if self.camera_thread and self.camera_thread.isRunning():
            # Restart camera with new FPS
            self.start_camera()

    def set_backend(self, backend):
        """Set camera backend based on platform availability"""
        import platform

        system = platform.system().lower()

        if backend == CameraBackend.ANY:
            self._backend = cv2.CAP_ANY
        elif system == "windows":
            if backend == CameraBackend.DSHOW:
                self._backend = cv2.CAP_DSHOW
            elif backend == CameraBackend.MSMF:
                self._backend = cv2.CAP_MSMF
            else:
                self._backend = cv2.CAP_ANY
        elif system == "linux":
            if backend == CameraBackend.V4L:
                self._backend = cv2.CAP_V4L2
            elif backend == CameraBackend.GSTREAMER:
                self._backend = cv2.CAP_GSTREAMER
            else:
                self._backend = cv2.CAP_ANY
        elif system == "darwin":  # macOS
            if backend == CameraBackend.AVFOUNDATION:
                self._backend = cv2.CAP_AVFOUNDATION
            else:
                self._backend = cv2.CAP_ANY
        else:
            self._backend = cv2.CAP_ANY

        if self.camera_thread and self.camera_thread.isRunning():
            # Restart camera with new backend
            self.start_camera()

    def get_camera_properties(self, camera_id: Optional[int] = None) -> Dict[str, Any]:
        """Get detailed properties of a specific camera
        This will temporarily open the camera to query its properties
        """
        if camera_id is None:
            camera_id = self.current_camera_id

        if camera_id is None:
            return {}

        properties = {}
        cap = None

        try:
            # Try to open the camera with the preferred backend
            if self.current_backend == CameraBackend.DSHOW:
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            elif self.current_backend == CameraBackend.V4L:
                cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L)
            else:
                cap = cv2.VideoCapture(camera_id)

            if not cap.isOpened():
                return {"error": "Failed to open camera"}

            # Collect properties
            properties["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            properties["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            properties["fps"] = cap.get(cv2.CAP_PROP_FPS)
            properties["format"] = cap.get(cv2.CAP_PROP_FORMAT)
            properties["mode"] = cap.get(cv2.CAP_PROP_MODE)
            properties["brightness"] = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            properties["contrast"] = cap.get(cv2.CAP_PROP_CONTRAST)
            properties["saturation"] = cap.get(cv2.CAP_PROP_SATURATION)
            properties["hue"] = cap.get(cv2.CAP_PROP_HUE)
            properties["gain"] = cap.get(cv2.CAP_PROP_GAIN)
            properties["exposure"] = cap.get(cv2.CAP_PROP_EXPOSURE)

            # Try to get supported resolutions
            supported_resolutions = []
            for res in [(640, 480), (800, 600), (1280, 720), (1920, 1080)]:
                if cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0]) and cap.set(
                    cv2.CAP_PROP_FRAME_HEIGHT, res[1]
                ):
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if (actual_width, actual_height) == res:
                        supported_resolutions.append(res)

            properties["supported_resolutions"] = supported_resolutions

        except Exception as e:
            properties["error"] = str(e)
        finally:
            if cap and cap.isOpened():
                cap.release()

        return properties
