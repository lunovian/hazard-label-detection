# YOLO Models Directory

Place your trained YOLO model files (*.pt) in this directory.

The application will look for a model named `yolov8n.pt` by default.

You can download pre-trained YOLOv8 models from Ultralytics or use your own custom-trained model for GHS hazard label detection.

## Troubleshooting Camera Issues

If you see errors like:
```
ERROR:0@15.759] global obsensor_uvc_stream_channel.cpp:158 cv::obsensor::getStreamChannelGroup Camera index out of range
```

Try the following:
1. Make sure your camera is properly connected
2. Try a different camera ID in the application's dropdown
3. Restart the application
4. Check if other applications are using the camera
