# YOLO Models for GHS Hazard Label Detection

Place your trained YOLO model files (*.pt) in this directory to use them with the GHS Hazard Label detection application.

## Models

The application will scan this directory for available `.pt` model files. For best performance, use a model specifically trained on GHS hazard label datasets.

### Default Model

By default, the application will download a general YOLOv8n model. However, for optimal hazard label detection, it's recommended to use a specialized model.

### Custom Models

If you have custom-trained models for hazard label detection:

1. Place the `.pt` model file in this directory
2. Start the application and select your model from the dropdown
3. Click "Load Selected Model" to activate it

## Performance Considerations

- Smaller models (nano, small) run faster but may have lower accuracy
- Larger models (medium, large) offer better detection but require more processing power
- For real-time detection, balance model size with your hardware capabilities

## Camera Troubleshooting

If you experience camera issues like:
```
ERROR:0@15.759] global obsensor_uvc_stream_channel.cpp:158 cv::obsensor::getStreamChannelGroup Camera index out of range
```

Try these solutions:
1. Make sure your camera is properly connected
2. Try a different camera ID in the application's dropdown
3. Select a different camera backend in the dropdown
4. Restart the application and other programs that might be using the camera
5. For IP cameras, check that the URL is correct and the camera is accessible
