# Offline Face Check-In ONNX Models

Place the following 3 ONNX models in this folder (`android/app/src/main/assets/models/`):

1. **`yunet.onnx`**
   - YuNet face detection model (OpenCV Zoo)
   - Input: 320x320 BGR float32 `[0..255]`
   - Output: Face bounding boxes + 5 facial landmarks (eyes, nose, mouth corners)

2. **`minifasnet_v2.onnx`**
   - MiniFASNetV2 anti-spoofing / liveness detection model
   - Input: 80x80 BGR float32 `[0..255]`
   - Output: 2 logits (Softmax index 1 = liveness probability)

3. **`facex_zoo_xs.onnx`**
   - FaceX-Zoo MobileFaceNet / ArcFace feature extractor
   - Input: 112x112 RGB float32 normalized `(px - 127.5) / 128.0`
   - Output: 512-dimensional face embedding vector (L2-normalized)
