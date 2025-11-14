# Face Detection Options for MegaFS

This document describes the face detection implementations available in MegaFS and suggests alternative approaches for future integration.

## Current Implementations

### 1. BlazeFace (with Padding Fix)

**Status**: ✅ Implemented and default

**Description**: MediaPipe's BlazeFace is a lightweight, real-time face detection model optimized for mobile devices. We've enhanced it with padding support to handle large faces that fill most of the image.

**Key Features**:
- Fast inference (~10ms on GPU)
- Lightweight model (~200KB)
- Anchor-based detection
- **Padding fix**: Automatically pads images before detection to handle large faces

**Configuration**:
```yaml
method: "blazeface_padded"
blazeface:
  padding_ratio: 1.5  # Pad image by 1.5x before detection
  threshold: 0.5
  nms_threshold: 0.3
```

**Limitations**:
- Originally designed for faces 20-50% of image size
- Padding fix helps but may still struggle with very large faces (>90% of image)
- Requires model weights and anchor files

**Use Cases**:
- Real-time applications
- When speed is critical
- Standard face sizes (20-80% of image)

### 2. BlazeFace (Original, No Padding)

**Status**: ✅ Implemented

**Description**: Original BlazeFace implementation without padding. Maintained for backward compatibility.

**Configuration**:
```yaml
method: "blazeface"
```

**Use Cases**:
- Backward compatibility
- When faces are known to be small-medium size

### 3. Haar-Cascade

**Status**: ✅ Implemented

**Description**: Classic OpenCV Haar-Cascade face detector. Robust and reliable, especially for large faces.

**Key Features**:
- No anchor limitations
- Works well with faces that fill most of image
- Built into OpenCV (no external dependencies)
- Slower than BlazeFace but more reliable for edge cases

**Configuration**:
```yaml
method: "haar"
haar:
  scale_factor: 1.1
  min_neighbors: 3
  min_size: 50
```

**Limitations**:
- Slower than deep learning methods (~50-100ms)
- Less accurate than modern methods
- May have false positives

**Use Cases**:
- Large faces (>80% of image)
- Fallback when BlazeFace fails
- When accuracy is more important than speed

## Suggested Alternative Approaches

### 1. MTCNN (Multi-task CNN)

**Status**: 🔄 Suggested for future implementation

**Description**: Multi-task Cascaded Convolutional Networks for face detection and alignment.

**Advantages**:
- Very accurate (state-of-the-art for small faces)
- Provides facial landmarks (eyes, nose, mouth)
- Handles various face sizes well
- Good for challenging conditions (lighting, angles)

**Disadvantages**:
- Slower than BlazeFace (~100-200ms)
- Larger model size (~2MB)
- Requires additional dependencies (mtcnn package)

**Integration Notes**:
- Install: `pip install mtcnn`
- Would need wrapper class: `MTCNNDetector(FaceDetector)`
- Returns bounding boxes + landmarks (can use bbox only)

**Recommended For**:
- High-accuracy requirements
- When facial landmarks are needed
- Challenging detection scenarios

### 2. RetinaFace

**Status**: 🔄 Suggested for future implementation

**Description**: State-of-the-art face detection with excellent handling of large faces and various scales.

**Advantages**:
- Excellent accuracy, especially for large faces
- Handles faces that fill >90% of image
- Provides facial landmarks
- Good performance on various face sizes

**Disadvantages**:
- Slower than BlazeFace (~50-100ms on GPU)
- Larger model size (~1.7MB)
- Requires PyTorch/TensorFlow

**Integration Notes**:
- Install: `pip install retinaface`
- Would need wrapper class: `RetinaFaceDetector(FaceDetector)`
- Excellent for the large face use case

**Recommended For**:
- Large face detection (primary use case)
- High accuracy requirements
- When landmarks are needed

### 3. MediaPipe Face Detection

**Status**: 🔄 Suggested for future implementation

**Description**: Google's MediaPipe face detection (different from BlazeFace, more recent version).

**Advantages**:
- Fast and accurate
- Handles various face sizes well
- Provides facial landmarks
- Well-maintained by Google

**Disadvantages**:
- Requires MediaPipe installation
- Slightly slower than BlazeFace
- Different API than current implementation

**Integration Notes**:
- Install: `pip install mediapipe`
- Would need wrapper class: `MediaPipeDetector(FaceDetector)`
- Good balance of speed and accuracy

**Recommended For**:
- General-purpose face detection
- When landmarks are needed
- Balanced speed/accuracy requirements

### 4. YOLOv5-Face

**Status**: 🔄 Suggested for future implementation

**Description**: YOLOv5 adapted for face detection, real-time performance.

**Advantages**:
- Very fast (real-time capable)
- Good accuracy
- Handles various face sizes
- Can detect multiple faces efficiently

**Disadvantages**:
- Requires YOLOv5 installation
- Model size (~14MB)
- May need fine-tuning for specific use cases

**Integration Notes**:
- Install: `pip install ultralytics` (YOLOv5)
- Would need wrapper class: `YOLOv5FaceDetector(FaceDetector)`
- Good for batch processing multiple faces

**Recommended For**:
- Real-time applications
- Multiple face detection
- Batch processing

### 5. SCRFD (Sample and Computation Redistribution for Face Detection)

**Status**: 🔄 Suggested for future implementation

**Description**: Efficient face detection with good accuracy-speed tradeoff.

**Advantages**:
- Efficient and accurate
- Good handling of various face sizes
- Provides facial landmarks
- Balanced performance

**Disadvantages**:
- Less well-known (smaller community)
- May require custom installation
- Documentation may be limited

**Integration Notes**:
- Would need wrapper class: `SCRFDDetector(FaceDetector)`
- Good alternative to RetinaFace with similar performance

**Recommended For**:
- Balanced accuracy-speed requirements
- When landmarks are needed
- Alternative to RetinaFace

## Comparison Table

| Method | Speed (ms) | Accuracy | Large Face Handling | Model Size | Landmarks | Status |
|--------|-----------|----------|---------------------|------------|-----------|--------|
| BlazeFace (padded) | ~10 | Good | Good (with padding) | ~200KB | No | ✅ Implemented |
| BlazeFace (original) | ~10 | Good | Fair | ~200KB | No | ✅ Implemented |
| Haar-Cascade | ~50-100 | Fair | Excellent | Built-in | No | ✅ Implemented |
| MTCNN | ~100-200 | Excellent | Good | ~2MB | Yes | 🔄 Suggested |
| RetinaFace | ~50-100 | Excellent | Excellent | ~1.7MB | Yes | 🔄 Suggested |
| MediaPipe | ~20-30 | Good | Good | Built-in | Yes | 🔄 Suggested |
| YOLOv5-Face | ~15-25 | Good | Good | ~14MB | No | 🔄 Suggested |
| SCRFD | ~30-50 | Excellent | Excellent | ~1.5MB | Yes | 🔄 Suggested |

## Integration Guide for Future Detectors

To add a new face detector:

1. **Create Detector Class** in `utils/face_detectors.py`:
```python
class NewDetector(FaceDetector):
    def __init__(self, **kwargs):
        # Initialize detector
        pass
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        # Detect faces and return bboxes as (x, y, w, h)
        pass
```

2. **Update Factory Function** `get_face_detector()`:
```python
elif method == "new_detector":
    return NewDetector(**kwargs)
```

3. **Add Configuration** to `configs/attack_config.yaml`:
```yaml
mask_generation:
  method: "new_detector"
  new_detector:
    param1: value1
    param2: value2
```

4. **Update Documentation** in this file.

## Recommendations

### For Large Face Detection (Primary Use Case)
1. **RetinaFace** - Best accuracy for large faces
2. **SCRFD** - Good alternative with similar performance
3. **Haar-Cascade** - Current fallback, reliable but slower

### For Speed-Critical Applications
1. **BlazeFace (padded)** - Current default, fastest
2. **YOLOv5-Face** - Good speed with better accuracy
3. **MediaPipe** - Balanced option

### For Maximum Accuracy
1. **RetinaFace** - State-of-the-art
2. **MTCNN** - Excellent for small faces
3. **SCRFD** - Good balance

### For Production Use
1. **BlazeFace (padded)** - Current default, well-tested
2. **Haar-Cascade** - Reliable fallback
3. **MediaPipe** - Well-maintained by Google

## Current Default Configuration

The default configuration uses:
- **Method**: `blazeface_padded` (BlazeFace with padding fix)
- **Strict Detection**: `true` (attack stops if detection fails)
- **Fallback**: `haar` (if primary fails and strict_detection=false)

This provides a good balance of speed and reliability, with the padding fix addressing the large face detection issue.

