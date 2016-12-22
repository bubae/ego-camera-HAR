# Convolutional Recurrent Neural Network for first-person action recognition

By Byeong-uk Bae (KAIST)

This code implemented Python 2.7 and OpenCV 2.4.11 version.

### Requirements: Software

1. Python 2.7 or 3.x version
2. OpenCV 2.4 later version
3. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```
  
4. Keras (Tensorflow based)

### Requirements: Hardware

1. Overall GPU RAM >= 8GB (GTX 1080, TITAN X etc..)
2. For training ObjectNet GPU >= 3GB
3. For training MotionNet GPU >= 8GB
4. For training LSTM model GPU >= 8GB


- This can be changed by resizing batches.

### Requirements: Pre-trained Model

1. ObjectNet Pre-trained Model:
2. MotionNet Pre-trained Model:
3. LSTM Pre-trained Model:

