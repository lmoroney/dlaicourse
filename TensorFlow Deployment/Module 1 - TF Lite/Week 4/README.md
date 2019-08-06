# Module 4 - TensorFlow Lite  on Raspberry Pi

## Building TensorFlow Lite

### Cross Compile
We recommend cross-compiling the TensorFlow Raspbian package. Cross-compilation is using a different platform to build the package than deploy to. Instead of using the Raspberry Pi's limited RAM and comparatively slow processor, it's easier to build TensorFlow on a more powerful host machine running Linux, macOS, or Windows. You can see detailed instructions [here](https://www.tensorflow.org/install/source_rpi).

### Install from pip
```
pip install tensorflow
```

## Examples

[Image Classification](./image_classification)  
[Object Detection](./object_detection)  
[Transfer Learning](./transfer_learning)  
[Hyperparameter Tuning](./hyperparameter_tuning)  

## Exercise
[Rock, Paper & Scissors](./exercise)
