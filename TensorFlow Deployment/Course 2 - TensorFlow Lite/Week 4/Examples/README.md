# Module 4 - TensorFlow Lite  on Raspberry Pi

## Building TensorFlow Lite

### Cross Compile
We recommend cross-compiling the TensorFlow Raspbian package. Cross-compilation is using a different platform to build the package than deploy to. Instead of using the Raspberry Pi's limited RAM and comparatively slow processor, it's easier to build TensorFlow on a more powerful host machine running Linux, macOS, or Windows. You can see detailed instructions [here](https://www.tensorflow.org/install/source_rpi).

## Install just the TensorFlow Lite interpreter
To quickly start executing TensorFlow Lite models with Python, you can install just the TensorFlow Lite interpreter, instead of all TensorFlow packages.

This interpreter-only package is a fraction the size of the full TensorFlow package and includes the bare minimum code required to run inferences with TensorFlow Lite—it includes only the `tf.lite.Interpreter` Python class. This small package is ideal when all you want to do is execute .tflite models and avoid wasting disk space with the large TensorFlow library.

### Install from pip
To install just the interpreter, download the appropriate Python wheel for your system from the following [link](https://www.tensorflow.org/lite/guide/python), and then install it with the `pip install` command.

For example, if you're setting up a Raspberry Pi Model B (using Raspbian Stretch, which has Python 3.5), install the Python wheel as follows (after you click to download the `.whl` file in the provided link):

```
pip install tflite_runtime-1.14.0-cp35-cp35m-linux_armv7l.whl
```

### Running inference

So instead of importing Interpreter from the tensorflow module, you need to import it from tflite_runtime.
```
from tflite_runtime.interpreter import Interpreter
```

## Additional notes
In case you have built TensorFlow from source, you need to import the Interpreter as follows:
```
from tensorflow.lite.python.interpreter import Interpreter 
```

## Examples

[Image Classification](./image_classification)  
[Object Detection](./object_detection)  
[Transfer Learning](./transfer_learning)  
[Hyperparameter Tuning](./hyperparameter_tuning)  

