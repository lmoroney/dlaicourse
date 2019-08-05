# Object Detection in Raspberry Pi
This is the code repository of Object Detection module for running on Edge device of Raspberry Pi. To get started, you will have to download the pretrained model along with its label file. We recommend making use of the quantized SSD MobileNet V1 model trained on COCO capable of classifying around 80 different common object categories. To download and extract it, run the below command

# Prerequisites
## Python
* PiCamera  
* matplotlib (with tkinter as backend)
* TensorFlow

```
wget http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```

Next, to run the code on Raspberry Pi, paste the following command

```
python3 main.py --model_path detect.tflite --label_path labelmap.txt
```

We encourage you to look through all the available options in the code.

For more information, go [here](https://www.tensorflow.org/lite/models/object_detection/overview).
