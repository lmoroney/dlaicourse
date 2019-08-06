# Image Classification in Raspberry Pi

This is the code repository of Image Classification module for running on Edge device of Raspberry Pi. To get started, you will have to download the pretrained model along with its label file. We recommend making use of the quantized MobileNet V1 model trained on the ImageNet dataset comprising of 1001 labels. To download and extract it, run the following:

```
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip
unzip mobilenet_v1_1.0_224_quant_and_labels
```

## Prerequisites
To install the Python dependencies, run:
```
pip install -r requirements.txt
```

Next, to run the code on Raspberry Pi, use `classify.py` as follows:

```
python3 classify.py --filename dog.jpg --model_path mobilenet_v1_1.0_224_quant.tflite --label_path labels_mobilenet_quant_v1_224.txt
```

We encourage you to look through all the available options in the code. 

For more information, go [here](https://www.tensorflow.org/lite/models/image_classification/overview).
