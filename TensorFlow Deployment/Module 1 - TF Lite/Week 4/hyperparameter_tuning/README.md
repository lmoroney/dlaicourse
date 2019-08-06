# Deploying a hyperparameter tuned model on Raspberry Pi

We'll be performing Hyperparameter Tuning on an image classification dataset known as 'Horses or Humans'. You can run the trained model on a Raspberry Pi. To get started, go to [this](https://colab.research.google.com/drive/1WO2pcqhuGAclTIzmJUXIgQhnr1JTmlct) Python notebook. Generate the required TFLite assets from it and copy over to the Raspberry Pi device. 

## Prerequisites
To install the Python dependencies, run:
```
pip install -r requirements.txt
```

Next, to run the code on Raspberry Pi, use `classify.py` as follows:

```
python3 classify.py --filename input.jpg --model_path converted_model.tflite
```
