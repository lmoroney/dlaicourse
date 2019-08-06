# Deploying a hyperparameter tuned model on Raspberry Pi

We'll be performing Hyperparameter Tuning on an image classification dataset known as 'Horses or Humans'. You can run the trained model on a Raspberry Pi. To get started, go to [this](https://colab.research.google.com/drive/1UOKj5LUMM6HR0RR2A4uOBvo1k2iQEow8) Python notebook. Generate the required TFLite assets from it and copy over to the Raspberry Pi device. 

Next, to run the code on Raspberry Pi, paste the following command

```
python3 classify.py --filename input.jpg --model_path converted_model.tflite
```
