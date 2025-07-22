# Driver Drowsiness Detection System
A real-time AI model system that detects driver drowsiness by analyzing eye states through a webcam and triggers an alert to prevent accidents.

## What do I need to start this program?
You’ll need to install the following libraries: tensorflow, opencv, numpy, and playsound.

To install these libraries open the comand prompt and enter these commands:
```
pip install tensorflow
```
```
pip install opencv-python
```
```
pip install numpy
```
```
pip install playsound
```

## How do I set up the program?
Run the train_eye_model.py file first as that will create the model and train it using the eye image dataset. After that run the detection.py file for the AI model to start detecting.

## Key things to note
Make sure all hair is brushed or tied back as it may interfere with the detection.
The eye image dataset is missing a lot of images as github did not allow a lot of images to be uploaded at once so I have provided a link below which has all eye images including the ones in this dataset. Make sure to download the file which contains 24x24 sized images.
https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html
