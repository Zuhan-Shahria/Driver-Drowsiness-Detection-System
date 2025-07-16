#Be aware newer python versions may not support the modules
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Build a simple CNN
'''
The CNN below has 7 layers, I have explained what each layer does to the image

Conv2D is a convolution layer
The first layer looks at your eye image and scans it 32 times, each time looking for a different pattern.
It creates 32 new images (feature maps), each one showing where and how strongly a certain pattern appears.
These are passed to the next layer for further processing.
32 means you have 32 different filters — each looking for different patterns
(3,3) is the filter size (each filter is 3 pixels by 3 pixels)
activation='relu' introduces non-linearity so the network can learn complex shapes
input_shape=(24,24,1) means:
    images are 24x24 pixels
    only 1 channel/feature map (grayscale, so 1 instead of 3 for RGB)
The output is 32 feature maps and the whole image is 22x22x32 each feature map has a size of 22x22 because the 3x3 filter removes a pixel from each side of the image
    

MaxPooling2D compresses the image by creating a 2x2 pixel window on the image and it picks the pixel with the largest value (brightness value ranges from 0-255)
The other unpicked pixels are removed
It's used so that the neural network runs faster
This layer goes through each of the 32 feature maps and shrinks them from 22x22 to 11x11 by keeping only the strongest signal in each 2x2 patch.
So you're left with smaller, stronger feature maps.


Conv2D(64, (3,3), activation='relu') is the third layer
Its the same as the first layer but this time its generating 
You have 32 feature maps coming from the second layer (after MaxPooling2D).
These are stacked together as a single 3D volume of shape 11x11x32.
Each filter has a size of 3x3x32 — meaning it looks across all 32 feature maps at once.
Just like before the size of each new filter map is reduced to 9x9
The final output of the layer is 9x9x64 so 64 9x9 complex feature maps


For the fourth layer its MaxPooling2D again and its the same as before the size of each feature map is halved
So the output is 64 4x4 compressed feature maps


The fifth layer is Flatten() and all it does is it converts the 3D volume we have into a 1D vector which is neccesary for the next layer


The next layer which is the sixth is called a Dense layer and its where the decision making begins
From the previous layer Flatten() the output is now a 1D array with 1024 values
The dense layer creates 128 neurons where each is connected to all the 1024 inputs and does some processing
The dense layer outputs a new vector with 128 values (one from each neuron)
Each value represents some kind of "signal" that helps the model decide whether the eye is open or closed


The final layer of the CNN is a dense layer with 1 neuron and a sigmoid activation function.
Its purpose is to make the final binary decision: whether the eye in the image is open or closed
It takes the 128 values from the previous dense layer and combines them using learned weights to produce a single output between 0 and 1
This output represents the probability that the input image belongs to the "open eye" class (with values closer to 0 indicating "closed" and closer to 1 indicating "open")
The sigmoid function ensures the output stays within this interpretable range, making it ideal for binary classification tasks
'''
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(24,24,1)), 
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # binary classification
])


#This line sets up your model to learn using the Adam optimizer, measure its errors with binary cross-entropy, and report accuracy while training.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# load images using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    'Eye_Images_Dataset',
    target_size=(24,24),
    color_mode='grayscale',
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    'Eye_Images_Dataset',
    target_size=(24,24),
    color_mode='grayscale',
    class_mode='binary',
    subset='validation'
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

model.save('eye_state_model.h5')