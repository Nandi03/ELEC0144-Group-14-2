# Import necessary libraries
from keras.applications import AlexNet
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained AlexNet model without the top (fully connected) layers
base_model = AlexNet(weights='imagenet', include_top=False, input_shape=(227, 227, 3))

# Freeze the convolutional layers so that they are not trainable
for layer in base_model.layers:
    layer.trainable = False

# Create a new model and add the pre-trained layers
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(15, activation='softmax'))  # Output layer for 15-class classification

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()



# Data augmentation and loading
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('train',
                                                    target_size=(227, 227),
                                                    batch_size=5,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory('validation',
                                                        target_size=(227, 227),
                                                        batch_size=5,
                                                        class_mode='categorical')

# Train the model
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator))

