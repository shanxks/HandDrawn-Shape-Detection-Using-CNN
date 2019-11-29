from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

model = Sequential()
model.add(Conv2D(16, 3, 3, input_shape=(28, 28, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(56, activation='relu', kernel_initializer='uniform'))
model.add(Dense(3, activation='softmax', kernel_initializer='uniform'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_data = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,vertical_flip=True)
test_data = ImageDataGenerator(rescale=1. / 255)
eval_data = ImageDataGenerator(rescale=1. / 255)

training_set = train_data.flow_from_directory('S:\\Project\\shapes\\train', target_size=(28, 28),color_mode="rgb", batch_size=1, class_mode='categorical')
test_set = test_data.flow_from_directory('S:\\Project\\shapes\\validation', target_size=(28, 28),color_mode="rgb",batch_size=1, class_mode='categorical')
eval_set = eval_data.flow_from_directory('S:\\Project\\shapes\\test', target_size=(28, 28),color_mode="rgb",batch_size=1, class_mode='categorical')

steps = len(training_set.filenames) 
val_steps = len(test_set.filenames) 

history=model.fit_generator(training_set, steps_per_epoch=steps, epochs=25, validation_data=test_set,
                                      validation_steps=val_steps)
model.save("S:\\Project\\trained model\\model.h5")
print('\nModel Saved')
print('\nModel evaluation')
lo=model.evaluate(eval_set)

print("\nTest Loss", lo[0])
print("Test Accuracy", lo[1])

print('\nGRAPHS FOR ACCURACY AND LOSS DURING TRAINING')

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()

