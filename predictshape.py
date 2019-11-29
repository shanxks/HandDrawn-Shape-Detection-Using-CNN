import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as m
model = tf.keras.models.load_model('S:\\Project\\trained model\\model1.h5')
label = {0: "Circle", 1: "Square", 2: "Triangle"}
r="S:\\trainge.png"
im = Image.open(r)#.convert('L')
im = im.resize((28,28), PIL.Image.ANTIALIAS)
img= np.asarray( im, dtype="float32" ).reshape([1,28, 28, 3])
classes = model.predict_classes(img)[0]
print(classes)
category = label[classes]
print("\nGiven image has a shape close to "+category)
plt.imshow(m.imread(r))
