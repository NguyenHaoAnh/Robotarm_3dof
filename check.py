from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
model = load_model('Robotarm3b.h5')
plt.plot(model.model['accuracy'])
plt.plot(model.model['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epochs')
plt.legend(['train','Validation'])
plt.show()