import matplotlib.pyplot as plt
import numpy as np

###################################################################################
###################################################################################

def plot_image(image, prediction_array, true_label, class_names):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(image, cmap=plt.cm.binary)

  predicted_label = np.argmax(prediction_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(prediction_array),
                                class_names[true_label]),
                                color=color)

###################################################################################
###################################################################################

def plot_value_array(prediction_array, true_label):
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])

  thisplot = plt.bar(range(10), prediction_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(prediction_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

###################################################################################
###################################################################################

def plot_predictions(images, labels, predictions_array, class_names, num_rows = 5, num_cols = 3, width_per_subplot = 2):
  # Plot the first X test images, their predicted labels, and the true labels.
  # Color correct predictions in blue and incorrect predictions in red.
  num_images = num_rows*num_cols

  plt.figure(figsize=(width_per_subplot*2*num_cols, width_per_subplot*num_rows))

  for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(images[i], predictions_array[i], labels[i], class_names)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(predictions_array[i], labels[i])
  plt.tight_layout()
  plt.show()

  ###################################################################################
  ###################################################################################

def plot_training_trend(epochs, history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()