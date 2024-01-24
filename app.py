import streamlit as st
import numpy as np

from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model


def main():
  st.title("Malaria Cell Detection App")

  # File upload
  uploaded_file = st.file_uploader("Choose an image...", type="png")

  if uploaded_file is not None:
    # Preprocess the image
    image = Image.open(uploaded_file).resize((32, 32))

    # Make predictions
    prediction = predict(image)

    # Display the result
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Prediction:", prediction)


def predict(img):
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array /= 255.0

  # Load model
  loaded_model = load_model('model.h5')

  # Make predictions
  predictions = loaded_model.predict(img_array)

  # train_generator.class_indices is a dictionary mapping class names to indices
  class_indices = {'Parasitized': 0, 'Uninfected': 1}

  # Get the class label with the highest probability
  predicted_class = max(class_indices,
                        key=lambda k: predictions[0][class_indices[k]])

  return predicted_class


if __name__ == "__main__":
  main()