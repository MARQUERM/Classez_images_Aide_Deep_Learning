import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions
import io
import joblib

# Charge le modèle EfficientNetV2B2 pré-entraîné
model_path = "model/model_efficientnetv2b0_50races_best_weights.h5" 
model = load_model(model_path)

# Chargement du dictionnaire des noms de races associés aux labels
label_dict = joblib.load("label_dict.joblib")

# Inversion du dictionnaire pour obtenir les noms de race comme clés
inverse_label_dict = {v: k for k, v in label_dict.items()}

st.title('LE Refuge: classification des chiens')

# Téléchargement de l'image
uploaded_file = st.file_uploader("Choisir une image...", type="jpg")

if uploaded_file is not None:
    # Affiche l'image téléchargée
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption='Uploaded Image.', use_column_width=True)

    # Converti l'image en un tableau numpy
    img_array = np.array(image_pil)

    # Redimensionne l'image à la taille attendue par EfficientNetV2B2 (224, 224)
    img_resized = image.img_to_array(image_pil.resize((224, 224)))
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)

    # Prédiction
    predictions = model.predict(img_array)

    # Obtient les trois meilleures prédictions avec leurs scores
    top_predictions = np.argsort(predictions[0])[::-1][:3]

    # Affiche les trois meilleures prédictions avec leurs scores
    st.subheader('Top 3 Prédictions:')
    for i, class_index in enumerate(top_predictions):
        label = inverse_label_dict.get(class_index, "Race inconnue")  
        score = predictions[0, class_index]
        st.write(f"{i + 1}: {label} avec un score de {score:.2f}")