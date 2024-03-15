from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import string
import random
import pickle

# Configuración global
modelo = None
tokenizer = None
le = None
responses = None
input_shape = None
unique_words = None

# Cargar el modelo entrenado, tokenizer y otras configuraciones al inicio
def cargar_modelo():
    global modelo, tokenizer, le, responses, input_shape , unique_words

    try:
        # Cargar modelo
        modelo = tf.keras.models.load_model("modelo_entrenado.h5")

        # Cargar tokenizer
        with open('tokenizer.pkl', 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        # Cargar LabelEncoder
        with open('label_encoder.pkl', 'rb') as le_file:
            le = pickle.load(le_file)

        # Cargar respuestas desde datos.json
        with open('datos.json', 'r', encoding='utf-8') as datos_file:
            data1 = json.load(datos_file)
            responses = {}
            for intento in data1['intentos']:
                responses[intento['etiqueta']] = intento['respuestas']
 
        # Cargar palabras únicas / para comparar con la palabra del usuario en caso no lo encuentre , mencione NO TENGO ESA INFORMACIÓN
        with open('unique_words.pkl', 'rb') as unique_words_file:
            unique_words = pickle.load(unique_words_file)        

        # Otras configuraciones (ajustar según lo que necesites)
        input_shape = modelo.input_shape[1]

        print("Modelo y configuraciones cargados desde los archivos.")
    except (OSError, IOError):
        print("No se encontró un modelo existente. Asegúrate de haber entrenado el modelo y guardado los archivos.")

# Inicializar modelo, tokenizer y configuraciones al inicio del servidor
        
cargar_modelo()

# Configurar la aplicación Flask
app = Flask(__name__)
CORS(app)

# Función para verificar si una palabra está en la lista de palabras únicas
def is_word_known(word):
    return word in unique_words

# Rutas y funciones del servidor
@cross_origin
@app.route('/msg', methods=['POST'])
def predict_msg():
    global modelo, tokenizer, le, responses, input_shape

    if modelo is None:
        return "Error: Modelo no cargado correctamente."

    # Obtener el mensaje del usuario
    message = request.form.get('mensaje')
    print("Pregunta enviada por el usuario:", message)

    # Filtrar las palabras que no están en la lista de palabras únicas
    filtered_message = ' '.join(word.lower() for word in message.split() if is_word_known(word))

    # Verificar si el mensaje no contiene palabras únicas
    if not filtered_message:
        respuesta_desconocida = "No tengo información sobre eso. ¿Hay algo más en lo que pueda ayudarte?"
        print(respuesta_desconocida)
        return (respuesta_desconocida)
       # return jsonify({"respuesta": respuesta_desconocida})

    # Procesar el mensaje filtrado
    texts_p = [filtered_message]

    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], maxlen=input_shape)

    output = modelo.predict(prediction_input)
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]

    if response_tag in responses:
        respuesta_seleccionada = random.choice(responses[response_tag])
        print("Respuesta acertada:", respuesta_seleccionada)
        return (respuesta_seleccionada)
    else:
        print(f"No se encontró respuesta para la etiqueta {response_tag}")
        respuesta_seleccionada = "No tengo información sobre eso. ¿Hay algo más en lo que pueda ayudarte?"
        print(respuesta_seleccionada)
        return (respuesta_seleccionada)

    #print(random.choice(responses[response_tag]))
    #return random.choice(responses.get(response_tag))


# Ejecutar la aplicación si este script es el principal
if __name__ == '__main__':
    app.run(port=4000)
