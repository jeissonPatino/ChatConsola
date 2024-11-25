import streamlit as st
import json
import sys
from aut.authentication import registrar_usuario, iniciar_sesion
import os 
# --- Código del chatbot ---
import random
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Importamos los archivos generados en el código anterior
intents = json.loads(open('../DATA/intents.json',encoding='utf-8').read())
words = pickle.load(open('../DATA/entrenamiento/words.pkl', 'rb'))
classes = pickle.load(open('../DATA/entrenamiento/classes.pkl', 'rb'))
model = load_model('../DATA/entrenamiento/chatbot_model.h5')

# Pasamos las palabras de oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

# Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Obtenemos una respuesta aleatoria
def get_response(ints, intents_json):
    try:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i["tag"]==tag:
                result = random.choice(i['responses'])
                break
    except IndexError:
        result = "No entiendo lo que dices"
    return result

def respuesta(message, usuario):
    ints = predict_class(message)
    res = get_response(ints, intents)
    tag = ints[0]['intent']  # Obtener el 'tag' del intent reconocido

    if tag == "nombre_usuario":
        res = f"Te llamas {usuario['nombre']} {usuario['apellidos']}."
    elif tag == "correo_usuario":
        res = f"Tu correo es {usuario['correo']}."
    elif tag == "nacionalidad_usuario":
        res = f"Tu eres de {usuario['nacionalidad']}."
    elif tag == "edad_usuario":
        res = f"Tu tienes {usuario['edad']} años."

    return res

# --- Fin del código del chatbot ---

sys.path.append('App') 

def main():
    st.title("Sistema de autenticación con Streamlit")
    if "autenticado" not in st.session_state:
        st.session_state.autenticado = False

    if not st.session_state.autenticado:
        opcion = st.sidebar.radio("Selecciona una opción", ["Registrarse", "Iniciar sesión"])
        if opcion == "Registrarse":
            registrar_usuario()
        elif opcion == "Iniciar sesión":
            if iniciar_sesion():
                st.session_state.autenticado = True
                st.session_state.correo = st.session_state.ultimo_correo
                st.rerun()
    else:   
        try:
            with open(f'../DATA/User/{st.session_state.correo}.json', 'r') as f:
                usuario = json.load(f)
                
                nombre_archivo = f"../DATA/historial_chat/{usuario['correo']}.json"
                if os.path.exists(nombre_archivo):
                    with open(nombre_archivo, "r") as f:
                        st.session_state.messages = json.load(f)
                else:
                    st.session_state.messages = []
                # --- Mostrar el chat ---
                # Inicializar la variable de sesión para el historial del chat
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Función para manejar el envío de mensajes
                def send_message(usuario):
                    user_message = st.session_state.user_input
                    st.session_state.messages.append({"role": "user", "content": user_message})
                    bot_message = respuesta(user_message, usuario)
                    st.session_state.messages.append({"role": "assistant", "content": bot_message})
                    st.session_state.user_input = ""  # Limpiar el input

                    # --- Guardar el historial del chat ---
                    nombre_archivo = f"../DATA/historial_chat/{usuario['correo']}.json"
                    os.makedirs(os.path.dirname(nombre_archivo), exist_ok=True)  # Crear carpeta si no existe
                    with open(nombre_archivo, "w") as f:
                        json.dump(st.session_state.messages, f)
                    # --- Fin de guardar el historial ---

                # ... (Interfaz de usuario de Streamlit) ...

                # Interfaz de usuario de Streamlit
                st.title("Chat con Consola")

                # Mostrar el historial de mensajes
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Input para el mensaje del usuario
                st.text_input("Escribe tu mensaje:", key="user_input")
                st.button("Enviar", on_click=lambda: send_message(usuario))
                # --- Fin del chat ---
                
        except FileNotFoundError:
            st.error("Error al obtener la información del usuario")
        if st.sidebar.button("Cerrar sesión"):
            st.session_state.autenticado = False
            st.rerun()

if __name__ == '__main__':
    main()