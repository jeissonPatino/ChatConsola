import streamlit as st
import json
import sys
from aut.authentication import registrar_usuario, iniciar_sesion
#from chatbot import mostrar_chat

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
                #mostrar_chat(usuario) 
        except FileNotFoundError:
            st.error("Error al obtener la información del usuario")
        if st.sidebar.button("Cerrar sesión"):
            st.session_state.autenticado = False
            st.rerun()

if __name__ == '__main__':
    main()