import streamlit as st
import hashlib
import json

def registrar_usuario():
    """
    Maneja el registro de nuevos usuarios.
    """
    st.subheader("Registro de usuario")
    nombre = st.text_input("Nombre")
    apellidos = st.text_input("Apellidos")
    edad = st.number_input("Edad", min_value=0, step=1)
    nacionalidad = st.text_input("Nacionalidad")
    correo = st.text_input("Correo electrónico")
    contraseña = st.text_input("Contraseña", type="password")
    confirmar_contraseña = st.text_input("Confirmar contraseña", type="password")

    if st.button("Registrarse"):
        if contraseña != confirmar_contraseña:
            st.error("Las contraseñas no coinciden")
        else:
            contraseña_encriptada = hashlib.sha256(contraseña.encode()).hexdigest()
            usuario = {
                'nombre': nombre,
                'apellidos': apellidos,
                'edad': edad,
                'nacionalidad': nacionalidad,
                'correo': correo,
                'contraseña': contraseña_encriptada
            }
            try:
                 
                
                with open(f'../DATA/User/{correo}.json', 'w') as f:
                    json.dump(usuario, f, indent=4)
                st.success("Usuario registrado exitosamente")
            except Exception as e:
                st.error(f"Error al registrar el usuario: {e}")


def iniciar_sesion():
    st.subheader("Iniciar sesión")
    correo = st.text_input("Correo electrónico")
    contraseña = st.text_input("Contraseña", type="password")
    if st.button("Iniciar sesión"):
        try:
            with open(f'../DATA/User/{correo}.json', 'r') as f:
                usuario = json.load(f)
            contraseña_encriptada = hashlib.sha256(contraseña.encode()).hexdigest()
            if usuario['contraseña'] == contraseña_encriptada:
                st.success("Inicio de sesión exitoso")
                st.session_state.ultimo_correo = correo
                return True
            else:
                st.error("Contraseña incorrecta")
                return False
        except FileNotFoundError:
            st.error("Usuario no encontrado")
            return False
        except Exception as e:
            st.error(f"Error al iniciar sesión: {e}")
            return False