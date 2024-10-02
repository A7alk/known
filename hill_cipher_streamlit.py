
import numpy as np
from sympy import Matrix
import streamlit as st

# Hill Cipher Functions with Streamlit Output
def mod_inverse(matrix, mod):
    try:
        inv_matrix = Matrix(matrix).inv_mod(mod)
        st.write(f"**Modular Inverse of Key Matrix (mod {mod}):**\n{inv_matrix}")
        return np.array(inv_matrix).astype(int)
    except:
        st.write("Matrix is not invertible under modulo", mod)
        return None

def matrix_mod_mult(matrix, vector, mod):
    st.write(f"Matrix:\n{matrix}")
    st.write(f"Vector:\n{vector}")
    result = np.dot(matrix, vector)
    st.write(f"Matrix Multiplication Result (before mod {mod}):\n{result}")
    result_mod = result % mod
    st.write(f"Result after Modulo {mod} Operation:\n{result_mod}\n")
    return result_mod

def text_to_numeric(text):
    numeric = [ord(char.upper()) - ord('A') for char in text]
    st.write(f"Text '{text}' to Numeric: {numeric}")
    return numeric

def numeric_to_text(numbers):
    text = ''.join([chr((num % 26) + ord('A')) for num in numbers])
    st.write(f"Numeric {numbers} to Text: '{text}'")
    return text

def pad_text(text, size):
    st.write(f"Original Text: '{text}'")
    while len(text) % size != 0:
        text += 'X'
    st.write(f"Padded Text: '{text}' (to match matrix size {size})\n")
    return text

def hill_encrypt(plain_text, key_matrix):
    mod = 26
    plain_text = pad_text(plain_text, len(key_matrix))
    numeric_plain = text_to_numeric(plain_text)
    cipher_text = ''

    # Break text into chunks and encrypt using the key matrix
    for i in range(0, len(numeric_plain), len(key_matrix)):
        chunk = numeric_plain[i:i + len(key_matrix)]
        st.write(f"\nEncrypting Chunk: {chunk}")
        encrypted_chunk = matrix_mod_mult(key_matrix, chunk, mod)
        cipher_text += numeric_to_text(encrypted_chunk)

    return cipher_text

def hill_decrypt(cipher_text, key_matrix):
    mod = 26
    st.write("\n--- Decryption Process ---")
    st.write(f"Cipher Text: {cipher_text}\n")
    inv_key_matrix = mod_inverse(key_matrix, mod)
    if inv_key_matrix is None:
        return None

    numeric_cipher = text_to_numeric(cipher_text)
    plain_text = ''

    # Break cipher text into chunks and decrypt using the inverse matrix
    for i in range(0, len(numeric_cipher), len(key_matrix)):
        chunk = numeric_cipher[i:i + len(key_matrix)]
        st.write(f"\nDecrypting Chunk: {chunk}")
        decrypted_chunk = matrix_mod_mult(inv_key_matrix, chunk, mod)
        plain_text += numeric_to_text(decrypted_chunk)

    return plain_text

# Streamlit Web Interface
st.title("Hill Cipher Web App with Intermediate Steps")
st.write("A simple web app to demonstrate Hill Cipher encryption, decryption, and detailed steps.")

# Inputs
message = st.text_input("Enter the message:")
matrix_size = st.number_input("Matrix Size (2 or 3):", min_value=2, max_value=3, value=2)
key_values = st.text_input("Enter Key Matrix Values (space-separated):")

# Run the operations
if st.button("Encrypt"):
    try:
        key_matrix_np = np.array(list(map(int, key_values.split()))).reshape(matrix_size, matrix_size)
        st.write(f"**Key Matrix:**\n{key_matrix_np}")
        encrypted_message = hill_encrypt(message, key_matrix_np)
        st.success(f"Encrypted Message: {encrypted_message}")
    except:
        st.error("Encryption failed. Check your inputs and key matrix.")

if st.button("Decrypt"):
    try:
        key_matrix_np = np.array(list(map(int, key_values.split()))).reshape(matrix_size, matrix_size)
        st.write(f"**Key Matrix:**\n{key_matrix_np}")
        decrypted_message = hill_decrypt(message, key_matrix_np)
        st.success(f"Decrypted Message: {decrypted_message}")
    except:
        st.error("Decryption failed. Check your inputs and key matrix.")
