import numpy as np
from sympy import Matrix
import streamlit as st

# Utility Functions
def text_to_numeric(text):
    """Converts text to numeric values (A=0, B=1, ..., Z=25)."""
    return [ord(char.upper()) - ord('A') for char in text]

def numeric_to_text(numbers):
    """Converts numeric values back to text (A=0, B=1, ..., Z=25)."""
    return ''.join([chr((num % 26) + ord('A')) for num in numbers])

def matrix_mod_inverse(matrix, mod):
    """Finds the modular inverse of a matrix and displays determinant and GCD."""
    det = int(np.round(np.linalg.det(matrix)))  # Calculate the determinant
    gcd_value = np.gcd(det, mod)  # Calculate GCD of determinant and mod

    # Convert determinant to positive under modulo 26
    det = det % mod

    st.write(f"**Determinant:** {det} **(mod {mod})**")
    st.write(f"**GCD(Determinant, {mod}):** {gcd_value}")

    if gcd_value != 1:
        st.error("This matrix is not invertible under modulo 26 because GCD(det, 26) is not 1.")
        return None

    try:
        inv_matrix = Matrix(matrix).inv_mod(mod)
        st.write(f"**Modular Inverse of the Matrix (mod {mod}):**\n{inv_matrix}")
        return np.array(inv_matrix).astype(int)
    except:
        st.write("Matrix is not invertible under modulo", mod)
        return None

def display_matrix(matrix, title="Matrix"):
    """Displays a matrix in Streamlit UI."""
    st.write(f"**{title}**")
    st.write(matrix)

def generate_invertible_matrix(size, mod=26):
    """Generates a random invertible matrix under modulo 26."""
    while True:
        # Generate a random matrix of the given size
        matrix = np.random.randint(0, mod, (size, size))
        det = int(np.round(np.linalg.det(matrix)))  # Calculate determinant
        gcd_value = np.gcd(det, mod)  # Calculate GCD

        if gcd_value == 1:  # Check if the matrix is invertible
            return matrix

# Function for Chosen Ciphertext Attack
def chosen_ciphertext_attack(plain_text, cipher_text, size, auto_generate=False):
    """Performs Chosen Ciphertext Attack to recover the key matrix."""
    mod = 26

    if auto_generate:
        st.write("### Automatically Generating an Invertible Plaintext Matrix...")
        # Generate an invertible plaintext matrix and its corresponding ciphertext matrix
        plain_matrix = generate_invertible_matrix(size)
        st.write("**Generated Invertible Plaintext Matrix:**")
        display_matrix(plain_matrix, "Generated Plaintext Matrix (Invertible)")

        # Generate a random ciphertext matrix of the same size
        cipher_matrix = generate_inver
