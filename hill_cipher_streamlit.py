import numpy as np
from sympy import Matrix
import streamlit as st

# Utility Functions
def text_to_numeric(text):
    """Converts text to numeric values (A=0, B=1, ..., Z=25)."""
    numeric = [ord(char.upper()) - ord('A') for char in text]
    return numeric

def numeric_to_text(numbers):
    """Converts numeric values back to text (A=0, B=1, ..., Z=25)."""
    text = ''.join([chr((num % 26) + ord('A')) for num in numbers])
    return text

def matrix_mod_inverse(matrix, mod):
    """Finds the modular inverse of a matrix and displays determinant and GCD."""
    det = int(np.round(np.linalg.det(matrix)))  # Calculate the determinant
    gcd_value = np.gcd(det, mod)  # Calculate GCD of determinant and mod
    st.write(f"**Determinant:** {det} **(mod {mod} = {det % mod})**")
    st.write(f"**GCD(Determinant, {mod}):** {gcd_value}")

    if gcd_value != 1:
        st.write("**This matrix is not invertible under modulo 26.** Please regenerate the matrix or use another pair.")
        return None

    try:
        inv_matrix = Matrix(matrix).inv_mod(mod)
        st.write(f"**Modular Inverse of Key Matrix (mod {mod}):**\n{inv_matrix}")
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
def chosen_ciphertext_attack(plain_text, cipher_text, size):
    """Performs Chosen Ciphertext Attack to recover the key matrix."""
    mod = 26

    # Step 1: Prepare the plaintext and ciphertext matrices
    st.write("### Step 1: Preparing the Plaintext and Ciphertext Matrices (Column-wise)")
    plain_numeric = text_to_numeric(plain_text)
    cipher_numeric = text_to_numeric(cipher_text)

    st.write(f"**Plaintext Numeric Values:** {plain_numeric}")
    st.write(f"**Ciphertext Numeric Values:** {cipher_numeric}")

    # Step 2: Reshape into column-wise matrices
    plain_matrix = np.array(plain_numeric).reshape(size, size, order='F')  # Use 'F' for column-major order
    cipher_matrix = np.array(cipher_numeric).reshape(size, size, order='F')  # Use 'F' for column-major order

    display_matrix(plain_matrix, "Plaintext Matrix (Column-wise)")
    display_matrix(cipher_matrix, "Ciphertext Matrix (Column-wise)")

    # Step 3: Find the inverse of the Plaintext Matrix
    st.write("### Step 3: Calculating the Inverse of the Plaintext Matrix")
    inv_plain_matrix = matrix_mod_inverse(plain_matrix, mod)
    if inv_plain_matrix is None:
        st.error("The plaintext matrix is not invertible under modulo 26. Click 'Generate New Pair' to try another matrix.")
        return None

    display_matrix(inv_plain_matrix, "Inverse of Plaintext Matrix (mod 26)")

    # Step 4: Calculate the Key Matrix using the equation: Key = Cipher * Inverse(Plain)
    st.write("### Step 4: Calculating the Key Matrix using `Key = Cipher * Inverse(Plain)`")
   
