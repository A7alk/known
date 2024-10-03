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
        return None, gcd_value

    try:
        inv_matrix = Matrix(matrix).inv_mod(mod)
        st.write(f"**Modular Inverse of the Matrix (mod {mod}):**\n{inv_matrix}")
        return np.array(inv_matrix).astype(int), gcd_value
    except:
        st.write(f"Matrix is not invertible under modulo {mod}")
        return None, gcd_value

def display_matrix(matrix, title="Matrix"):
    """Displays a matrix in Streamlit UI."""
    st.write(f"**{title}**")
    st.write(matrix)

def make_matrix_invertible(matrix, mod=26):
    """Attempts to make the matrix invertible by modifying its elements slightly."""
    det = int(np.round(np.linalg.det(matrix))) % mod
    gcd_value = np.gcd(det, mod)
    attempt_count = 0

    while gcd_value != 1:
        attempt_count += 1
        # Randomly modify a single element in the matrix to change its determinant
        i, j = np.random.randint(0, matrix.shape[0]), np.random.randint(0, matrix.shape[1])
        matrix[i][j] = (matrix[i][j] + 1) % mod  # Increment the value at (i, j)

        # Recalculate determinant and GCD
        det = int(np.round(np.linalg.det(matrix))) % mod
        gcd_value = np.gcd(det, mod)
        st.write(f"Attempt {attempt_count}: Modified matrix element at position ({i}, {j}), Determinant = {det}, GCD(Det, 26) = {gcd_value}")

        if attempt_count > 100:  # Prevent infinite loops
            st.error("Unable to find an invertible matrix after 100 attempts. Please try a different matrix.")
            break

    return matrix

# Function for Chosen Ciphertext Attack
def chosen_ciphertext_attack(plain_text, cipher_text, size):
    """Performs Chosen Ciphertext Attack to recover the key matrix."""
    mod = 26

    # Step 1: Convert the input plaintext and ciphertext to numeric values
    st.write("### Step 1: Preparing the Plaintext and Ciphertext Matrices (Column-wise)")
    plain_numeric = text_to_numeric(plain_text)
    cipher_numeric = text_to_numeric(cipher_text)

    st.write(f"**Plaintext Numeric Values:** {plain_numeric}")
    st.write(f"**Ciphertext Numeric Values:** {cipher_numeric}")

    # Step 2: Reshape into column-wise matrices
    plain_matrix = np.array(plain_numeric).reshape(size, size, order='F')
    cipher_matrix = np.array(cipher_numeric).reshape(size, size, order='F')

    display_matrix(plain_matrix, "Plaintext Matrix (Column-wise)")
    display_matrix(cipher_matrix, "Ciphertext Matrix (Column-wise)")

    # Step 3: Calculate the Inverse of the Plaintext Matrix
    st.write("### Step 3: Calculating the Inverse of the Plaintext Matrix")
    inv_plain_matrix, gcd_value = matrix_mod_inverse(plain_matrix, mod)
    if inv_plain_matrix is None:
        st.warning(f"The plaintext matrix is not invertible because GCD(det, 26) = {gcd_value}. Attempting to modify it...")
        plain_matrix = make_matrix_invertible(plain_matrix, mod)
        inv_plain_matrix, _ = matrix_mod_inverse(plain_matrix, mod)

    if inv_plain_matrix is None:
        st.error("Unable to make the matrix invertible after modification attempts.")
        return None

    display_matrix(inv_plain_matrix, "Inverse of Plaintext Matrix (mod 26)")

    # Step 4: Calculate the Key Matrix using the equation: Key = Cipher * Inverse(Plain)
    st.write("### Step 4: Calculating the Key Matrix using `Key = Cipher * Inverse(Plain)`")
    key_matrix = np.dot(cipher_matrix, inv_plain_matrix) % mod
    display_matrix(key_matrix, "Recovered Key Matrix (mod 26)")

    return key_matrix

# Streamlit UI for Hill Cipher - Chosen Ciphertext Attack
st.title("Hill Cipher - Chosen Ciphertext Attack (2x2 and 3x3)")
st.write("This app demonstrates a chosen ciphertext attack on the Hill Cipher for 2x2 and 3x3 matrices, using column-wise matrices.")

# Step 1: Input for Known Plaintext and Ciphertext
st.subheader("Step 1: Input Known Plaintext and Ciphertext")
matrix_size = st.selectbox("Select Matrix Size", [2, 3], index=0)

# Inputs for Plaintext and Ciphertext
plain_text_input = st.text_input("Enter a Known Plaintext:", value="ACTG" if matrix_size == 2 else "HELPQUICK")
cipher_text_input = st.text_input("Enter the Corresponding Ciphertext:", value="PQMI" if matrix_size == 2 else "RSQPBFPVI")

# Ensure the input lengths match the chosen matrix size
expected_length = matrix_size ** 2
if len(plain_text_input) != expected_length or len(cipher_text_input) != expected_length:
    st.warning(f"Please enter exactly {expected_length} characters for the plaintext and ciphertext.")
else:
    if st.button("Perform Chosen Ciphertext Attack"):
        # Perform the Chosen Ciphertext Attack
        key_matrix = chosen_ciphertext_attack(plain_text_input, cipher_text_input, matrix_size)
        
        if key_matrix is not None:
            st.success("Key Matrix Successfully Recovered!")
            display_matrix(key_matrix, "Final Recovered Key Matrix")

# Example Instructions
st.write("---")
st.write("### Instructions:")
st.write("""
1. **Matrix Size**: Select either a 2x2 or 3x3 matrix size.
2. **Plaintext**: Enter a known plaintext of appropriate length (4 characters for 2x2, 9 characters for 3x3).
3. **Ciphertext**: Enter the corresponding ciphertext.
4. **Perform Attack**: Click the button to perform the Chosen Ciphertext Attack.
""")











