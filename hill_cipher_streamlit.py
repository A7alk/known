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
    attempt_count = 0
    while True:
        attempt_count += 1
        # Generate a random matrix of the given size
        matrix = np.random.randint(0, mod, (size, size))
        det = int(np.round(np.linalg.det(matrix)))  # Calculate determinant
        gcd_value = np.gcd(det, mod)  # Calculate GCD

        st.write(f"Attempt {attempt_count}: Determinant = {det} (mod {mod} = {det % mod}), GCD(Determinant, {mod}) = {gcd_value}")
        if gcd_value == 1:  # Check if the matrix is invertible
            st.write("Found an invertible matrix!")
            return matrix

# Function for Chosen Ciphertext Attack
def chosen_ciphertext_attack(plain_text, cipher_text, size, auto_generate=False):
    """Performs Chosen Ciphertext Attack to recover the key matrix."""
    mod = 26

    if auto_generate:
        st.write("### Automatically Generating an Invertible Ciphertext Matrix...")
        # Generate an invertible ciphertext matrix and its corresponding plaintext matrix
        cipher_matrix = generate_invertible_matrix(size)
        st.write("**Generated Invertible Ciphertext Matrix:**")
        display_matrix(cipher_matrix, "Generated Ciphertext Matrix (Invertible)")

        # Generate a random plaintext matrix of the same size
        plain_matrix = generate_invertible_matrix(size)
        st.write("**Generated Plaintext Matrix:**")
        display_matrix(plain_matrix, "Generated Plaintext Matrix")

        # Return these matrices for automatic attack demonstration
        return plain_matrix, cipher_matrix

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

    # Step 3: Calculate the Inverse of the Ciphertext Matrix (instead of plaintext)
    st.write("### Step 3: Calculating the Inverse of the Ciphertext Matrix")
    inv_cipher_matrix = matrix_mod_inverse(cipher_matrix, mod)
    if inv_cipher_matrix is None:
        return None, None

    display_matrix(inv_cipher_matrix, "Inverse of Ciphertext Matrix (mod 26)")

    # Step 4: Calculate the Key Matrix using the equation: Key = Plain * Inverse(Cipher)
    st.write("### Step 4: Calculating the Key Matrix using `Key = Plain * Inverse(Cipher)`")
    key_matrix = np.dot(plain_matrix, inv_cipher_matrix) % mod
    display_matrix(key_matrix, "Recovered Key Matrix (mod 26)")

    return plain_matrix, key_matrix

# Streamlit UI for Hill Cipher - Chosen Ciphertext Attack
st.title("Hill Cipher - Chosen Ciphertext Attack")
st.write("This app demonstrates a chosen ciphertext attack on the Hill Cipher for 2x2 and 3x3 matrices, using column-wise matrices.")

# Input for Known Plaintext and Ciphertext
st.subheader("Step 1: Input Known Plaintext and Ciphertext")
matrix_size = st.selectbox("Select Matrix Size", [2, 3], index=0)

# Option to Automatically Generate a Valid Matrix Pair
auto_generate = st.checkbox("Automatically Generate a Valid Plaintext-Ciphertext Pair")

# Inputs for Plaintext and Ciphertext
plain_text_input = st.text_input("Enter a Known Plaintext:", value="ACTG" if matrix_size == 2 else "ATTACKNOW")
cipher_text_input = st.text_input("Enter the Corresponding Ciphertext:", value="PQMI" if matrix_size == 2 else "FTZZHPXOA")

# Ensure the input lengths match the chosen matrix size
expected_length = matrix_size ** 2
if len(plain_text_input) != expected_length or len(cipher_text_input) != expected_length:
    st.warning(f"Please enter exactly {expected_length} characters for the plaintext and ciphertext.")
else:
    if st.button("Perform Chosen Ciphertext Attack"):
        # Perform the Chosen Ciphertext Attack
        plain_matrix, key_matrix = chosen_ciphertext_attack(plain_text_input, cipher_text_input, matrix_size, auto_generate)
        
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
4. **Automatically Generate**: Optionally enable the checkbox to automatically generate a valid pair.
""")















