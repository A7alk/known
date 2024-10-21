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
    """Finds the modular inverse of a matrix and displays detailed steps."""
    st.write("### Matrix Inversion Steps")
    
    # Step 1: Calculate the determinant
    det = int(np.round(np.linalg.det(matrix)))  # Calculate the determinant
    st.write(f"**Step 1:** Determinant of the matrix: {det} (mod {mod} = {det % mod})")
    
    # GCD check to ensure invertibility
    gcd_value = np.gcd(det, mod)
    st.write(f"**Step 2:** GCD of determinant and {mod}: {gcd_value}")

    if gcd_value != 1:
        st.error("This matrix is not invertible under mod 26 because GCD(det, 26) is not 1.")
        return None

    # Step 3: Calculate modular inverse of determinant mod 26
    det_mod_inv = pow(det % mod, -1, mod)
    st.write(f"**Step 3:** Modular inverse of determinant under mod {mod}: {det_mod_inv}")

    # Step 4: Calculate cofactor matrix (for adjugate matrix)
    cofactor_matrix = np.linalg.inv(matrix).T * det  # Cofactor matrix calculation
    st.write(f"**Step 4:** Cofactor Matrix (before mod {mod}):")
    st.write(np.round(cofactor_matrix).astype(int))

    # Step 5: Calculate adjugate matrix (transpose of cofactor matrix)
    adjugate_matrix = np.round(cofactor_matrix).astype(int) % mod
    st.write(f"**Step 5:** Adjugate Matrix (Cofactor Matrix mod {mod}):")
    st.write(adjugate_matrix)

    # Step 6: Calculate the inverse matrix by multiplying adjugate matrix by modular inverse of determinant
    inv_matrix = (det_mod_inv * adjugate_matrix) % mod
    st.write(f"**Step 6:** Final Inverse Matrix (mod {mod}):")
    st.write(inv_matrix)

    return inv_matrix

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

        if gcd_value == 1:  # Check if the matrix is invertible
            return matrix

# Function for Chosen Ciphertext Attack (Finding Decryption Key)
def chosen_ciphertext_attack(plain_text, cipher_text, size, auto_generate=False, invert_choice="Ciphertext"):
    """Performs Chosen Ciphertext Attack to recover the decryption key."""
    mod = 26

    if auto_generate:
        st.write("### Automatically Generating an Invertible Matrix...")
        # Generate an invertible matrix for both plaintext and ciphertext
        cipher_matrix = generate_invertible_matrix(size)
        plain_matrix = generate_invertible_matrix(size)

        # Return generated matrices
        display_matrix(cipher_matrix, "Generated Ciphertext Matrix")
        display_matrix(plain_matrix, "Generated Plaintext Matrix")
        return plain_matrix, cipher_matrix

    # Convert text input to numeric values
    st.write("### Step 1: Preparing the Plaintext and Ciphertext Matrices (Column-wise)")
    plain_numeric = text_to_numeric(plain_text)
    cipher_numeric = text_to_numeric(cipher_text)

    st.write(f"**Plaintext Numeric Values:** {plain_numeric}")
    st.write(f"**Ciphertext Numeric Values:** {cipher_numeric}")

    # Reshape into matrices (column-wise)
    plain_matrix = np.array(plain_numeric).reshape(size, size, order='F')
    cipher_matrix = np.array(cipher_numeric).reshape(size, size, order='F')

    display_matrix(plain_matrix, "Plaintext Matrix")
    display_matrix(cipher_matrix, "Ciphertext Matrix")

    # Step 2: Inversion Based on User Choice
    if invert_choice == "Ciphertext":
        st.write("### Step 2: Inverting the Ciphertext Matrix")
        inv_matrix = matrix_mod_inverse(cipher_matrix, mod)
        if inv_matrix is None:
            return None, None
        display_matrix(inv_matrix, "Inverse of Ciphertext Matrix (mod 26)")
        
        # Calculate the decryption key: Decryption Key = Inverse(Ciphertext) * Plaintext
        st.write("### Step 3: Calculating the Decryption Key")
        decryption_key_matrix = np.dot(inv_matrix, plain_matrix) % mod
    else:
        st.write("### Step 2: Inverting the Plaintext Matrix")
        inv_matrix = matrix_mod_inverse(plain_matrix, mod)
        if inv_matrix is None:
            return None, None
        display_matrix(inv_matrix, "Inverse of Plaintext Matrix (mod 26)")

        # Calculate the decryption key: Decryption Key = Ciphertext * Inverse(Plaintext)
        st.write("### Step 3: Calculating the Decryption Key")
        decryption_key_matrix = np.dot(cipher_matrix, inv_matrix) % mod

    display_matrix(decryption_key_matrix, "Recovered Decryption Key Matrix (mod 26)")
    return plain_matrix, decryption_key_matrix

# Streamlit UI for Hill Cipher - Chosen Ciphertext Attack
st.title("Hill Cipher - Chosen Ciphertext Attack")
st.write("This app demonstrates a chosen ciphertext attack on the Hill Cipher for 2x2 and 3x3 matrices, using column-wise matrices.")

# User input for known plaintext and ciphertext
st.subheader("Step 1: Input Known Plaintext and Ciphertext")
matrix_size = st.selectbox("Select Matrix Size", [2, 3], index=0)

# Option to Automatically Generate a Valid Matrix Pair
auto_generate = st.checkbox("Automatically Generate a Valid Plaintext-Ciphertext Pair")

# Inputs for Plaintext and Ciphertext
plain_text_input = st.text_input("Enter a Known Plaintext:", value="ACTG" if matrix_size == 2 else "ATTACKNOW")
cipher_text_input = st.text_input("Enter the Corresponding Ciphertext:", value="PQMI" if matrix_size == 2 else "FTZZHPXOA")

# Choice: Invert Ciphertext or Plaintext
invert_choice = st.radio("Choose which matrix to invert:", ("Ciphertext", "Plaintext"))

# Ensure the input lengths match the chosen matrix size
expected_length = matrix_size ** 2
if len(plain_text_input) != expected_length or len(cipher_text_input) != expected_length:
    st.warning(f"Please enter exactly {expected_length} characters for the plaintext and ciphertext.")
else:
    if st.button("Perform Chosen Ciphertext Attack"):
        # Perform the Chosen Ciphertext Attack
        plain_matrix, decryption_key_matrix = chosen_ciphertext_attack(plain_text_input, cipher_text_input, matrix_size, auto_generate, invert_choice)
        
        if decryption_key_matrix is not None:
            st.success("Decryption Key Matrix Successfully Recovered!")
            display_matrix(decryption_key_matrix, "Final Recovered Decryption Key Matrix")

# Example Instructions
st.write("---")
st.write("### Instructions:")
st.write("""
1. **Matrix Size**: Select either a 2x2 or 3x3 matrix size.
2. **Plaintext**: Enter a known plaintext of appropriate length (4 characters for 2x2, 9 characters for 3x3).
3. **Ciphertext**: Enter the corresponding ciphertext.
4. **Matrix Inversion Choice**: Select whether to invert the Ciphertext or Plaintext matrix.
5. **Automatically Generate**: Optionally enable the checkbox to automatically generate a valid pair.
""")



                                                                                                                               















