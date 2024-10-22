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
    """Finds the modular inverse of a matrix mod 26 using sympy."""
    st.write("### Matrix Inversion Steps")
    
    # Convert matrix to sympy Matrix to handle mod inverse
    sympy_matrix = Matrix(matrix)
    
    # Calculate the determinant and check invertibility
    det = int(sympy_matrix.det()) % mod
    gcd_value = np.gcd(det, mod)

    st.write(f"Determinant of matrix: {det} (mod {mod})")
    st.write(f"GCD of determinant and {mod}: {gcd_value}")

    if gcd_value != 1:
        st.error(f"This matrix is not invertible under mod {mod} because GCD({det}, {mod}) is {gcd_value}.")
        return None

    try:
        # Find the modular inverse of the matrix using sympy's inv_mod function
        inv_matrix = sympy_matrix.inv_mod(mod)
        st.write(f"**Step:** Modular Inverse of Matrix (mod {mod}):")
        st.write(np.array(inv_matrix).astype(int))  # Display the matrix as a NumPy array
        return np.array(inv_matrix).astype(int)
    except ValueError as e:
        st.error(f"Matrix is not invertible under mod {mod}. Error: {str(e)}")
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
        try:
            Matrix(matrix).inv_mod(mod)  # Try inverting the matrix
            return matrix  # Return the matrix if invertible
        except ValueError:
            pass  # Continue if matrix is not invertible

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






                                                                                                                               















