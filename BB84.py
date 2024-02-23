import numpy as np
from quantum_utils import generate_secret_key, cipher_message, decipher_message

def main():
    choice = input("Choose an option (encrypt/decrypt): ").strip().lower()
    if choice == "encrypt":
        text = input("Enter the text to encrypt: ")
        key = generate_secret_key()[:len(text)]
        encrypted_text = cipher_message(text, ''.join(map(chr, key)))
        print(f"Encrypted text: {encrypted_text}")
    elif choice == "decrypt":
        encrypted_text = input("Enter the encrypted text: ")
        key = input("Enter the key: ")
        decrypted_text = decipher_message(encrypted_text, key)
        print(f"Decrypted text: {decrypted_text}")
    else:
        print("Invalid choice. Please enter 'encrypt' or 'decrypt'.")

if __name__ == "__main__":
    main()
