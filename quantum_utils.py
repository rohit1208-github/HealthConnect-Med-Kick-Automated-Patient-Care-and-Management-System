import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, BasicAer

# Constants
NUM_QUBITS = 15
MAX_RANDOM_SEED = 2**NUM_QUBITS

# Functions
def generate_secret_key():
    return np.random.randint(0, 256, 60, dtype='int')

def create_sender_circuit(secret_key):
    sender_quantum_register = QuantumRegister(60, name='sender-qureg') 
    sender_classical_register = ClassicalRegister(60, name='sender-clreg')
    
    sender_quantum_circuit = QuantumCircuit(sender_quantum_register, sender_classical_register)

    # Apply X gates based on secret key array 
    for index, bit in enumerate(secret_key):
        if bit == 1:
            sender_quantum_circuit.x(sender_quantum_register[index])

    return sender_quantum_circuit, sender_quantum_register, sender_classical_register

def apply_polarization(sender_quantum_circuit, sender_quantum_register):
    polarization_table = []
    for index in range(NUM_QUBITS):
        polarization_axis = np.random.random()
        if polarization_axis < 0.25:
            polarization_table.append("↕")
        elif polarization_axis < 0.5:
            sender_quantum_circuit.h(sender_quantum_register[index])
            polarization_table.append("⤢")
        elif polarization_axis < 0.75:
            polarization_table.append("↔")
        else:
            sender_quantum_circuit.h(sender_quantum_register[index])
            polarization_table.append("⤡")
    return polarization_table

def send_quantum_state(sender_quantum_circuit):
    receiver_quantum_register = QuantumRegister(NUM_QUBITS, name='receiver-qureg')
    receiver_classical_register = ClassicalRegister(NUM_QUBITS, name='receiver-clreg')
    receiver_quantum_circuit = QuantumCircuit(receiver_quantum_register, receiver_classical_register)

    # Create a mapping of sender qubit indexes to receiver qubit indexes  
    qubit_mapping = {qubit.index: index for index, qubit in enumerate(receiver_quantum_register)}
    
    # Append mapped gates to receiver circuit
    for gate in sender_quantum_circuit: 
        receiver_quantum_circuit.append(gate[0], 
                                       [qubit_mapping[qubit.index] for qubit in gate[1]])

    return receiver_quantum_circuit, receiver_quantum_register, receiver_classical_register

def measure_receiver_state(receiver_quantum_circuit, receiver_quantum_register):
    filtration_table = []
    for index in range(NUM_QUBITS):
        measurement_axis = np.random.random()
        if measurement_axis < 0.5:
            receiver_quantum_circuit.h(receiver_quantum_register[index])
            filtration_table.append("✕")
        else:
            filtration_table.append("✛")
        receiver_quantum_circuit.measure(receiver_quantum_register[index], index)
    
    return filtration_table

def execute_quantum_circuit(quantum_circuit):
    backend = BasicAer.get_backend('qasm_simulator')
    result = execute(quantum_circuit, backend=backend, shots=1).result()

    # Initialize counts dictionary to return all qubit measurement results
    num_qubits = len(quantum_circuit.qubits)
    counts = {format(n, '0'+str(num_qubits)+'b'): 0 for n in range(2**num_qubits)}

    # Update with returned counts
    counts.update(result.get_counts(quantum_circuit))
    
    return counts

def extract_final_secret_key(polarization_table, filtration_table, measured_values):

    final_key = []
    bitstring = list(measured_values.keys())[0]
    
    for index in reversed(range(len(bitstring))):
        
        bit = bitstring[index]
        
        final_key.insert(0, int(bit)) 

    return "".join([str(b) for b in final_key][:len(bitstring)])

def cipher_message(msg, key):
    return "".join([chr(ord(c) ^ ord(k)) for c,k in zip(msg, key)])

def decipher_message(encrypted, key):
    return cipher_message(encrypted, key)

