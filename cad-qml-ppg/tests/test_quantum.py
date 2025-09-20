import pennylane as qml
import numpy as np

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def circuit(theta):
    qml.RX(theta, wires=0)
    return qml.expval(qml.PauliZ(0))

print("PennyLane OK, expval:", circuit(0.1))

from qiskit import __version__ as qv
print("Qiskit OK, version:", qv)