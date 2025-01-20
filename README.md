# QUANTUM-CODES
Code 1
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
classical_bits = [0, 1, 0, 1, 1, 0, 1, 0]
qc = QuantumCircuit(1, 1)  
for bit in classical_bits:
    if bit == 1:
        qc.x(0)  
    qc.measure(0, 0) 
simulator = AerSimulator()
job = simulator.run(qc, shots=1000)
result = job.result()
counts = result.get_counts(qc)
plt.figure(figsize=(8, 6))
plt.bar(counts.keys(), counts.values(), color='blue', alpha=0.7)
plt.xlabel('Measurement Outcomes', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Measurement Results Histogram', fontsize=16)
plt.show()
all_results = []
for bit in classical_bits:
    result_bit = 1 if bit == 1 else 0  
    all_results.extend([result_bit] * 1000) 
sns.kdeplot(np.array(all_results), bw_adjust=0.5, fill=True, color='orange', alpha=0.6)
plt.title('Kernel Density Plot of Measurement Results', fontsize=16)
plt.xlabel('Measurement Outcomes', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.show()

Code 2
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
message = [0, 1, 0, 1, 1, 0, 1, 0]
for bit in message:
    if bit == 1:
        qc.h(0)  
        qc.x(0)  
    else:
        qc.h(0)  
    qc.measure(0, 0)  
simulator = AerSimulator()
job = simulator.run(qc, shots=1000)
result = job.result()
counts = result.get_counts(qc)
plt.figure(figsize=(8, 6))
plt.bar(counts.keys(), counts.values(), color='blue', alpha=0.7)
plt.xlabel('Measurement Outcomes', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Measurement Results Histogram', fontsize=16)
plt.show()
all_results = []
for bit in message:
    result_bit = 1 if bit == 1 else 0  
    all_results.extend([result_bit] * 1000)  
sns.kdeplot(np.array(all_results), bw_adjust=0.5, fill=True, color='orange', alpha=0.6)
plt.title('Kernel Density Plot of Measurement Results', fontsize=16)
plt.xlabel('Measurement Outcomes', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.show()




























Code 3
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
def gaussian_distribution(n, mean=0, std_dev=1):
    qc = QuantumCircuit(n, n)
    angles = np.random.normal(mean, std_dev, n)
    for i in range(n):
        qc.h(i)  
        qc.rz(angles[i], i)  
    qc.measure(range(n), range(n))
    return qc
n = 3  
qc = gaussian_distribution(n)
simulator = Aer.get_backend('qasm_simulator')
job = simulator.run(qc, shots=1000)
result = job.result()
counts = result.get_counts()
plot_histogram(counts)
plt.title(f'{n}-Boyutlu Gauss Dağılımı', fontsize=16)
plt.show()









Code 4
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=1000)
plt.figure(figsize=(10, 6))
sns.kdeplot(data, color='blue', fill=True, alpha=0.5, linewidth=2)
plt.title("KDE Grafiği", fontsize=16)
plt.xlabel("Değerler", fontsize=14)
plt.ylabel("Yoğunluk", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


















Code 5
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=1000)
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=30, color='blue', kde=False, alpha=0.6, edgecolor='black')
plt.title("Histogram Grafiği", fontsize=16)
plt.xlabel("Değerler", fontsize=14)
plt.ylabel("Frekans", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

















Code 6
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=1000)
gmm = GaussianMixture(n_components=2, random_state=42)  
gmm.fit(data.reshape(-1, 1))  
x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)
gmm_pdf = np.exp(gmm.score_samples(x))  
plt.figure(figsize=(10, 6))
plt.plot(x, gmm_pdf, color='red', label='GMM Eğrisi', linewidth=2)
plt.title("GMM Eğrisi", fontsize=16)
plt.xlabel("Değerler", fontsize=14)
plt.ylabel("Yoğunluk", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()













Code 7
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=1000)
gmm = GaussianMixture(n_components=2, random_state=42)  
gmm.fit(data.reshape(-1, 1))  
x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)
gmm_pdf = np.exp(gmm.score_samples(x))  
hist, bins = np.histogram(data, bins=30, density=True)
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='blue', alpha=0.6, edgecolor='black', density=True, label='Histogram')
plt.plot(x, gmm_pdf, color='red', label='GMM Eğrisi', linewidth=2)
sns.kdeplot(data, color='green', label='KDE Eğrisi', linewidth=2)
plt.title("Histogram, GMM ve KDE Eğrisi", fontsize=16)
plt.xlabel("Değerler", fontsize=14)
plt.ylabel("Yoğunluk", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()









Code 8
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from qiskit_aer import Aer 
from qiskit.circuit.random import random_circuit 
from qiskit.quantum_info import Statevector 
num_qubits = 3 
depth = 5 
random_qc = random_circuit(num_qubits, depth, measure=False) 
simulator = Aer.get_backend('statevector_simulator') 
statevector = Statevector.from_instruction(random_qc)  
probabilities = statevector.probabilities_dict() 
data = list(probabilities.values())  
plt.figure(figsize=(10, 6)) 
sns.kdeplot(data, color='blue', fill=True, alpha=0.5, linewidth=2) 
plt.title("Kuantum Devresinden Elde Edilen KDE Grafiği", fontsize=16) 
plt.xlabel("Olasılıklar", fontsize=14) 
plt.ylabel("Yoğunluk", fontsize=14) 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.show() 










Code 9
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from qiskit import QuantumCircuit 
from qiskit.quantum_info import Statevector 
from qiskit.quantum_info import random_statevector 
num_qubits = 3  
state = random_statevector(2**num_qubits)  
statevector = Statevector(state) 
probabilities = statevector.probabilities_dict() 
data = list(probabilities.values())  
plt.figure(figsize=(10, 6)) 
sns.histplot(data, bins=30, color='blue', kde=False, alpha=0.6, edgecolor='black') 
plt.title("Kuantum Devresinden Elde Edilen Histogram Grafiği", fontsize=16) 
plt.xlabel("Olasılıklar", fontsize=14) 
plt.ylabel("Frekans", fontsize=14) 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.show() 












Code 10
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.mixture import GaussianMixture 
from qiskit.quantum_info import Statevector 
from qiskit.quantum_info import random_statevector 
seed = 42 
num_qubits = 3 
state = random_statevector(2**num_qubits, seed=seed)  
statevector = Statevector(state) 
probabilities = statevector.probabilities_dict() 
data = list(probabilities.values
gmm = GaussianMixture(n_components=2, random_state=seed)  
gmm.fit(np.array(data).reshape(-1, 1))  
x = np.linspace(min(data), max(data), 1000).reshape(-1, 1) 
gmm_pdf = np.exp(gmm.score_samples(x))  
plt.figure(figsize=(10, 6)) 
plt.hist(data, bins=15, color='blue', alpha=0.6, edgecolor='black', density=True, label='Histogram') 
plt.plot(x, gmm_pdf, color='red', label='GMM Eğrisi', linewidth=2) 
sns.kdeplot(data, color='green', label='KDE Eğrisi', linewidth=2) 
plt.title("Qiskit Verileriyle Histogram, GMM ve KDE Eğrisi", fontsize=16) 
plt.xlabel("Olasılıklar", fontsize=14) 
plt.ylabel("Yoğunluk", fontsize=14) 
plt.legend() 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.show() 




Code 11
from typing import Any

class Qubit:
    def __init__(self):
        self.state = 0  
    
    def x(self):
        self.state = 1 - self.state  
    
    def measure(self) -> bool:
        return bool(self.state)  
    
    def reset(self):
        self.state = 0  

class QuantumDevice:
    def using_qubit(self) -> Any:
        return self  
    
    def __enter__(self):
        return Qubit()  
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  

def prepare_classical_message(bit: bool, q: Qubit) -> None:
    if bit:
        q.x()  

def eve_measure(q: Qubit) -> bool:
    return q.measure()  

def send_classical_bit(device: QuantumDevice, bit: bool) -> None:
    with device.using_qubit() as q:
        prepare_classical_message(bit, q)  
        result = eve_measure(q)  
        q.reset()  
        assert result == bit, f"Hata: Gönderilen bit {bit}, ölçülen bit {result}"  

device = QuantumDevice()


for bit in [True, False]:
    send_classical_bit(device, bit)
    print(f"Bit {bit} başarıyla gönderildi ve doğrulandı.")



from abc import ABC, abstractmethod

class Qubit(ABC):
    
    
    @abstractmethod
    def h(self):
        
        pass
    
    @abstractmethod
    def x(self):
        
        pass
    
    @abstractmethod
    def measure(self) -> bool:
        pass
    
    @abstractmethod
    def reset(self):
        
        pass






















Code 12
import numpy as np
from abc import ABC, abstractmethod

class Qubit(ABC):
    @abstractmethod
    def h(self):
               pass

    @abstractmethod
    def x(self):
        
        pass

    @abstractmethod
    def measure(self) -> bool:
       
        pass

    @abstractmethod
    def reset(self):
        
        pass

KET_0 = np.array([
    [1],
    [0]
], dtype=complex)

H = np.array([
    [1, 1],
    [1, -1]
], dtype=complex) / np.sqrt(2)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

class SimulatedQubit(Qubit):
    def __init__(self):
        self.reset()

    def h(self):
        self.state = H @ self.state

    def x(self):
        self.state = X @ self.state

    def measure(self) -> bool:
        pr0 = np.abs(self.state[0, 0]) ** 2  
        sample = np.random.random() <= pr0  
        return bool(0 if sample else 1)

    def reset(self):
        self.state = KET_0.copy()

qubit = SimulatedQubit()
print("Başlangıç durumu (|0⟩):", qubit.state)

qubit.h()
print("Hadamard sonrası durum:", qubit.state)

qubit.x()
print("Pauli-X sonrası durum:", qubit.state)

result = qubit.measure()
print("Ölçüm sonucu:", result)

qubit.reset()
print("Reset sonrası durum:", qubit.state)





















Code 13
import numpy as np

class SingleQubitSimulator:
    def __init__(self):
        self.qubit_state = None

    def using_qubit(self):
        
        class QubitContextManager:
            def __init__(self, simulator):
                self.simulator = simulator

            def __enter__(self):
                self.simulator.qubit_state = np.array([[1], [0]], dtype=complex)  
                return self.simulator

            def __exit__(self, exc_type, exc_value, traceback):
                self.simulator.qubit_state = None  

        return QubitContextManager(self)

    def x(self):
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.qubit_state = X @ self.qubit_state

    def measure(self) -> bool:
        
        prob_0 = np.abs(self.qubit_state[0, 0]) ** 2
        measurement = np.random.random() < prob_0
        self.qubit_state = np.array([[1], [0]], dtype=complex) if measurement else np.array([[0], [1]], dtype=complex)
        return measurement



def prepare_classical_message(bit: int, simulator: SingleQubitSimulator):
    
    if bit:
        simulator.x()

def eve_measure(simulator: SingleQubitSimulator) -> bool:
    
    return simulator.measure()

qrng_simulator = SingleQubitSimulator()

key_bit = int(np.random.random() > 0.5)  

qkd_simulator = SingleQubitSimulator()

with qkd_simulator.using_qubit() as q:
    prepare_classical_message(key_bit, q)
    print(f"You prepared the classical key bit: {key_bit}")
    eve_measurement = int(eve_measure(q))
    print(f"Eve measured the classical key bit: {eve_measurement}")





Code 14
import numpy as np
import qutip as qt

ket_0 = qt.basis(2, 0)

H = np.array([[1, 1],
              [1, -1]], dtype=complex) / np.sqrt(2)

ket_plus = qt.Qobj(H @ ket_0.full())  

initial_state = qt.tensor(ket_plus, ket_0)

swap_matrix = np.array([[0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]], dtype=complex)

final_state = qt.Qobj(swap_matrix @ initial_state.full())

print("State after applying SWAP:")
print(final_state)

tensor_product = qt.tensor(ket_0, ket_plus)

print("\nTensor product of |0> and |+>:")
print(tensor_product)




Code 15
from qutip.qip.operations import hadamard_transform
import numpy as np

H = hadamard_transform()

print("Hadamard Matrisi (H):")
print(H)

print("\nHadamard Matrisi Verisi:")
print(np.array(H.full()))




















Code 16
import qutip as qt


ket0 = qt.Qobj([[1], [0]])


print("Ket Durumu (|0>):")
print(ket0)

print("\nKet Durumu Verisi (numpy dizisi olarak):")
print(ket0.full())  




















Code 17
import qutip as qt

ket0 = qt.basis(2, 0)

print("Ket Durumu |0>:")
print(ket0)

ket1 = qt.basis(2, 1)

print("\nKet Durumu |1>:")
print(ket1)

print("\nKet Durumu |0> Verisi (numpy dizisi olarak):")
print(ket0.full())

print("\nKet Durumu |1> Verisi (numpy dizisi olarak):")
print(ket1.full())












Code 18
import qutip as qt

sx = qt.sigmax()

print("Pauli-X Matrisi (σx):")
print(sx)


print("\nPauli-X Matrisi Verisi (numpy dizisi olarak):")
print(sx.full())  




















Code 19
import qutip as qt
from qutip.qip.operations import hadamard_transform

psi = qt.basis(2, 0)  
phi = qt.basis(2, 1)  

tensor_state = qt.tensor(psi, phi)

print("Tensör Durumu (|0> ⊗ |1>):")
print(tensor_state)

H = hadamard_transform()

I = qt.qeye(2)

tensor_op = qt.tensor(H, I)

print("\nHadamard ve Birim Matrislerinin Tensör Çarpımı:")
print(tensor_op)

print("\nTensör Operatör Verisi (numpy dizisi olarak):")
print(tensor_op.full())








Code 20
import qutip as qt
from qutip.qip.operations import hadamard_transform


psi = qt.basis(2, 0)  
phi = qt.basis(2, 1)  

H = hadamard_transform()

I = qt.qeye(2)

H_tensor_I = qt.tensor(H, I)

psi_tensor_phi = qt.tensor(psi, phi)

H_psi = qt.tensor(H * psi, I * phi)


result = H_tensor_I * psi_tensor_phi - H_psi

print("Sonuç:")
print(result)








Code 21
from abc import ABCMeta, abstractmethod


class Qubit(metaclass=ABCMeta):
    
    
    @abstractmethod
    def h(self):
        pass

    @abstractmethod
    def x(self):
        pass

    
    @abstractmethod
    def ry(self, angle: float):
        pass

    
    @abstractmethod
    def measure(self) -> bool:
        pass

    
    @abstractmethod
    def reset(self):
        pass

class QuantumBit(Qubit):
    
    def __init__(self, state: bool = False):
        self.state = state  

    def h(self):
        
        self.state = not self.state
        print(f"Hadamard applied. New state: {self.state}")

    def x(self):
        
        self.state = not self.state
        print(f"Pauli-X applied. New state: {self.state}")

    def ry(self, angle: float):
        
        print(f"Rotating by {angle} radians around Y-axis.")

    def measure(self) -> bool:
        
        print(f"Measuring state: {self.state}")
        return self.state

    def reset(self):
        
        self.state = False
        print("Qubit reset to state 0.")

qubit = QuantumBit(state=False)

qubit.h()

qubit.x()

qubit.ry(1.57) 

measurement = qubit.measure()
print(f"Measurement result: {measurement}")

qubit.reset()


measurement = qubit.measure()
print(f"Measurement result after reset: {measurement}")


















Code 22
from typing import List
import qutip as qt

class SimulatedQubit:
    def __init__(self, simulator, index: int):
        self.simulator = simulator  
        self.index = index          
        self.state = qt.basis(2, 0)  

    def __str__(self):
        return f"Qubit {self.index}, state: {self.state}"

class QuantumDevice:
    def __init__(self):
        pass  


class Simulator(QuantumDevice):
    capacity: int
    available_qubits: List[SimulatedQubit]
    register_state: qt.Qobj

    def __init__(self, capacity: int = 3):
        self.capacity = capacity
        self.available_qubits = [
            SimulatedQubit(self, idx) for idx in range(capacity)
        ]
        self.register_state = qt.tensor(
            *[qt.basis(2, 0) for _ in range(capacity)]
        )

    def allocate_qubit(self) -> SimulatedQubit:
        if self.available_qubits:
            return self.available_qubits.pop()
        else:
            raise Exception("No available qubits to allocate.")

    def deallocate_qubit(self, qubit: SimulatedQubit):
        self.available_qubits.append(qubit)

    def get_register_state(self):
        return self.register_state

sim = Simulator(capacity=5)

qubit1 = sim.allocate_qubit()
print(f"Allocated qubit: {qubit1}")

sim.deallocate_qubit(qubit1)
print(f"Deallocated qubit: {qubit1}")

print(f"Remaining available qubits: {[q.index for q in sim.available_qubits]}")

print("Current register state:")
print(sim.get_register_state())
