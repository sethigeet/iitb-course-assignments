import json
import random
from collections import deque

generator1=0o5
generator2=0o7

random.seed(42) ### For reproducibility. Comment it out when you want to create testcases

def add_noise(signal, probability_matrix):
    noisy_output = []
    for bit in signal:
        noisy_output.append(random.choices(range(len(probability_matrix[bit])), weights=probability_matrix[bit], k=1)[0])
    return noisy_output

def encoder(systematic_input, generator_1, generator_2, history_length=3):
    binary_generator_feedforward = bin(generator_1)[2:][::-1].zfill(history_length)  
    binary_generator_feedback = bin(generator_2)[2:][::-1].zfill(history_length)  
    history = deque([0] * history_length)
    systematic=[]
    parity = []
    for bit in systematic_input:
        systematic.append(bit)
        history.popleft()
        history.append(bit)
        feedback_val=sum(int(bit) * int(g) for bit, g in zip(history, binary_generator_feedback)) % 2
        history.pop()
        history.append(feedback_val)
        parity.append(sum(int(bit) * int(g) for bit, g in zip(history, binary_generator_feedforward)) % 2)
    return systematic, parity

def generate_bitstring(length=1000):
    return [random.randint(0,1) for _ in range(length)]

def interleave(array1, array2, array3=None):
    interleaved = []
    for i in range(len(array1)):
        interleaved.append(array1[i])
        interleaved.append(array2[i])
        if array3 is not None:
            interleaved.append(array3[i])
    return interleaved

def normalize(array):
    total = sum(array)
    return [x / total for x in array] if total > 0 else array

def generate_testcases(num_cases=20):
    matrices = [
        [20, 10, 3, 2, 1],
        [50, 10, 5, 4, 3, 2, 1]
    ]
    
    testcases = []
    for _ in range(num_cases):
        bitstring = generate_bitstring(1000)
        systematic, parity1=encoder(bitstring, generator1, generator2, history_length=3)
        permutation = list(range(len(bitstring)))
        random.shuffle(permutation)
        _, parity2=encoder([bitstring[permutation[i]] for i in range(len(permutation))], generator1, generator2, history_length=3)
        prob_matrix = random.choice(matrices)
        probability_matrix = [normalize(prob_matrix), normalize(prob_matrix)[::-1]]

        noisy_output=add_noise(interleave(systematic, parity1, parity2), probability_matrix)
        testcases.append({
            "bitstring": "".join(map(str, bitstring)),
            "noisy_output": noisy_output,
            "probability_matrix": prob_matrix,
            "permutation": permutation,
        })
    return testcases

if __name__ == "__main__":
    data = {"testcases": generate_testcases(20)}
    
    with open("turbocodes_testcases.json", "w") as f:
        json.dump(data, f, indent=4)
