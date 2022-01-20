import numpy as np
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image

root = Tk()

# Definim functia sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Algoritmul Perceptron
# Inputs
training_inputs = np.array([[0, 0, 1, 0],
                            [1, 1, 1, 1],
                            [1, 0, 1, 1],
                            [0, 1, 1, 0],
                            [1, 1, 0, 1]])

# Outputs
# x.T este transpusa lui x, facandu-l un vector coloana
training_outputs = np.array([[0, 1, 0, 1, 1]]).T

# Alegem o samanta aleatorie pentru rezultate reproductibile
np.random.seed()

synaptic_weights = 2 * np.random.random((4, 1)) - 1

print("Initializarea random a ponderilor: ")
print(synaptic_weights)

# Aplicam metoda Backpropagation cu numarul de iteratii 30000
for i in range(30000):
    # Faza forward
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    # Faza backward
    # Termenul de eroare al stratului de iesire
    err = training_outputs - outputs
    adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))
    # Actualizam ponderile
    synaptic_weights += adjustments
print("Ponderile dupa invatare: ")
print(synaptic_weights)
# Printam iesirile finale ale retelei neuronale
print("Rezultatul: ")
print(outputs)

# Adaugam 2 situatii noi pentru a verifica daca totul funcioneaza bine
new_inputs = np.array([[0, 1, 0, 1],
                       [1, 0, 1, 0]])
outputs = sigmoid(np.dot(new_inputs, synaptic_weights))
print("Situatia noua: ")
print(outputs)
canvas = tk.Canvas(root, width=1100, height=600)
canvas.grid(columnspan=3)
root.title("Perceptron Project by Gorea Andrei")
root.iconbitmap("c:/utm.ico")
img = Image.open("c:/perceptron.png")
img = ImageTk.PhotoImage(img)
img_label = tk.Label(image=img)
img_label.image = img
img_label.grid(column = 1, row = 0)
root.mainloop()
