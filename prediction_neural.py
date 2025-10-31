from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
test_images_flat = test_images.reshape((10000, 28 * 28)).astype("float32") / 255


with open("risultati.txt") as f:
    line = f.readline()
    miglior_modello_id = int(line.split("Modello")[1].split()[0])

print(f"Caricamento del modello migliore: mio_modello_v{miglior_modello_id}.keras")
miglior_modello = keras.models.load_model(f"mio_modello_v{miglior_modello_id}.keras")


predizioni = miglior_modello.predict(test_images_flat)
classi_predette = np.argmax(predizioni, axis=1)


for i in range(10):
    print(f"Immagine {i} -> Predetto: {classi_predette[i]}, Reale: {test_labels[i]}")

for i in range(5):
    plt.imshow(test_images[i], cmap="gray")
    plt.title(f"Predetto: {classi_predette[i]} | Reale: {test_labels[i]}")
    plt.axis("off")
    plt.show()
