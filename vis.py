import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob


# Funzione per caricare e analizzare i giudizi
def analyze_judges(data_path):
    # Carica tutti i file .npy
    files = glob.glob(str(Path(data_path) / "*.npy"))

    # Leggi i valori
    values = []
    for f in files:
        value = np.load(f)
        values.append(float(value))

    values = np.array(values)

    # Crea l'istogramma
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=20, range=(0, 1), density=True)
    plt.title(f"Distribuzione dei judge in {data_path}")
    plt.xlabel("Valore del judge")
    plt.ylabel("Densità")

    # Stampa alcune statistiche
    print(f"\nStatistiche per {data_path}")
    print(f"Numero di giudizi: {len(values)}")
    print(f"Media: {values.mean():.3f}")
    print(f"Mediana: {np.median(values):.3f}")
    print(f"Dev. Standard: {values.std():.3f}")

    # Stampa la distribuzione dei valori in alcuni intervalli
    intervals = list(zip(np.linspace(0, 1, 11), np.linspace(0.1, 1.1, 11)))
    print("\nDistribuzione per intervalli:")
    for start, end in intervals:
        count = np.sum((values >= start) & (values < end))
        percentage = count / len(values) * 100
        print(f"{start:.1f}-{end:.1f}: {percentage:.1f}%")

    plt.show()


# Analizza sia train che val
paths = ["dataset/2afc/train/traditional/judge", "dataset/2afc/val/traditional/judge"]

for path in paths:
    analyze_judges(path)
