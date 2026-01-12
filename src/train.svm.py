from matplotlib import pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import config

DATA_PATH = config.PROCESSED_DATA_DIR

def train_svm_benchmark():
    print("Starting Benchmark with SVM (Support Vector Machine)")
    print(" -> Loading Data Manually (Subject Isolation Strategy)")

    try:
        X = np.load('data/processed/X_stew.npy')
        Y = np.load('data/processed/Y_stew.npy')
    except FileNotFoundError:
        print("ERROR: .npy files not found. Run preprocessing.py first.")
        return

    print(f" -> Original Shape (3D) {X.shape}")

    # Pré-processamento para SVM (Flatten)
    # O SVM não entende 3D, ele precisa ser 2D
    # Transformar (3552, 14, 33, 17) em (3552, 7854)
    num_samples = X.shape[0]
    X_flat = X.reshape(num_samples, -1)

    # Divisão dos dados (80/20)
    # Fatiamento manual
    split_index = int(0.8 * num_samples)

    X_train = X_flat[:split_index]
    Y_train = Y[:split_index]

    X_test = X_flat[split_index:]
    Y_test = Y[split_index:]

    print(f" -> Split: {len(X_train)} Train samples | {len(X_test)} Test samples")

    # Treinamento
    print(" -> Training Linear SVM")
    clf = SVC(kernel='linear', random_state=42) #rando_state para reprodutibilidade
    clf.fit(X_train, Y_train)

    # Predição
    print(" -> Predicting")
    Y_pred = clf.predict(X_test)

    # Acurácia
    acc = accuracy_score(Y_test, Y_pred)
    print(f"\n=== RESULTS ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print("-" * 30)

    # Relatório de Classificação (Precision, Recall e F1-Score)
    print("Classification Report (F1-Score)")
    print(classification_report(Y_test, Y_pred, target_names=['Relaxado', 'Burnout']))
    print("-" * 30)

    # Matriz de Confusão
    cm = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix:")
    print(cm)

def plot_confusion_matrix(cm):
    # Função auxiliar visual para entender os erros
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Relaxado', 'Burnout'], 
                yticklabels=['Relaxado', 'Burnout'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('SVM Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    train_svm_benchmark()