import subprocess
import re
import numpy as np

def run_training_loop(n_runs=5):
    losses = []
    accuracies = []
    print(f"Initiating {n_runs} Trainings Sessions.")

    for i in range(n_runs):
        print(f"\nRunning {i+1}/{n_runs} Training Session.")

        # Chama o script de treino e captura o que ele imprime na tela
        result = subprocess.run(
            ['python', 'src/train_fewshot.py'],
            capture_output=True,
            text=True
        )

        # Procura pelo valor de Loss no console
        output = result.stdout
        match = re.search(r"Test\s+->\s+Error\s+\(Loss\):\s+(\d+\.\d+)\s+\|\s+Acc(?:uracy)?:\s+(\d+\.\d+)%", output)

        if match:
            loss = float(match.group(1))
            acc = float(match.group(2))
            losses.append(loss)
            accuracies.append(acc)
            print(f"Registered Loss: {loss} | Acc: {acc}")
        else:
            print("Error to read Loss. Check the training output.")
            print(output)
        
        if not losses:
            print("\nNo results captured. Something went wrong.")
            return None, None

        losses_mean = np.mean(losses)
        std = np.std(losses)

        acc_mean = np.mean(accuracies)
        acc_std = np.std(accuracies)

    print("\n" + "="*50)
    print(f"Final Results ({n_runs} Sessions)")
    print("="*50)
    print(f"LOSS     -> Mean: {losses_mean:.4f}  | Std Dev: {std:.4f}")
    print(f"ACCURACY -> Mean: {acc_mean:.2f}% | Std Dev: {acc_std:.2f}")
    print("-" * 50)
    print(f"Raw Losses: {losses}")
    print(f"Raw Accs:   {accuracies}")
    print("="*50)

    return losses, losses_mean, std

if __name__ == "__main__":
    run_training_loop()