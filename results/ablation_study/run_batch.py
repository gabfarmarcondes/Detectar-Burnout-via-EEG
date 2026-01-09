import subprocess
import re
import numpy as np

def run_training_loop(n_runs=5):
    losses = []
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
        match = re.search(r"Error \(Loss\): (\d+\.\d+)", output)

        if match:
            loss = float(match.group(1))
            losses.append(loss)
            print(f"Registered Loss: {loss}")
        else:
            print("Error to read Loss. Check the training output.")
            print(output)
        
        if not losses:
            return None, None

        losses_mean = np.mean(losses)
        sd = np.std(losses)

        print("\n" + "="*40)
        print(f"Final Result: {n_runs} Sessions.")
        print(f"Loss Mean: {losses_mean}")
        print(f"Standard Deviation: {sd}")
        print(f"Brute Values: {losses}")
        print("\n" + "="*40)

    return losses, losses_mean, sd

if __name__ == "__main__":
    run_training_loop()