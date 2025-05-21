import csv
import matplotlib.pyplot as plt

best_scores = []
second_best_scores = []

with open("../reports/tetris_ga_log.txt", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if len(row) >= 7 and row[0].isdigit():
            best_fitness = float(row[1])
            second_best_fitness = float(row[2])
            pieces_played_best = int(row[4])
            best_score = float(row[5])
            # Estimate second-best score
            second_best_score = second_best_fitness - (pieces_played_best * 5)
            print(row, "  -->  ", second_best_score)
            best_scores.append(best_score)
            second_best_scores.append(second_best_score)

# Generate the plot
generations = list(range(1, len(best_scores) + 1))

plt.figure(figsize=(10, 6))
plt.plot(
    generations,
    best_scores,
    label="Best Chromosome Score",
    color="#4CAF50",
    marker="o",
    linestyle="-",
)
plt.plot(
    generations,
    second_best_scores,
    label="Second Best Chromosome Score",
    color="#2196F3",
    marker="s",
    linestyle="--",
)
plt.title("Score Progress of Best and Second Best Chromosomes")
plt.xlabel("Generation")
plt.ylabel("Score (Points from Line Clears)")
plt.grid(True)
plt.legend()
plt.tight_layout()
# Save the plot
plt.savefig("../reports/chromosome_score_progress.png", dpi=400)
plt.close()
