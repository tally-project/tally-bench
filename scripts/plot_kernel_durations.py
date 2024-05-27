import matplotlib.pyplot as plt
import seaborn as sns

def plot_kernel_durations(file_name, out_image="kernel_durations.png"):

    kernel_durations = []
    with open(file_name, "r") as f:
        for line in f:
            kernel_durations.append(float(line.strip()))

    categories = {
        "0 to 0.1": (0, 0.1),
        "0.1 to 1": (0.1, 1),
        "1 to 5": (1, 5),
        "above 5": (5, 10),
    }

    def get_category(value):
        for category, bounds in categories.items():
            if bounds[0] <= value <= bounds[1]:
                return category
        return "Other"

    counts = {category: 0 for category in categories.keys()}

    for value in kernel_durations:
        category = get_category(value)
        if category in counts:
            counts[category] += 1

    labels = list(counts.keys())
    sizes = list(counts.values())

    plt.cla()
    plt.pie(sizes, labels=labels, autopct="%1.1f%%")
    plt.title("Kernel Duration Distribution")
    plt.savefig(out_image)

data_folder = "tally_bench_results/kernel_durations"
plot_kernel_durations(f"{data_folder}/pegasus_kernel_durations.txt", f"{data_folder}/pegasus_kernel_durations.png")
plot_kernel_durations(f"{data_folder}/resnet50_kernel_durations.txt", f"{data_folder}/resnet50_kernel_durations.png")
plot_kernel_durations(f"{data_folder}/whisper_kernel_durations.txt", f"{data_folder}/whisper_kernel_durations.png")