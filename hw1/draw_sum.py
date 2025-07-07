import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the CSV data
df = pd.read_csv("sum_performance_no_opt.csv")

# Figure 1: Execution time vs Array Size
plt.figure(figsize=(10, 6))
plt.plot(df["Array_Sizes"], df["Common_Time"], "o-", label="Common Sum")
plt.plot(df["Array_Sizes"], df["Multichain_Time"], "s-", label="Multichain")
plt.plot(df["Array_Sizes"], df["Superscalar_Time"], "^-", label="Recursion")
plt.plot(df["Array_Sizes"], df["Multichain_Unroll_Time"], "D-", label="Multichain_Unroll (4-way)")
plt.plot(df["Array_Sizes"], df["Multichain_Unroll16_Time"], "*-", label="Multichain_Unroll (16-way)")

plt.xscale("log", base=2)  # Log scale for x-axis with base 2
plt.yscale("log")  # Log scale for y-axis
plt.xlabel("Array Size")
plt.ylabel("Execution Time (seconds)")
plt.title("Average Execution Time vs Array Size")
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend()
plt.savefig("./output/sum_execution_time.pdf", format = "pdf", bbox_inches="tight")
plt.close()

# Figure 2: Speedup compared to common sum
plt.figure(figsize=(10, 6))
speedup_multichain = df["Common_Time"] / df["Multichain_Time"]
speedup_superscalar = df["Common_Time"] / df["Superscalar_Time"]
speedup_unroll = df["Common_Time"] / df["Multichain_Unroll_Time"]
speedup_unroll16 = df["Common_Time"] / df["Multichain_Unroll16_Time"]

plt.plot(df["Array_Sizes"], speedup_multichain, "s-", label="Multichain")
plt.plot(df["Array_Sizes"], speedup_superscalar, "^-", label="Recursion")
plt.plot(df["Array_Sizes"], speedup_unroll, "D-", label="Multichain_Unroll (4-way)")
plt.plot(df["Array_Sizes"], speedup_unroll16, "*-", label="Multichain_Unroll (16-way)")

# Add horizontal line at y=1 for reference (same speed as common sum)
plt.axhline(y=1, color="r", linestyle="--", alpha=0.7)

plt.xscale("log", base=2)
plt.xlabel("Array Size")
plt.ylabel("Speedup (relative to Common Sum)")
plt.title("Speedup Relative to Common Sum")
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend()
plt.savefig("./output/sum_speedup.pdf", format = "pdf", bbox_inches="tight")
plt.close()

# Figure 3: Time per element vs Array Size
plt.figure(figsize=(10, 6))
time_per_element_common = df["Common_Time"] / df["Array_Sizes"]
time_per_element_multichain = df["Multichain_Time"] / df["Array_Sizes"]
time_per_element_superscalar = df["Superscalar_Time"] / df["Array_Sizes"]
time_per_element_unroll = df["Multichain_Unroll_Time"] / df["Array_Sizes"]
time_per_element_unroll16 = df["Multichain_Unroll16_Time"] / df["Array_Sizes"]

plt.plot(df["Array_Sizes"], time_per_element_common * 1e9, "o-", label="Common Sum")
plt.plot(df["Array_Sizes"], time_per_element_multichain * 1e9, "s-", label="Multichain")
plt.plot(df["Array_Sizes"], time_per_element_superscalar * 1e9, "^-", label="Superscalar")
plt.plot(df["Array_Sizes"], time_per_element_unroll * 1e9, "D-", label="Unroll (4-way)")
plt.plot(df["Array_Sizes"], time_per_element_unroll16 * 1e9, "*-", label="Unroll (16-way)")

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Array Size")
plt.ylabel("Time per Element (nanoseconds)")
plt.title("Time per Element vs Array Size")
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend()
plt.savefig("./output/sum_time_per_element.pdf", format = "pdf", bbox_inches="tight")
plt.close()

# Figure 4: Speedup comparison between different methods
plt.figure(figsize=(10, 6))
speedup_multichain_vs_common = df["Common_Time"] / df["Multichain_Time"]
speedup_unroll4_vs_multichain = df["Multichain_Time"] / df["Multichain_Unroll_Time"]
speedup_unroll16_vs_unroll4 = df["Multichain_Unroll_Time"] / df["Multichain_Unroll16_Time"]

plt.plot(df["Array_Sizes"], speedup_multichain_vs_common, "s-", label="Multichain vs Common")
plt.plot(df["Array_Sizes"], speedup_unroll4_vs_multichain, "D-", label="4-way Unroll vs Multichain")
plt.plot(df["Array_Sizes"], speedup_unroll16_vs_unroll4, "*-", label="16-way vs 4-way Unroll")

plt.axhline(y=1, color="r", linestyle="--", alpha=0.7)
plt.xscale("log", base=2)
plt.xlabel("Array Size")
plt.ylabel("Speedup")
plt.title("Incremental Optimization Speedup")
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend()
plt.savefig("./output/sum_incremental_speedup.pdf", format = "pdf", bbox_inches="tight")
plt.close()

# Generate a comparison table for large arrays (last 5 entries)
print("\nPerformance comparison for large arrays:")
large_arrays = df.tail(5).copy()
large_arrays["Multichain_Speedup"] = large_arrays["Common_Time"] / large_arrays["Multichain_Time"]
large_arrays["Superscalar_Speedup"] = large_arrays["Common_Time"] / large_arrays["Superscalar_Time"]
large_arrays["Unroll4_Speedup"] = large_arrays["Common_Time"] / large_arrays["Multichain_Unroll_Time"]
large_arrays["Unroll16_Speedup"] = large_arrays["Common_Time"] / large_arrays["Multichain_Unroll16_Time"]

print(large_arrays[["Array_Sizes", "Multichain_Speedup", "Superscalar_Speedup", 
                   "Unroll4_Speedup", "Unroll16_Speedup"]])

# Save this comparison table to CSV
# large_arrays[["Array_Sizes", "Multichain_Speedup", "Superscalar_Speedup", 
#               "Unroll4_Speedup", "Unroll16_Speedup"]].to_csv("speedup_comparison.csv", index=False)

# Optional: Create a bar chart of the speedups for the largest array size
plt.figure(figsize=(10, 6))
largest_size = df['Array_Sizes'].max()
largest_data = df[df['Array_Sizes'] == largest_size]

methods = ['Multichain', 'Superscalar', 'Unroll (4-way)', 'Unroll (16-way)']
speedups = [
    largest_data['Common_Time'].values[0] / largest_data['Multichain_Time'].values[0],
    largest_data['Common_Time'].values[0] / largest_data['Superscalar_Time'].values[0],
    largest_data['Common_Time'].values[0] / largest_data['Multichain_Unroll_Time'].values[0],
    largest_data['Common_Time'].values[0] / largest_data['Multichain_Unroll16_Time'].values[0]
]

colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
bar_plot = plt.bar(methods, speedups, color=colors)

# Add values on top of bars
for bar, speedup in zip(bar_plot, speedups):
    plt.text(bar.get_x() + bar.get_width()/2, 
             speedup + 0.1, 
             f'{speedup:.2f}', 
             ha='center', va='bottom')

plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
plt.ylabel('Speedup vs Common Sum')
plt.title(f'Performance Comparison for Largest Array Size ({largest_size})')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("./output/largest_array_speedup.pdf", format = "pdf", bbox_inches="tight")
plt.close()