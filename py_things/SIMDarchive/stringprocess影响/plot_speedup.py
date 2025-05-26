import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files
parallel_df = pd.read_csv('parallel_hash_times.csv')
serial_df = pd.read_csv('serial_md5_hash_times.csv')

# Calculate speedup for the parallel portion
speedup = serial_df['Parallel_Hash_Time_s'] / parallel_df['Parallel_Hash_Time_s']

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(parallel_df['Guess_Limit'], speedup, marker='o', linestyle='-', color='b', label='SIMD-O2')
plt.xlabel('Guess Limit')
plt.ylabel('Speedup')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('speedup_plot.pdf',bbox_inches='tight')