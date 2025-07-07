import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
data = pd.read_csv("./matrix_performance_no_opt.csv")

# 计算平均每次时间
data["CommonTimePerRun"] = data["Common_Time"] / data["Repeat_Times"]
data["CacheTimePerRun"] = data["Cache_Optimized_Time"] / data["Repeat_Times"]

# 计算加速比
data["Speedup"] = data["Common_Time"] / data["Cache_Optimized_Time"]

# 图 1: 平均每次执行时间
plt.figure(figsize=(10, 6))
plt.plot(
    data["Matrix_Size"], data["CommonTimePerRun"], label="Common Algorithm", marker="o"
)
plt.plot(
    data["Matrix_Size"], data["CacheTimePerRun"], label="Cache Optimized", marker="o"
)
plt.xlabel("Matrix Size")
plt.ylabel("Time per Run (seconds)")
plt.title("Average Time per Run vs Matrix Size")
plt.legend()
plt.grid(True)
# plt.savefig("output/time_per_run.png")
plt.savefig("output/time_per_run.pdf", format="pdf", bbox_inches="tight")
plt.show()

# 图 2: 加速比
plt.figure(figsize=(10, 6))
plt.plot(
    data["Matrix_Size"],
    data["Speedup"],
    label="Speedup (Cache/Common)",
    marker="o",
    color="green",
)
plt.xlabel("Matrix Size")
plt.ylabel("Speedup Ratio")
plt.title("Speedup Ratio of Cache Optimized vs Common Algorithm")
plt.legend()
plt.grid(True)
# plt.savefig("output/speedup_ratio.png")
plt.savefig("output/speedup_ratio.pdf", format="pdf", bbox_inches="tight")

plt.show()
