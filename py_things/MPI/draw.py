import matplotlib.pyplot as plt
import re


def extract_guess_time(file_path):
    """从文件中提取 Guess time 数值"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 使用正则表达式匹配 "Guess time:数字seconds"
        match = re.search(r"Guess time:(\d+\.?\d*)seconds", content)
        if match:
            return float(match.group(1))
        else:
            print(f"在文件 {file_path} 中未找到 Guess time")
            return None
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None


def create_guess_time_chart():
    """创建 Guess time 对比柱状图"""

    # 文件路径
    normal_file = "normal-O2.o"
    mpi_file = "MPI-O2.o"

    # 提取数据
    normal_time = extract_guess_time(normal_file)
    mpi_time = extract_guess_time(mpi_file)

    if normal_time is None or mpi_time is None:
        print("无法提取完整数据，请检查文件")
        return

    # 准备数据
    methods = ["Normal (Sequential)", "MPI (8 processes)"]
    times = [normal_time, mpi_time]

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, times, color=["#3498db", "#e74c3c"], alpha=0.8)

    # 添加数值标签
    for bar, time in zip(bars, times):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{time:.5f}s",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # 设置图表样式

    plt.ylabel("Time (seconds)", fontsize=12)
    plt.xlabel("Implementation Method", fontsize=12)

    # 添加网格
    plt.grid(axis="y", alpha=0.3)

    # 计算加速比
    if normal_time > 0:
        speedup = normal_time / mpi_time
        plt.text(
            0.5,
            max(times) * 0.8,
            f"Speedup: {speedup:.2f}x",
            transform=plt.gca().transAxes,
            ha="center",
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    # 调整布局
    plt.tight_layout()

    # 保存图表
    # plt.savefig("guess_time_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig("guess_time_comparison.pdf", bbox_inches="tight")

    # 显示图表
    plt.show()

    # 输出数据摘要
    print(f"\n=== Guess Time 对比结果 ===")
    print(f"Normal: {normal_time:.5f} seconds")
    print(f"MPI (8 processes): {mpi_time:.5f} seconds")
    if normal_time > 0:
        speedup = speedup if "speedup" in locals() else normal_time / mpi_time
        print(f"加速比: {speedup:.2f}x")
        if speedup < 1:
            print("注意: MPI版本比串行版本慢，可能存在问题")
    print(f"图表已保存为: guess_time_comparison.png 和 guess_time_comparison.pdf")


if __name__ == "__main__":
    create_guess_time_chart()
