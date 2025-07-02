import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置输入和输出文件夹
INPUT_DIR = "./processed"  # 输入文件夹：包含.ed文件的目录
OUTPUT_DIR = "./output"  # 输出文件夹：生成图片的目录
BASELINE_FILE = "normal"


def setup_matplotlib():
    """配置 matplotlib 支持中文"""
    # 查找系统中的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]

    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break

    if selected_font:
        plt.rcParams["font.sans-serif"] = [selected_font] + plt.rcParams[
            "font.sans-serif"
        ]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    else:
        print(
            "Warning: No Chinese font found. Chinese characters may not display correctly."
        )
        print("Consider installing a Chinese font like SimHei or Microsoft YaHei.")

    # 设置 PDF 输出字体类型
    plt.rcParams["pdf.fonttype"] = 42  # 嵌入字体，确保 PDF 显示一致


def extract_data(file_path):
    """从 .ed 文件中提取猜测上限和 Guess time"""
    data = []
    current_guess_limit = None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("猜测上限:"):
                    current_guess_limit = int(line.split(":")[1].strip())
                elif line.startswith("Guess time:"):
                    guess_time = float(line.split(":")[1].split("seconds")[0].strip())
                    if current_guess_limit is not None:
                        data.append((current_guess_limit, guess_time))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data


def plot_guess_time(all_data, output_dir):
    """绘制 Guess time vs 猜测上限"""
    plt.figure(figsize=(12, 6))  # 增加宽度为图例留空间
    for file_name, data in all_data.items():
        if data:
            guess_limits = [x[0] for x in data]
            guess_times = [x[1] for x in data]
            plt.plot(guess_limits, guess_times, marker="o", label=file_name)

    plt.xlabel("猜测上限 (Guess Limit)")
    plt.ylabel("Guess Time (秒)")
    # 将图例放在图的右侧外面
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    output_path = os.path.join(output_dir, "guess_time_plot.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Guess time plot saved to: {output_path}")


def plot_speedup(all_data, baseline_file="scalar-O0", output_dir="."):
    """绘制相对于 scalar-O0 的加速比"""
    if baseline_file not in all_data:
        print(f"Error: Baseline file {baseline_file}.ed not found")
        return

    baseline_data = all_data[baseline_file]
    baseline_dict = {x[0]: x[1] for x in baseline_data}  # 猜测上限 -> Guess time

    plt.figure(figsize=(12, 6))  # 增加宽度为图例留空间
    for file_name, data in all_data.items():
        if file_name == baseline_file:
            continue  # 跳过基准文件
        if data:
            guess_limits = []
            speedups = []
            for guess_limit, guess_time in data:
                if guess_limit in baseline_dict and guess_time > 0:
                    speedup = baseline_dict[guess_limit] / guess_time
                    guess_limits.append(guess_limit)
                    speedups.append(speedup)
            if guess_limits:
                plt.plot(guess_limits, speedups, marker="o", label=file_name)

    plt.xlabel("猜测上限 (Guess Limit)")
    plt.ylabel(f"加速比 (相对于 {baseline_file})")
    # 将图例放在图的右侧外面
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    output_path = os.path.join(output_dir, "speedup_plot.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Speedup plot saved to: {output_path}")


def main():
    """主函数：处理指定目录下所有 .ed 文件并绘图"""
    # 配置 matplotlib 支持中文
    setup_matplotlib()

    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 查找输入目录下所有 .ed 文件
    ed_pattern = os.path.join(INPUT_DIR, "*.ed")
    ed_files = glob.glob(ed_pattern)
    if not ed_files:
        print(f"Error: No .ed files found in directory: {INPUT_DIR}")
        print(f"Looking for pattern: {ed_pattern}")
        return

    print(f"Found {len(ed_files)} .ed files in {INPUT_DIR}")

    # 提取所有文件的数据
    all_data = {}
    for ed_file in ed_files:
        file_name = os.path.splitext(os.path.basename(ed_file))[0]  # 去掉 .ed
        data = extract_data(ed_file)
        if data:
            all_data[file_name] = sorted(data)  # 按猜测上限排序
            print(f"Loaded data from {file_name}: {len(data)} data points")

    if not all_data:
        print("Error: No valid data extracted from .ed files")
        return

    # 绘制 Guess time 图
    plot_guess_time(all_data, OUTPUT_DIR)

    # 绘制加速比图（相对于 normal-output）
    plot_speedup(all_data, baseline_file=BASELINE_FILE, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
