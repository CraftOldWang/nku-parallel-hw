import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_matplotlib():
    """配置 matplotlib 支持中文"""
    # 查找系统中的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'Arial Unicode MS']
    
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    else:
        print("Warning: No Chinese font found. Chinese characters may not display correctly.")
        print("Consider installing a Chinese font like SimHei or Microsoft YaHei.")

    # 设置 PDF 输出字体类型
    plt.rcParams['pdf.fonttype'] = 42  # 嵌入字体，确保 PDF 显示一致

def extract_data(file_path):
    """从 .ed 文件中提取猜测上限和 Hash time"""
    data = []
    current_guess_limit = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('猜测上限:'):
                    current_guess_limit = int(line.split(':')[1].strip())
                elif line.startswith('Hash time:'):
                    hash_time = float(line.split(':')[1].split('seconds')[0].strip())
                    if current_guess_limit is not None:
                        data.append((current_guess_limit, hash_time))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data

def plot_hash_time(all_data):
    """绘制 Hash time vs 猜测上限"""
    plt.figure(figsize=(10, 6))
    for file_name, data in all_data.items():
        if data:
            guess_limits = [x[0] for x in data]
            hash_times = [x[1] for x in data]
            plt.plot(guess_limits, hash_times, marker='o', label=file_name)
    
    plt.xlabel('猜测上限 (Guess Limit)')
    plt.ylabel('Hash Time (秒)')
    plt.legend()
    plt.grid(True)
    plt.savefig('hash_time_plot.pdf', bbox_inches='tight')
    plt.close()

def plot_speedup(all_data, baseline_file='scalar-O0'):
    """绘制相对于 scalar-O0 的加速比"""
    if baseline_file not in all_data:
        print(f"Error: Baseline file {baseline_file}.ed not found")
        return
    
    baseline_data = all_data[baseline_file]
    baseline_dict = {x[0]: x[1] for x in baseline_data}  # 猜测上限 -> Hash time

    plt.figure(figsize=(10, 6))
    for file_name, data in all_data.items():
        if file_name == baseline_file:
            continue  # 跳过基准文件
        if data:
            guess_limits = []
            speedups = []
            for guess_limit, hash_time in data:
                if guess_limit in baseline_dict and hash_time > 0:
                    speedup = baseline_dict[guess_limit] / hash_time
                    guess_limits.append(guess_limit)
                    speedups.append(speedup)
            if guess_limits:
                plt.plot(guess_limits, speedups, marker='o', label=file_name)
    
    plt.xlabel('猜测上限 (Guess Limit)')
    plt.ylabel(f'加速比 (相对于 {baseline_file})')
    # 修改图例位置：固定在左上角
    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.5))
    plt.grid(True)
    plt.savefig('speedup_plot.pdf', bbox_inches='tight')
    plt.close()

def main():
    """主函数：处理当前目录下所有 .ed 文件并绘图"""
    # 配置 matplotlib 支持中文
    setup_matplotlib()

    # 查找当前目录下所有 .ed 文件
    ed_files = glob.glob('*.ed')
    if not ed_files:
        print("Error: No .ed files found in current directory")
        return

    # 提取所有文件的数据
    all_data = {}
    for ed_file in ed_files:
        file_name = os.path.splitext(os.path.basename(ed_file))[0]  # 去掉 .ed
        data = extract_data(ed_file)
        if data:
            all_data[file_name] = sorted(data)  # 按猜测上限排序

    if not all_data:
        print("Error: No valid data extracted from .ed files")
        return

    # 绘制 Hash time 图
    plot_hash_time(all_data)

    # 绘制加速比图（相对于 scalar-O0）
    plot_speedup(all_data, baseline_file='scalar_win-O2')

if __name__ == "__main__":
    main()