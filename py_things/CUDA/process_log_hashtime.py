import os
import glob

def process_log(input_file, output_file):
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        print(f"Warning: Could not decode {input_file} with UTF-8. Skipping.")
        return
    except FileNotFoundError:
        print(f"Error: Input file {input_file} does not exist. Skipping.")
        return

    # 初始化输出内容
    output_lines = []
    in_experiment = False  # 是否在实验部分
    expect_label = False  # 是否期待实验标号行

    for i, line in enumerate(lines):
        # 保留前三行
        if i < 3:
            output_lines.append(line)
            continue

        # 检测实验开始分隔线
        if line.startswith('=========================================='):
            if not in_experiment:
                # 进入实验部分，期待标号行
                in_experiment = True
                expect_label = True
                output_lines.append(line)
            else:
                # 实验结束分隔线
                in_experiment = False
                expect_label = False
            continue

        # 检测实验标号行（如“实验 #X”）
        if expect_label and '实验 #' in line:
            output_lines.append(line)
            expect_label = False
            continue

        # 在实验部分，处理“猜测上限”和“批处理大小”行
        if in_experiment and '猜测上限' in line:
            # 分割“猜测上限”和“批处理大小”
            parts = line.split(',')
            if len(parts) == 2:
                output_lines.append(parts[0].strip() + '\n')  # 猜测上限
                output_lines.append(parts[1].strip() + '\n')  # 批处理大小
            continue

        # 检测“Hash time”行，保留
        if 'Hash time:' in line:
            output_lines.append(line)
            continue

    # 写入输出文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
        print(f"Processed {input_file} -> {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

# 查找 ../ 目录下的 .o 和 .txt 文件
input_dir = '../'
output_dir = '.'  # 当前目录
file_patterns = ['*.o', '*.txt']

for pattern in file_patterns:
    for input_file in glob.glob(os.path.join(input_dir, pattern)):
        # 生成输出文件名（同名但以 .ed 结尾）
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.ed")
        process_log(input_file, output_file)