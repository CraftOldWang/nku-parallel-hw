import os
import re # 导入正则表达式模块

def remove_ifdef_blocks_general(file_path, macro_name="DEBUG"):
    """
    移除 C++ 文件中 #ifdef <macro_name> 和 #endif 之间的代码块。
    能处理 #ifdef 和宏名称之间不定数量的空格。

    Args:
        file_path (str): 要处理的 .cpp 文件路径。
        macro_name (str): 要移除的条件编译宏的名称（例如："DEBUG", "RELEASE_MODE"）。
    """
    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在。")
        return

    output_lines = []
    in_target_block = False

    # 构建正则表达式，匹配 #ifdef 后面跟着一个或多个空格，然后是宏名称
    # \s+ 匹配一个或多个空白字符 (空格, 制表符等)
    ifdef_pattern = re.compile(r"^#ifdef\s+" + re.escape(macro_name) + r"$")
    endif_pattern = re.compile(r"^#endif$") # #endif 通常没有参数，但可以更严格

    with open(file_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            stripped_line = line.strip()

            # 使用正则表达式匹配 #ifdef <macro_name>
            if ifdef_pattern.match(stripped_line):
                in_target_block = True
                continue # 不写入 #ifdef 这一行
            # 匹配 #endif
            elif endif_pattern.match(stripped_line) and in_target_block:
                in_target_block = False
                continue # 不写入 #endif 这一行

            if not in_target_block:
                output_lines.append(line)

    # 生成新的文件名 (例如：your_file.cpp -> your_file_no_debug.cpp)
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    output_file_path = os.path.join(dir_name, f"{name}{ext}") # 文件名也包含宏名称

    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.writelines(output_lines)

    print(f"已处理文件：'{file_path}'")
    print(f"移除 '#ifdef {macro_name}' 块后的内容已保存到：'{output_file_path}'")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法：python your_script_name.py <你的_cpp_文件路径> [宏名称，默认为DEBUG]")
    else:
        cpp_file = sys.argv[1]
        # 如果提供了第二个参数，则用作宏名称，否则默认为 "DEBUG"
        macro = sys.argv[2] if len(sys.argv) > 2 else "DEBUG"
        remove_ifdef_blocks_general(cpp_file, macro)