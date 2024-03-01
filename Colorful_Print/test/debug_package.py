# # 导入需要的库
# import pytest
# from io import StringIO
# import sys
# # 假设你的函数位于一个名为 colored_box.py 的文件中
# from printk import print_colored_box_line, print_colored_box

# # 测试 print_colored_box_line 函数
# def test_print_colored_box_line():
#     # 捕获输出
#     captured_output = StringIO()
#     sys.stdout = captured_output
#     print_colored_box_line('Test Title', 'Test message')
#     sys.stdout = sys.__stdout__  # 重置输出到正常的标准输出
#     assert 'Test Title' in captured_output.getvalue()
#     assert 'Test message' in captured_output.getvalue()

# # 测试 print_colored_box 函数
# def test_print_colored_box():
#     # 捕获输出
#     captured_output = StringIO()
#     sys.stdout = captured_output
#     print_colored_box('Hello, World!')
#     sys.stdout = sys.__stdout__  # 重置输出到正常的标准输出
#     assert 'Hello, World!' in captured_output.getvalue()

# # 注意：这些测试假设你的函数能够执行而不抛出任何错误，并且输出中包含了特定的文本。
# # 实际的打印结果（包括颜色和格式）在这种测试中是不被直接验证的。
# from ..printk.printk import print_colored_box_line, print_colored_box

from termcolor import *
# def print_colored_box(text, text_background=False, pad_len=80, text_color='white', box_color='green', background_color='on_white',attrs=['bold']):
#     # 添加代码使用说明，以及一些示例
#     """
#     print("使用说明：")
#     print("print_colored_box(text, text_background=False, text_color='white', box_color='green', background_color='on_white')")
#     print("text: 要打印的文本")
#     print("text_background: 是否为文本添加背景色，默认为False")
#     print("text_color: 文本颜色，默认为'white'")
#     print("box_color: 方框颜色，默认为'green'")
#     print("background_color: 文本背景色，默认为'on_white'")
#     print("\n")
#     print("示例：")
#     print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_white')")
#     print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_red')")
#     print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_yellow')")
#     print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_blue')")
#     print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_magenta')")
#     print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_cyan')")
#     print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_grey')")
#     print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_black')")
#     print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_white')")
#     print("\n")
#     打印彩色方框
#     """
#     # if input text is list
#     # 判断输入text是字符串还是list
    
#     if type(text) == list:
#         if attrs is None:
#             attrs = []
#         # 计算最长的步骤字符串长度
#         # max_length = max(len(step) for step in text) + 4  # 加4是为了两边的空格和边框
#         max_length = min(max(len(step) for step in text) + 4, pad_len)  # 加4是为了两边的空格和边框，但不超过pad_len


#         # 生成上下边框
#         top_bottom_border = '+' + '-' * (max_length - 2) + '+'
        

#         # 为边框添加颜色
#         colored_top_bottom = colored(top_bottom_border, box_color, attrs=attrs)

#         # 打印上边框
#         print(colored_top_bottom)
        
#         # 打印每个步骤
#         for step in text:
#             # 根据最长的字符串长度，将每个步骤右侧填充空格以对齐
#             padded_step = step.ljust(max_length - 4)
#             # 为文本和边框添加颜色和样式
#             colored_step = colored("| ", box_color, attrs=attrs) + colored(padded_step, text_color, attrs=attrs) + colored(" |", box_color, attrs=attrs)
#             print(colored_step)
#             print(colored("| ", box_color, attrs=attrs) + colored(" " * (max_length - 4), text_color, attrs=attrs) + colored(" |", box_color, attrs=attrs))
            
#         # 打印下边框
#         print(colored_top_bottom)
#         return
#     # 测量文本长度，并为方框的左右添加空格
#     padded_text = " " + text + " "
#     text_length = len(padded_text)
    
#     # 生成上下边框
#     top_bottom_border = '+' + '-' * text_length + '+'
    
#     # 为边框添加颜色，并使其加粗
#     colored_top_bottom = colored(top_bottom_border, box_color, attrs=['bold'])
    
#     # 生成中间文本行，包括左右边框，文本颜色和加粗
#     # 注意：由于colored函数不支持直接在文本两侧添加颜色不同的字符，我们需要分开处理
#     if text_background == True:
#         if background_color is not None:
#             middle_text = colored(padded_text, text_color, on_color=background_color, attrs=['bold'])
#         else:
#             middle_text = colored(padded_text, text_color, on_color='on_' + box_color, attrs=['bold'])
#     else: 
#         middle_text = colored(padded_text, text_color, attrs=['bold'])
#     left_border = colored("|", box_color, attrs=['bold'])
#     right_border = colored("|", box_color, attrs=['bold'])
    
#     # 打印彩色方框
#     print(colored_top_bottom)
#     print(left_border + middle_text + right_border)
#     print(colored_top_bottom)


from termcolor import colored

def print_colored_box(text, pad_len=40, text_color='white', box_color='green', background_color='on_white', attrs=['bold'], text_background=False):
    """
    打印带颜色的文本框，自动调整框的宽度以适应文本长度。
    
    参数:
    - text: 要打印的文本，可以是字符串或字符串列表。
    - pad_len: 用户指定的输出宽度。
    - pad_len: 用户指定的输出宽度。
    - text_color: 文本颜色。
    - box_color: 边框颜色。
    - background_color: 文本背景颜色。
    - attrs: 文本样式属性列表。
    - text_background: 是否为文本添加背景颜色。
    """
    if attrs is None:
        attrs = ['bold']
    # 对于文本列表，找出最长的文本长度
    if isinstance(text, list):
        content_width = max(len(item) for item in text) + 2  # 文本两侧各有1个空格
    else:
        content_width = len(text) + 2  # 单个文本两侧各有1个空格

    # 确定总宽度，考虑到边框的宽度(+2)
    total_width = max(pad_len, content_width + 2)

    # 生成顶部和底部的边框
    top_bottom_border = '+' + '-' * (total_width - 2) + '+'
    print(colored(top_bottom_border, box_color, attrs=attrs))

    if isinstance(text, list):
        for item in text:
            # 确保文本左侧有1个空格，右侧填充至总宽度减去边框宽度和左侧空格
            line = f" {item} ".ljust(total_width - 2)
            print(colored("|", box_color, attrs=attrs) + colored(line, text_color, attrs=attrs, on_color=background_color if text_background else None) + colored("|", box_color, attrs=attrs))
    else:
        # 对于单个文本，处理方式相同
        line = f" {text} ".ljust(total_width - 2)
        print(colored("|", box_color, attrs=attrs) + colored(line, text_color, attrs=attrs, on_color=background_color if text_background else None) + colored("|", box_color, attrs=attrs))

    print(colored(top_bottom_border, box_color, attrs=attrs))


steps = [
    "1. run qnn-onnx-converter (to generate cpp file)",
    "2. run qnn-model-lib-generator (to generate .so shared lib) ",
    "3. run qnn-net-run (to collect the statistics)",
    "4. run qnn-profile-viewer (to view the profiles)"
]

# 调用函数，这次包含attrs参数以控制样式
print_colored_box(steps, text_color='green', box_color='yellow')
print_colored_box("hello world", 60, text_color='green', box_color='yellow')