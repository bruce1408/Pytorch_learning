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
from printk.printk import print_colored_box_line, print_colored_box

import unicodedata

# def get_display_width(text):
#     """
#     计算字符串在终端中的显示宽度。
#     全角字符计为宽度2，半角字符计为宽度1。
#     """
#     width = 0
#     for char in text:
#         if unicodedata.east_asian_width(char) in ('F', 'W', 'A'):
#             width += 2
#         else:
#             width += 1
#     return width

# def print_colored_box(text, pad_len=40, text_color='white', box_color='green', background_color='on_white', attrs=['bold'], text_background=False, align="left"):
#     """
#     打印带颜色的文本框，自动调整框的宽度以适应文本长度。
    
#     参数:
#     - text: 要打印的文本，可以是字符串或字符串列表。
#     - pad_len: 用户指定的输出宽度。
#     - pad_len: 用户指定的输出宽度。
#     - text_color: 文本颜色。
#     - box_color: 边框颜色。
#     - background_color: 文本背景颜色。
#     - attrs: 文本样式属性列表。
#     - text_background: 是否为文本添加背景颜色。
#     - align: 文本对齐方式，可以为'left'（默认）、'right'或'center'

#     """
#     if attrs is None:
#         attrs = ['bold']
#     # 对于文本列表，找出最长的文本长度
#     # if isinstance(text, list):
#     #     content_width = max(len(item) for item in text) + 2  # 文本两侧各有1个空格
#     # else:
#     #     content_width = len(text) + 2  # 单个文本两侧各有1个空格

    
#     if isinstance(text, list):
#         content_width = max(get_display_width(item) for item in text)
#     else:
#         content_width = get_display_width(text)

#     print(content_width)
#     # 确定总宽度，考虑到边框的宽度(+2)
#     total_width = max(pad_len, content_width + 4)

#     # 生成顶部和底部的边框
#     top_bottom_border = '+' + '-' * (total_width - 2) + '+'
#     print(colored(top_bottom_border, box_color, attrs=attrs))

#     if isinstance(text, list):
#         for item in text:
#             # 确保文本左侧有1个空格，右侧填充至总宽度减去边框宽度和左侧空格
#             # line = f" {item} ".ljust(total_width - 2)
#             space_padding = total_width - 2 - get_display_width(item) - 2  # 减去边框和文本两侧的空格
#             # line = f" {item} " + " " * space_padding
#             if align == 'left':
#                 line = f" {item} " + " " * space_padding
#             elif align == 'right':
#                 line = " " * space_padding + f" {item} "
#             elif align == 'center':
#                 left_padding = space_padding // 2
#                 right_padding = space_padding - left_padding
#                 line = " " * left_padding + f" {item} " + " " * right_padding
            
#             print(colored("|", box_color, attrs=attrs) + colored(line, text_color, attrs=attrs, on_color=background_color if text_background else None) + colored("|", box_color, attrs=attrs))
#     else:
#         # 对于单个文本，处理方式相同
#         # line = f" {text} ".ljust(total_width - 2)
#         space_padding = total_width - 2 - get_display_width(text) - 2
#         # line = f" {text} " + " " * space_padding
#         if align == 'left':
#             line = f" {text} " + " " * space_padding
#         elif align == 'right':
#             line = " " * space_padding + f" {text} "
#         elif align == 'center':
#             left_padding = space_padding // 2
#             right_padding = space_padding - left_padding
#             line = " " * left_padding + f" {text} " + " " * right_padding
#         print(colored("|", box_color, attrs=attrs) + colored(line, text_color, attrs=attrs, on_color=background_color if text_background else None) + colored("|", box_color, attrs=attrs))

#     print(colored(top_bottom_border, box_color, attrs=attrs))


# steps = [
#     "1. run onnx-converter (to generate cpp file)",
#     "2. run model-lib-generator (to generate .so shared lib) ",
#     "3. run net-run (to collect the statistics)",
#     "4. run profile-viewer (to view the profiles)"
# ]

# 调用函数，这次包含attrs参数以控制样式
# print_colored_box(steps, text_color='green', box_color='yellow')
print_colored_box("hello world", 60, text_color='green', box_color='yellow', align='center')
print_colored_box("请在此脚本目录运行该脚本", align='center')
