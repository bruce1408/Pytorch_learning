'''
version: 1.0.0
Author: BruceCui
Date: 2024-05-12 23:15:38
LastEditors: BruceCui
LastEditTime: 2024-10-28 14:32:46
'''

import os
import sys


# 获取项目的根目录并添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from printk.printk import print_colored_box_line, print_colored_box

print_colored_box("hello world", 60, text_color='green', box_color='yellow', align='center')
print_colored_box("请在此脚本目录运行该脚本", align='center')
