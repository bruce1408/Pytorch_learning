import os
import sys

# 将父目录（Colorful_Print）加入到 sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from printk.printk import print_colored_box, print_colored_box_line
steps = [
    "1. run qnn-onnx-converter (to generate cpp file)",
    "2. run qnn-model-lib-generator (to generate .so shared lib) ",
    "3. run qnn-net-run (to collect the statistics)",
    "4. run qnn-profile-viewer (to view the profiles)"
]

# 调用函数，这次包含attrs参数以控制样式
print_colored_box(steps, text_color='green', box_color='yellow')
print_colored_box("hello world", 60, text_color='green', box_color='yellow')
print_colored_box("请在此脚本目录运行该脚本")
print_colored_box_line("警告", "请立即检查系统！", attrs=['bold'], text_color='red', box_color='yellow')
best_acc = 0.98
print_colored_box(f"Best val Acc: {best_acc}", attrs=['bold'], text_color='red', box_color='yellow')
