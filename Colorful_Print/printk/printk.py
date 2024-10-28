from termcolor import colored, cprint
import unicodedata

def get_display_width(text):
    """
    计算字符串在终端中的显示宽度。
    全角字符计为宽度2，半角字符计为宽度1。
    """
    width = 0
    for char in text:
        if unicodedata.east_asian_width(char) in ('F', 'W', 'A'):
            width += 2
        else:
            width += 1
    return width

def print_colored_box_line(title, message, attrs=['bold'], text_color='white', box_color='yellow',box_width=80):
    # 详细描述代码的参数和功能
    """
    打印带颜色的文本框，自动调整框的宽度以适应文本长度。
    
    参数:
    - text: 要打印的文本，可以是字符串或字符串列表。
    - pad_len: 用户指定的输出宽度。
    - text_color: 文本颜色。
    - box_color: 边框颜色。
    - background_color: 文本背景颜色。
    - attrs: 文本样式属性列表。
    - text_background: 是否为文本添加背景颜色。

    print("使用说明：")
    print("print_colored_box_line(title, message, attrs=['bold'], text_color='white', box_color='yellow',box_width=80)")
    print("title: 方框的标题")
    print("message: 方框的消息")
    print("attrs: 方框的属性，默认为['bold']")
    print("text_color: 文本颜色，默认为'white'")
    print("box_color: 方框颜色，默认为'yellow'")
    print("box_width: 方框宽度，默认为80")
    print("\n")
    print("示例：")
    print("print_colored_box_line('Title', 'Hello, World!', attrs=['bold'], text_color='white', box_color='yellow',box_width=80)")
    print("print_colored_box_line('Title', 'Hello, World!', attrs=['bold'], text_color='white', box_color='green',box_width=80)")
    print("print_colored_box_line('Title', 'Hello, World!', attrs=['bold'], text_color='white', box_color='red',box_width=80)")
    print("print_colored_box_line('Title', 'Hello, World!', attrs=['bold'], text_color='white', box_color='blue',box_width=80)")
    print("print_colored_box_line('Title', 'Hello, World!', attrs=['bold'], text_color='white', box_color='magenta',box_width=80)")
    print("print_colored_box_line('Title', 'Hello, World!', attrs=['bold'], text_color='white', box_color='cyan',box_width=80)")
    print("print_colored_box_line('Title', 'Hello, World!', attrs=['bold'], text_color='white', box_color='grey',box_width=80)")
    print("print_colored_box_line('Title', 'Hello, World!', attrs=['bold'], text_color='white', box_color='black',box_width=80)")
    print("print_colored_box_line('Title', 'Hello, World!', attrs=['bold'], text_color='white', box_color='white',box_width=80)")
    print("\n")
    打印彩色方框
    """
    # 定义方框的宽度为终端的宽度，这里假定为80字符宽
    box_width = 80
    
    # 创建顶部和底部的边框
    horizontal_border = '+' + '-' * (box_width - 2) + '+'
    colored_horizontal_border = colored(horizontal_border, box_color, attrs=attrs)
    
    # 创建标题和消息文本，使其居中
    title_text = f"| {title.center(box_width - 4)} |"
    message_text = f"| {message.center(box_width - 4)} |"
    
    # 添加颜色到文本，并使其加粗
    colored_title = colored(title_text, text_color, 'on_' + box_color, attrs=attrs)
    colored_message = colored(message_text, text_color, 'on_' + box_color, attrs=attrs)
    
    # 打印方框
    print(colored_horizontal_border)
    print(colored_title)
    print(colored_horizontal_border)
    print(colored_message)
    print(colored_horizontal_border)
    


def print_colored_box(text, bbox_width=40, text_color='white', box_color='green', background_color='on_white', attrs=['bold'], text_background=False, align='left'):
    # 添加代码使用说明，以及一些示例
    """
    - text: 要打印的文本，可以是字符串或字符串列表。
    - pad_len: 用户指定的输出宽度。
    - pad_len: 用户指定的输出宽度。
    - text_color: 文本颜色。
    - box_color: 边框颜色。
    - background_color: 文本背景颜色。
    - attrs: 文本样式属性列表。
    - text_background: 是否为文本添加背景颜色。
    - align: 文本对齐方式，可以为'left'（默认）、'right'或'center'

    print("使用说明：")
    print("print_colored_box(text, text_background=False, text_color='white', box_color='green', background_color='on_white')")
    print("text: 要打印的文本")
    print("text_background: 是否为文本添加背景色，默认为False")
    print("text_color: 文本颜色，默认为'white'")
    print("box_color: 方框颜色，默认为'green'")
    print("background_color: 文本背景色，默认为'on_white'")
    print("\n")
    print("示例：")
    print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_white')")
    print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_red')")
    print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_yellow')")
    print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_blue')")
    print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_magenta')")
    print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_cyan')")
    print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_grey')")
    print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_black')")
    print("print_colored_box('Hello, World!', text_background=True, text_color='white', box_color='green', background_color='on_white')")
    print("\n")
    打印彩色方框
    """
    
    if isinstance(text, list):
        content_width = max(get_display_width(item) for item in text)
    else:
        content_width = get_display_width(text)

    # 确定总宽度，考虑到边框的宽度(+2)
    total_width = max(bbox_width, content_width + 4)

    # 生成顶部和底部的边框
    top_bottom_border = '+' + '-' * (total_width - 2) + '+'
    print(colored(top_bottom_border, box_color, attrs=attrs))

    if isinstance(text, list):
        for item in text:
            # 确保文本左侧有1个空格，右侧填充至总宽度减去边框宽度和左侧空格
            # line = f" {item} ".ljust(total_width - 2)
            space_padding = total_width - 2 - get_display_width(item) - 2  # 减去边框和文本两侧的空格
            # line = f" {item} " + " " * space_padding
            if align == 'left':
                line = f" {item} " + " " * space_padding
            elif align == 'right':
                line = " " * space_padding + f" {item} "
            elif align == 'center':
                left_padding = space_padding // 2
                right_padding = space_padding - left_padding
                line = " " * left_padding + f" {item} " + " " * right_padding
            
            print(colored("|", box_color, attrs=attrs) + colored(line, text_color, attrs=attrs, on_color=background_color if text_background else None) + colored("|", box_color, attrs=attrs))
    else:
        # 对于单个文本，处理方式相同
        # line = f" {text} ".ljust(total_width - 2)
        space_padding = total_width - 2 - get_display_width(text) - 2
        # line = f" {text} " + " " * space_padding
        if align == 'left':
            line = f" {text} " + " " * space_padding
        elif align == 'right':
            line = " " * space_padding + f" {text} "
        elif align == 'center':
            left_padding = space_padding // 2
            right_padding = space_padding - left_padding
            line = " " * left_padding + f" {text} " + " " * right_padding
        print(colored("|", box_color, attrs=attrs) + colored(line, text_color, attrs=attrs, on_color=background_color if text_background else None) + colored("|", box_color, attrs=attrs))

    print(colored(top_bottom_border, box_color, attrs=attrs))
