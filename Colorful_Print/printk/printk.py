from termcolor import colored, cprint


def print_colored_box_line(title, message, attrs=['bold'], text_color='white', box_color='yellow',box_width=80):
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
    


def print_colored_box(text, text_background=False, text_color='white', box_color='green', background_color='on_white'):
    # 测量文本长度，并为方框的左右添加空格
    padded_text = " " + text + " "
    text_length = len(padded_text)
    
    # 生成上下边框
    top_bottom_border = '+' + '-' * text_length + '+'
    
    # 为边框添加颜色，并使其加粗
    colored_top_bottom = colored(top_bottom_border, box_color, attrs=['bold'])
    
    # 生成中间文本行，包括左右边框，文本颜色和加粗
    # 注意：由于colored函数不支持直接在文本两侧添加颜色不同的字符，我们需要分开处理
    if text_background == True:
        if background_color is not None:
            middle_text = colored(padded_text, text_color, on_color=background_color, attrs=['bold'])
        else:
            middle_text = colored(padded_text, text_color, on_color='on_' + box_color, attrs=['bold'])
    else: 
        middle_text = colored(padded_text, text_color, attrs=['bold'])
    left_border = colored("|", box_color, attrs=['bold'])
    right_border = colored("|", box_color, attrs=['bold'])
    
    # 打印彩色方框
    print(colored_top_bottom)
    print(left_border + middle_text + right_border)
    print(colored_top_bottom)
