from termcolor import cprint


def print_info_custom(info, _type=None):
    """_summary_
    usage:
        print_info('=> Total {} images to test.'.format(img_txt_evm_res), ['yellow', 'bold'])
        cprint('Bold underline reverse cyan color', 'cyan', attrs=['bold', 'underline', 'reverse'])
        cprint('Dark blink concealed white color', 'white', attrs=['dark', 'blink', 'concealed'])
    Args:
        info (_type_): _description_
        _type (_type_, optional): _description_. Defaults to None.
    reference:
        https://blog.csdn.net/weixin_44751294/article/details/122435752
    """
    
    if _type is not None:
        if isinstance(info, str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info, list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)
        

def print_info(info):
    if isinstance(info, str):
        cprint("[INFO]   {:<15}".format(info), 'green')

    elif isinstance(info, list):        
        for i in range(info):
            cprint("{}".format(i), "green", attrs=['bold'])

def print_error(info):
    if isinstance(info, str):
        cprint("[ERROR]  {:<15}".format(info), "red", attrs=['bold'])
    elif isinstance(info, list):
        for i in range(info):
            cprint(i, "red", attrs=['bold'])
    
    
def print_log(info):
    if isinstance(info, str):
        cprint("[LOG]    {:<15}".format(info), 'yellow') 
    elif isinstance(info, list):
        for i in range(info):
            cprint(i, "yellow")


# from termcolor import cprint

# def print_aligned_message(message, label='INFO', width=10, color='blue'):
#     """
#     打印带标签的左对齐信息。

#     :param message: 要打印的信息。
#     :param label: 信息前的标签，默认为 'INFO'。
#     :param width: 对齐的宽度，默认为10个字符。
#     :param color: 打印的颜色，默认为蓝色。
#     """
#     # 构建带标签的信息字符串
#     full_message = f"{label}: {message}"

#     # 使用格式化字符串来左对齐文本，使其宽度为指定的字符数
#     formatted_message = "{:<{}}".format(full_message, width)

#     # 使用 cprint 打印格式化后的信息
#     cprint(formatted_message, color)

# 使用函数打印信息
# print_aligned_message("测试信息")
# print_aligned_message("警告信息", label='WARNING', color='red')
# print_aligned_message("详细日志信息", label='LOG', width=20)
# print_aligned_message("详细日志信息", label='ERROR', width=20)
# print_aligned_message("详细日志信息", label='INFO', width=20)




if __name__ == "__main__":
    img_path = "/Users/bruce/PycharmProjects/Pytorch_learning/Tools/val_imagenet_label.txt"
    # print_info_custom("the path is {} ".format(img_path), ['green', 'blink'])
    print_info("the path is {} ".format(img_path))
    print_error("error the info")
    print_log("print the info")
    # cprint("{:<{}}".format("error the log", 10))

   