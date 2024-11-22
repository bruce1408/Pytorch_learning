import logging
import os
import sys
import time
from termcolor import colored

# 假设这是你的 setup_logger 函数
logger = logging.getLogger("dipoorlet")

def setup_logger(args):
    global logger
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'
    logger.setLevel(logging.INFO)
    logger_file = os.path.join(args.output_dir, 'log-{}.txt'.format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))
    with open(logger_file, 'w') as f:
        f.write(str(args) + '\n')
    file_handler = logging.FileHandler(logger_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)


# 示例程序调用 setup_logger
if __name__ == "__main__":
    class Args:
        output_dir = "./logs"
    
    args = Args()

    # 确保日志目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 设置日志
    setup_logger(args)

    # 记录一些日志消息
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
