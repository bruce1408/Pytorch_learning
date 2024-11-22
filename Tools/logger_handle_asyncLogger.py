import logging
import os
import sys
import time
import queue
from colorama import Fore, Style, init
from termcolor import colored
from logging.handlers import QueueHandler, QueueListener
import atexit
import datetime

init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    LEVEL_COLOR_MAPPING = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA
    }

    # 全变色
    def format(self, record):
        log_message = super().format(record)
        level_color = self.LEVEL_COLOR_MAPPING.get(record.levelno, Fore.WHITE)
        return f"{level_color}{log_message}{Style.RESET_ALL}"
    
    # 单独变色
    # def format(self, record):
    #     prefix = f"[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s " % record.__dict__
    #     level_color = self.LEVEL_COLOR_MAPPING.get(record.levelno, '')
    #     if level_color:
    #         colored_prefix = f"{level_color}{prefix}{Style.RESET_ALL}"
    #     else:
    #         colored_prefix = prefix
    #     return f"{colored_prefix}{record.msg}"
    

        
class AsyncLoggerManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AsyncLoggerManager, cls).__new__(cls)
            cls._instance.init_logger(*args, **kwargs)
        return cls._instance

    def init_logger(self, name="QnnHelperLogger", work_dir=None, log_file=None, level=logging.INFO):
        if not hasattr(self, "initialized"):
            self.logger = logging.getLogger(name)
            self.logger.setLevel(level)
            self.log_queue = queue.Queue(-1)

            if log_file is None:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_file = f"qnnhelper_{timestamp}.log"
                if work_dir is not None:
                    log_file = os.path.join(work_dir, log_file)

            # 定义日志格式
            fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'

            # 创建文件处理器，将日志写入文件
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))

            # 创建控制台处理器，将日志输出到控制台，使用彩色格式
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(ColoredFormatter(fmt, datefmt='%Y-%m-%d %H:%M:%S'))

            # 创建日志队列和队列处理器
            queue_handler = QueueHandler(self.log_queue)
            self.logger.addHandler(queue_handler)

            # 创建队列监听器，监听日志队列并将日志发送到文件和控制台
            self.listener = QueueListener(self.log_queue, file_handler, console_handler)
            self.listener.start()
            self.initialized = True

            # 确保在程序退出时停止队列监听器
            atexit.register(self.stop_listener)

    def stop_listener(self):
        if self.listener:
            self.listener.stop()
            self.listener = None

if __name__ == "__main__":
    # 指定日志目录
    log_directory = "./logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    # 获取异步日志记录器实例
    async_logger_manager = AsyncLoggerManager(work_dir=log_directory)
    logger = async_logger_manager.logger
    
    # 记录一些日志消息
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
