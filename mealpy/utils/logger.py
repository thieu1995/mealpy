#!/usr/bin/env python
# Created by "Thieu" at 14:57, 12/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import logging


class Logger:

    def __init__(self, log_to="console", **kwargs):
        self.log_to = log_to
        self.log_file = None
        self.__set_keyword_arguments(kwargs)
        self.default_formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(name)s: %(message)s', datefmt="%Y/%m/%d %I:%M:%S %p")
        self.default_logfile = "mealpy.log"

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def create_logger(self, name=__name__, format_str=None):
        logger = logging.getLogger(name)
        if self.log_to == "console":
            logger.setLevel(logging.INFO)
            if format_str is None:
                formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(name)s: %(message)s', datefmt="%Y/%m/%d %I:%M:%S %p")
            else:
                formatter = logging.Formatter(format_str, datefmt="%Y/%m/%d %I:%M:%S %p")
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
        elif self.log_to == "file":
            logger.setLevel(logging.DEBUG)
            if format_str is None:
                formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(name)s: %(message)s', datefmt="%Y/%m/%d %I:%M:%S %p")
            else:
                formatter = logging.Formatter(format_str, datefmt="%Y/%m/%d %I:%M:%S %p")
            if self.log_file is None:
                self.log_file = self.default_logfile
            handler = logging.FileHandler(self.log_file)
            handler.setFormatter(formatter)
        else:
            logger.setLevel(logging.ERROR)
            if format_str is None:
                formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s', datefmt="%Y/%m/%d %I:%M:%S %p")
            else:
                formatter = logging.Formatter(format_str, datefmt="%Y/%m/%d %I:%M:%S %p")
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


# class Logger:
#
#     def __init__(self, **kwargs):
#         self.default_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s: %(message)s', datefmt="%Y/%m/%d %I:%M:%S %p")
#         self.default_filename = "mealpy.log"
#
#     def create_file_logger(self, name=__name__, level=logging.DEBUG, format_str=None, filename=None):
#         logger = logging.getLogger(name)
#         logger.setLevel(level)
#         if format_str is None:
#             formatter = logging.Formatter('%(asctime)s - %(levelname)-7s - %(name)s: %(message)s', datefmt="%Y/%m/%d %I:%M:%S %p")
#         else:
#             formatter = logging.Formatter(format_str, datefmt="%Y/%m/%d %I:%M:%S %p")
#         if filename is None:
#             filename = self.default_filename
#         handler = logging.FileHandler(filename)
#         handler.setFormatter(formatter)
#         logger.addHandler(handler)
#         return logger
#
#     def create_console_logger(self, name=__name__, level=logging.INFO, format_str=None):
#         logger = logging.getLogger(name)
#         logger.setLevel(level)
#         if format_str is None:
#             formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(name)s: %(message)s', datefmt="%Y/%m/%d %I:%M:%S %p")
#         else:
#             formatter = logging.Formatter(format_str, datefmt="%Y/%m/%d %I:%M:%S %p")
#         handler = logging.StreamHandler()
#         handler.setFormatter(formatter)
#         logger.addHandler(handler)
#         return logger
