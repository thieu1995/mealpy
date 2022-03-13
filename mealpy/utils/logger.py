#!/usr/bin/env python
# Created by "Thieu" at 14:57, 12/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import logging


class Logger:

    def __init__(self, **kwargs):
        self.default_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s: %(message)s', datefmt="%Y/%m/%d %I:%M:%S %p")
        self.default_filename = "mealpy.log"

    def create_file_logger(self, name=__name__, level=logging.DEBUG, format_str=None, filename=None):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if format_str is None:
            formatter = logging.Formatter('%(asctime)s - %(levelname)-7s - %(name)s: %(message)s', datefmt="%Y/%m/%d %I:%M:%S %p")
        else:
            formatter = logging.Formatter(format_str, datefmt="%Y/%m/%d %I:%M:%S %p")
        if filename is None:
            filename = self.default_filename
        handler = logging.FileHandler(filename)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def create_console_logger(self, name=__name__, level=logging.INFO, format_str=None):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if format_str is None:
            formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(name)s: %(message)s', datefmt="%Y/%m/%d %I:%M:%S %p")
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
