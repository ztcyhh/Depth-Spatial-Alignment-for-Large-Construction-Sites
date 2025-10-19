"""
这个py文件表示从GPS(WGS84,也叫lla,longitude-latitude-altitude)或国家大地坐标系(CGCS)转为局部坐标系ENU
'epsg:4490'就是表示国家大地坐标系(CGCS)2000
参考 笔记本电脑的 D:\learning\project\MVS\mvs_data_proc\gps_ecef2enu.py
"""
import numpy as np
from pyproj import Transformer
import pandas as pd

