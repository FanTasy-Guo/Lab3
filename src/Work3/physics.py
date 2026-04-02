import taichi as ti
import numpy as np
from src.Work3.config import *

# 初始化 Taichi GPU 后端
ti.init(arch=ti.gpu)

# --- GPU 数据字段定义 ---

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_color = ti.Vector.field(3, dtype=ti.f32, shape=())

# GUI 绘制数据缓冲池
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

# 用于存放曲线坐标的 GPU 缓冲区
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)


def de_casteljau(points, t):
    """
    纯 Python 递归实现 De Casteljau 算法
    计算贝塞尔曲线在参数 t 处的点
    """
    if len(points) == 1:
        return points[0]
    
    # 递推计算下一层点
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    
    return de_casteljau(next_points, t)


@ti.kernel
def clear_pixels():
    """GPU 并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector(BACKGROUND_COLOR)


@ti.kernel
def draw_curve_kernel(n: ti.i32):
    """
    GPU 并行绘制曲线
    在显存中将曲线对应像素涂色
    """
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        
        # 边界检查
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = curve_color[None]


def calculate_curve_points(control_points):
    """
    计算贝塞尔曲线上的所有采样点
    """
    if len(control_points) < 2:
        return None
    
    # 在 CPU 端计算所有采样点
    curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
    for t_int in range(NUM_SEGMENTS + 1):
        t = t_int / NUM_SEGMENTS
        curve_points_np[t_int] = de_casteljau(control_points, t)
    
    return curve_points_np


def update_gui_points(control_points):
    """
    更新 GUI 控制点数据
    """
    current_count = len(control_points)
    
    # 更新控制点位置
    if current_count > 0:
        np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
        np_points[:current_count] = np.array(control_points, dtype=np.float32)
        gui_points.from_numpy(np_points)
    
    return current_count


def update_gui_indices(control_count):
    """
    更新 GUI 控制点连线索引
    """
    if control_count >= 2:
        np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
        indices = []
        for i in range(control_count - 1):
            indices.extend([i, i + 1])
        np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
        gui_indices.from_numpy(np_indices)
        
        return indices
    return []