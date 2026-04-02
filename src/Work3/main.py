# ==============================================================
#  main.py  —  贝塞尔 & B 样条曲线交互程序
#
#  操作说明：
#    鼠标左键      —— 添加控制点
#    键盘 c        —— 清空画布
#    键盘 b        —— 在贝塞尔 / B 样条模式之间切换
#    键盘 a        —— 开关反走样（Anti-Aliasing）
# ==============================================================

import taichi as ti
import numpy as np

from src.Work3.config import (
    WIDTH, HEIGHT,
    NUM_SEGMENTS, MAX_CONTROL_POINTS,
    AA_RADIUS, AA_KERNEL_HALF,
    COLOR_CURVE_BEZIER, COLOR_CURVE_BSPLINE,
    COLOR_CURVE_AA_ON, COLOR_CURVE_AA_OFF,
    COLOR_CONTROL_POINT, COLOR_CONTROL_LINE,
)

# ── Taichi 初始化 ──────────────────────────────────────────────
ti.init(arch=ti.gpu)

# ── GPU 缓冲区（预分配，主循环中不再动态申请）─────────────────
pixels            = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)
gui_points        = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices       = ti.field(dtype=ti.i32,           shape=MAX_CONTROL_POINTS * 2)

# ── 曲线颜色（可在运行时从 Python 端写入）─────────────────────
curve_color = ti.Vector.field(3, dtype=ti.f32, shape=())


# ══════════════════════════════════════════════════════════════
#  CPU 端曲线算法
# ══════════════════════════════════════════════════════════════

def de_casteljau(points, t):
    """De Casteljau 递归算法，返回贝塞尔曲线在参数 t 处的 [x, y]。"""
    if len(points) == 1:
        return points[0]
    next_pts = []
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        next_pts.append([(1 - t) * p0[0] + t * p1[0],
                         (1 - t) * p0[1] + t * p1[1]])
    return de_casteljau(next_pts, t)


def b_spline_segment(p0, p1, p2, p3, t):
    """
    均匀三次 B 样条——单段求值。
    基矩阵（1/6 已提取）：
        [-1  3 -3  1]
        [ 3 -6  3  0]
        [-3  0  3  0]
        [ 1  4  1  0]
    """
    t2, t3 = t * t, t * t * t
    b0 = (-t3 + 3*t2 - 3*t + 1) / 6.0
    b1 = ( 3*t3 - 6*t2       + 4) / 6.0
    b2 = (-3*t3 + 3*t2 + 3*t + 1) / 6.0
    b3 =   t3                      / 6.0

    x = b0*p0[0] + b1*p1[0] + b2*p2[0] + b3*p3[0]
    y = b0*p0[1] + b1*p1[1] + b2*p2[1] + b3*p3[1]
    return [x, y]


def compute_bezier_points(ctrl_pts):
    """采样贝塞尔曲线，返回 shape=(NUM_SEGMENTS+1, 2) 的 numpy 数组。"""
    arr = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
    for k in range(NUM_SEGMENTS + 1):
        arr[k] = de_casteljau(ctrl_pts, k / NUM_SEGMENTS)
    return arr


def compute_bspline_points(ctrl_pts):
    """
    采样均匀三次 B 样条。
    n 个控制点 → n-3 段，每段均匀采样，总点数 ≤ NUM_SEGMENTS+1。
    返回 shape=(total, 2) 的 numpy 数组。
    """
    n = len(ctrl_pts)
    num_spans = n - 3                          # 段数
    pts_per_span = NUM_SEGMENTS // num_spans   # 每段采样点数

    all_pts = []
    for i in range(num_spans):
        p0, p1 = ctrl_pts[i], ctrl_pts[i + 1]
        p2, p3 = ctrl_pts[i + 2], ctrl_pts[i + 3]
        for k in range(pts_per_span + (1 if i == num_spans - 1 else 0)):
            t = k / pts_per_span
            all_pts.append(b_spline_segment(p0, p1, p2, p3, t))

    total = min(len(all_pts), NUM_SEGMENTS + 1)
    arr = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
    arr[:total] = np.array(all_pts[:total], dtype=np.float32)
    return arr, total


# ══════════════════════════════════════════════════════════════
#  GPU 内核
# ══════════════════════════════════════════════════════════════

@ti.kernel
def clear_pixels():
    """并行清空帧缓冲。"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def draw_curve_kernel(n: ti.i32):
    """
    基础光栅化：将曲线点映射为单像素。
    无反走样，适合对比观察锯齿现象。
    """
    color = curve_color[None]
    for i in range(n):
        pt = curve_points_field[i]
        px = ti.cast(pt[0] * WIDTH,  ti.i32)
        py = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
            pixels[px, py] = color


@ti.kernel
def draw_curve_kernel_aa(n: ti.i32):
    """
    反走样光栅化：对每个精确浮点坐标，考察 3×3 邻域，
    按距离权重（线性衰减）累加颜色，实现平滑边缘。
    """
    color = curve_color[None]
    half  = AA_KERNEL_HALF      # 邻域半径，当前 = 1
    r     = AA_RADIUS           # 距离衰减半径（像素单位）

    for i in range(n):
        pt = curve_points_field[i]
        fx = pt[0] * WIDTH        # 精确浮点像素坐标 x
        fy = pt[1] * HEIGHT       # 精确浮点像素坐标 y

        base_x = ti.cast(fx, ti.i32)
        base_y = ti.cast(fy, ti.i32)

        # 遍历 3×3 邻域
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                px = base_x + dx
                py = base_y + dy

                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    # 邻域像素中心坐标
                    cx = ti.cast(px, ti.f32) + 0.5
                    cy = ti.cast(py, ti.f32) + 0.5

                    # 到精确点的欧氏距离
                    dist = ti.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)

                    # 线性权重衰减：距离=0→weight=1，距离≥r→weight=0
                    weight = ti.max(0.0, 1.0 - dist / r)

                    # 原子累加（防止多个曲线点竞争同一像素时溢出）
                    old = pixels[px, py]
                    pixels[px, py] = ti.Vector([
                        ti.min(1.0, old[0] + color[0] * weight),
                        ti.min(1.0, old[1] + color[1] * weight),
                        ti.min(1.0, old[2] + color[2] * weight),
                    ])


# ══════════════════════════════════════════════════════════════
#  主循环
# ══════════════════════════════════════════════════════════════

def main():
    window = ti.ui.Window("Bezier & B-Spline  |  [b] switch  [a] AA  [c] clear",
                          (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    control_points: list = []
    mode      = 'bezier'   # 'bezier' | 'bspline'
    use_aa    = True        # 反走样开关

    # 预填充对象池（控制点全部藏到屏幕外）
    np_points  = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
    np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)

    def print_status():
        aa_str   = "ON" if use_aa else "OFF"
        pts_str  = str(len(control_points))
        print(f"[Mode: {mode.upper():8s}]  [AA: {aa_str}]  [Points: {pts_str}]")

    # ── 事件处理 ─────────────────────────────────────────────
    while window.running:
        for e in window.get_events(ti.ui.PRESS):

            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(list(pos))
                    print_status()

            elif e.key == 'c':
                control_points.clear()
                print("Canvas cleared.")

            elif e.key == 'b':
                mode = 'bspline' if mode == 'bezier' else 'bezier'
                print_status()

            elif e.key == 'a':
                use_aa = not use_aa
                print_status()

        # ── 每帧渲染 ─────────────────────────────────────────
        clear_pixels()
        n = len(control_points)

        # ── 绘制曲线 ─────────────────────────────────────────
        draw_bezier  = (mode == 'bezier'  and n >= 2)
        draw_bspline = (mode == 'bspline' and n >= 4)

        if draw_bezier or draw_bspline:
            # 1. 设置当前曲线颜色到 GPU
            if mode == 'bezier':
                c = COLOR_CURVE_AA_ON if use_aa else COLOR_CURVE_AA_OFF
            else:
                c = COLOR_CURVE_BSPLINE
            curve_color[None] = [c[0], c[1], c[2]]

            # 2. CPU 计算采样点
            if draw_bezier:
                pts_np     = compute_bezier_points(control_points)
                total_pts  = NUM_SEGMENTS + 1
            else:
                pts_np, total_pts = compute_bspline_points(control_points)

            # 3. 批量拷贝到 GPU（1 次通信）
            curve_points_field.from_numpy(pts_np)

            # 4. GPU 光栅化
            if use_aa:
                draw_curve_kernel_aa(total_pts)
            else:
                draw_curve_kernel(total_pts)

        elif mode == 'bspline' and 2 <= n < 4:
            # B 样条至少需要 4 个点，给用户提示（控制台）
            pass  # 可选：在画面上显示文字提示

        # ── 传帧 ─────────────────────────────────────────────
        canvas.set_image(pixels)

        # ── 绘制控制点与折线 ─────────────────────────────────
        if n > 0:
            # 对象池技巧：先全部置于屏幕外，再覆盖真实点
            np_points[:] = -10.0
            np_points[:n] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points,
                           radius=0.006,
                           color=COLOR_CONTROL_POINT)

            if n >= 2:
                # 构造折线索引
                indices = []
                for i in range(n - 1):
                    indices.extend([i, i + 1])
                np_indices[:] = 0
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points,
                             width=0.002,
                             indices=gui_indices,
                             color=COLOR_CONTROL_LINE)

        window.show()


if __name__ == '__main__':
    main()