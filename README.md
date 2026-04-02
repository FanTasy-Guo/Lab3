# Lab3
计算机图形学实验报告

## 项目概述
本实验基于 Python + Taichi 实现了一套交互式贝塞尔曲线绘制系统。项目涵盖从基础光栅化到 GPU 并行优化、再到选做的反走样与 B 样条扩展的完整链路，展示了现代图形渲染管线的核心设计思路。

## 项目框架
### 目录结构
project/
├── config.py       # 全局常量配置中心（尺寸、颜色、AA 参数）
├── physics.py      # GPU 内核 + CPU 曲线算法（核心计算层）
├── main.py         # 主循环 + 交互逻辑（应用层）
└── __pycache__/
### 模块职责划分
模块职责关键内容config.py配置中心WIDTH/HEIGHT、NUM_SEGMENTS、AA_RADIUS、各曲线颜色常量physics.py计算 & GPU 核心de_casteljau、b_spline_segment、GPU kernel（clear/draw/draw_aa）main.py交互 & 渲染循环GGUI 窗口、鼠标/键盘事件、帧缓冲传递、对象池管理
### 数据流向
CPU（Python）
  → 计算曲线采样点数组（de_casteljau / b_spline_segment）
  → 一次性 from_numpy 到 GPU（1 次通信）
  → GPU 并行光栅化（draw_curve_kernel / draw_curve_kernel_aa）
  → canvas.set_image(pixels) 显示

CPU 与 GPU 之间只发生 1 次内存通信，而非每点一次，这是避免帧率卡顿的核心设计。


## 基础实验
### 实验目标

理解贝塞尔曲线的几何意义与控制点概念
实现 De Casteljau 递归插值算法
掌握像素缓冲区（Frame Buffer）的基本操作与坐标映射
实现鼠标交互与实时曲线重绘

### De Casteljau 算法
核心数学原理
给定 n 个控制点 P₀, P₁, …, Pₙ₋₁，对每对相邻点做线性插值：
Pi′=(1−t)⋅Pi+t⋅Pi+1,t∈[0,1]P_i' = (1-t) \cdot P_i + t \cdot P_{i+1}, \quad t \in [0, 1]Pi′​=(1−t)⋅Pi​+t⋅Pi+1​,t∈[0,1]
对 n−1 个新点递归执行此操作，直到只剩 1 个点，即为曲线在参数 t 处的精确坐标。
代码实现
pythondef de_casteljau(points, t):
    if len(points) == 1:
        return points[0]          # 递归终止：唯一点即曲线上的点
    next_pts = []
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        next_pts.append([
            (1 - t) * p0[0] + t * p1[0],   # x 插值
            (1 - t) * p0[1] + t * p1[1],   # y 插值
        ])
    return de_casteljau(next_pts, t)        # 递归处理 n-1 个点
### 光栅化基础
屏幕本质是一个 800×800 的像素网格。将浮点坐标乘以屏幕尺寸并截断为整数，得到像素索引，写入颜色即完成"点亮像素"。
python@ti.kernel
def draw_curve_kernel(n: ti.i32):
    color = curve_color[None]           # 从共享 Field 读取当前颜色
    for i in range(n):                  # GPU 并行执行此循环
        pt = curve_points_field[i]
        px = ti.cast(pt[0] * WIDTH,  ti.i32)   # 浮点 → 像素索引
        py = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
            pixels[px, py] = color
### GPU 性能优化：Batching

为什么要 Batching？
CPU 与 GPU 物理分离，每次数据传输都需要经过 PCIe 总线。若在 Python 循环中每算出一个点就写一次 GPU Field，1000 个点会产生 1000 次通信，帧率大幅下降。
正确做法：CPU 算好全部点 → 一次性 from_numpy → GPU 并行点亮所有像素。

python# CPU 端计算所有采样点
curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
for k in range(NUM_SEGMENTS + 1):
    curve_points_np[k] = de_casteljau(control_points, k / NUM_SEGMENTS)
#1 次通信：整批发送到 GPU
curve_points_field.from_numpy(curve_points_np)
#GPU 并行光栅化
draw_curve_kernel(NUM_SEGMENTS + 1)
### 交互功能
操作触发方式说明添加控制点鼠标左键单击坐标归一化到 [0,1]，加入控制点列表清空画布键盘 C清空控制点列表，重置帧缓冲切换模式键盘 B贝塞尔 ↔ B 样条，曲线颜色同步切换切换反走样键盘 A开/关反走样，颜色变化辅助区分状态
### 对象池技巧
canvas.circles() 只能接受定长 Field。预分配大小为 MAX_CONTROL_POINTS=100 的数组，不需要显示的位置填充为 -10.0（藏在屏幕外），仅将真实控制点覆盖到数组前 n 个位置，避免动态内存分配。
pythonnp_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
np_points[:n] = np.array(control_points, dtype=np.float32)
gui_points.from_numpy(np_points)           # 一次性写入定长 Field
canvas.circles(gui_points, radius=0.006, color=COLOR_CONTROL_POINT)

## 选做优化
### 选做一：反走样（Anti-Aliasing）
#### 问题来源
基础光栅化将浮点坐标强制截断为整数，每个曲线点只点亮单一像素。斜线或曲线边缘因此出现明显的阶梯状锯齿效果（走样现象）。
#### 算法原理
利用浮点坐标的亚像素精度，考察精确点周围的 3×3 像素邻域。对每个邻域像素，计算其中心到精确点的欧氏距离，并通过线性衰减模型分配颜色权重：
weight=max⁡(0, 1−distradius)\text{weight} = \max\left(0,\ 1 - \frac{\text{dist}}{\text{radius}}\right)weight=max(0, 1−radiusdist​)
距离越近权重越大（最亮），距离超过 radius 时权重为 0（不贡献）。颜色以累加方式写入，实现多点平滑叠加。
#### GPU 内核实现
python@ti.kernel
def draw_curve_kernel_aa(n: ti.i32):
    color = curve_color[None]
    for i in range(n):
        fx = curve_points_field[i][0] * WIDTH    # 精确浮点像素坐标
        fy = curve_points_field[i][1] * HEIGHT
        base_x = ti.cast(fx, ti.i32)
        base_y = ti.cast(fy, ti.i32)

        for dx in range(-1, 2):      # 遍历 3×3 邻域
            for dy in range(-1, 2):
                px, py = base_x + dx, base_y + dy
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    cx = ti.cast(px, ti.f32) + 0.5  # 像素中心坐标
                    cy = ti.cast(py, ti.f32) + 0.5
                    dist = ti.sqrt((cx - fx)**2 + (cy - fy)**2)
                    w = ti.max(0.0, 1.0 - dist / AA_RADIUS)
                    old = pixels[px, py]
                    pixels[px, py] = ti.Vector([    # 颜色累加
                        ti.min(1.0, old[0] + color[0] * w),
                        ti.min(1.0, old[1] + color[1] * w),
                        ti.min(1.0, old[2] + color[2] * w),
                    ])
#### 视觉对比
状态颜色标识像素点亮逻辑反走样关（A 键）🟢 绿色单像素截断，边缘呈锯齿阶梯反走样开（A 键）🟡 黄色3×3 邻域距离权重累加，边缘平滑渐变

### 选做二：均匀三次 B 样条曲线
#### 贝塞尔的局限性

全局控制：移动任意一个控制点，整条曲线都会改变
阶数绑定：n 个控制点对应 n−1 阶多项式，控制点多时计算复杂且数值不稳定

#### B 样条的优势

局部控制：移动第 i 个控制点，只影响附近的 4 段曲线
阶数固定：无论多少控制点，始终保持三次（3 阶），计算稳定
连续性好：相邻段之间自动保证 C² 连续（二阶导数连续）

#### 矩阵形式算法
均匀三次 B 样条采用固定基矩阵，每 4 个相邻控制点构成一段曲线：
P(t)=16[t3t2t1][−13−313−630−30301410][P0P1P2P3]P(t) = \frac{1}{6} \begin{bmatrix} t^3 & t^2 & t & 1 \end{bmatrix} \begin{bmatrix} -1 & 3 & -3 & 1 \\ 3 & -6 & 3 & 0 \\ -3 & 0 & 3 & 0 \\ 1 & 4 & 1 & 0 \end{bmatrix} \begin{bmatrix} P_0 \\ P_1 \\ P_2 \\ P_3 \end{bmatrix}P(t)=61​[t3​t2​t​1​]​−13−31​3−604​−3331​1000​​​P0​P1​P2​P3​​​
展开为 4 个混合函数（Basis Functions）：
b0(t) = (−t³ + 3t² − 3t + 1) / 6
b1(t) = ( 3t³ − 6t²       + 4) / 6
b2(t) = (−3t³ + 3t² + 3t + 1) / 6
b3(t) =    t³                   / 6
#### 分段拼接逻辑
n 个控制点 → n−3 段三次曲线平滑拼接，每段在 t ∈ [0,1] 上独立采样后汇总：
pythondef compute_bspline_points(ctrl_pts):
    n = len(ctrl_pts)
    num_spans = n - 3                        # 段数
    pts_per_span = NUM_SEGMENTS // num_spans
    all_pts = []
    for i in range(num_spans):               # 逐段采样
        p0, p1, p2, p3 = ctrl_pts[i], ctrl_pts[i+1], ctrl_pts[i+2], ctrl_pts[i+3]
        for k in range(pts_per_span + 1):
            t = k / pts_per_span
            all_pts.append(b_spline_segment(p0, p1, p2, p3, t))
    return np.array(all_pts, dtype=np.float32), len(all_pts)
#### 模式切换设计
模式颜色最少控制点曲线特性贝塞尔（默认）🟢/🟡（AA 状态）2 个过端点，全局控制B 样条（B 键）🔵 蓝色4 个不过控制点，局部控制

## 关键设计亮点
### 颜色即状态——视觉化反馈系统
通过曲线颜色直观反映当前程序状态，用户无需查看控制台即可判断模式：
曲线颜色含义🟢 绿色贝塞尔模式 · 反走样关🟡 黄色贝塞尔模式 · 反走样开🔵 蓝色B 样条模式（反走样状态同步生效）
### 模块化配置中心
所有可调参数集中在 config.py，修改曲线颜色、AA 半径、采样精度时无需触碰算法代码，维护成本低。
### GPU-CPU 职责清晰分离

CPU 负责：算法逻辑（De Casteljau、B 样条基函数）、交互状态管理、颜色决策
GPU 负责：并行光栅化（draw_curve_kernel / draw_curve_kernel_aa）、帧缓冲清空
通信最小化：每帧只有 1 次 from_numpy 跨界传输，保证 60 FPS 流畅性

### 两种曲线的可观察行为对比
在同一组控制点下切换 B 键，可直观看到：

贝塞尔曲线穿过首尾控制点，整体形态随任意控制点移动而改变
B 样条曲线在控制点凸包内部平滑延伸，不穿过控制点
在密集添加控制点的区域，B 样条只有局部曲线段发生形变，贝塞尔则整体重塑

## 总结
本实验完整实现了从基础光栅化到 GPU 并行优化的完整渲染管线，并通过反走样和 B 样条两项选做扩展，展示了现代图形学的核心概念：

De Casteljau 算法：优雅的递归插值，数值稳定，实现简洁
Batching 优化：最小化 CPU-GPU 通信，是高性能渲染管线的基本原则
反走样：利用亚像素精度与距离权重，在不增加采样点的前提下提升视觉质量
B 样条：分段多项式设计解决了贝塞尔的全局耦合与高阶不稳定问题，是工程实践中更常用的样条方案
