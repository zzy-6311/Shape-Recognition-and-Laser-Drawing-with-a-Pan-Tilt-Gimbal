"""
work-pi.py — 二维云台激光绘图控制器（比赛主程序，单文件）
=========================================================

【做什么】
  用激光笔在画板上"画"图形，并用摄像头识别 A4 纸上的图案后复现到画板上。
    · 基本部分：画点回程、正方形、等边三角形、梯形
    · 发挥部分：转 180° 用摄像头识别目标图形 → 转到画板方位 → 按识别尺寸画图

【硬件平台】
  树莓派 + 两个 JC 系列步进电机驱动器（Modbus-RTU 串口）+ 激光笔
  + 7 段数码管（显示模式号）+ 两个按钮（切换模式 / 开始执行）

  接线（BCM 编号）：
    GPIO15                  激光继电器，高电平 = 激光亮
    GPIO9/11/0/5/6/13/19    数码管 a/b/c/d/e/f/g 段（共阳极，低电平点亮）
    GPIO21                  模式切换按钮（接 GND，启用内部上拉）
    GPIO20                  开始按钮（接 GND，启用内部上拉）
    /dev/ttyACM0 → Pan（航偏）电机驱动板
    /dev/ttyACM1 → Tilt（俯仰）电机驱动板

【运行环境与依赖】
  树莓派 OS + Python 3；依赖 pyserial、opencv-python、numpy、gpiozero
  非树莓派环境会自动降级为"无激光"模式，但本版本仍需要数码管+按钮才能选模式，
  所以在 PC 上主要用 my-cv.py / my-cv-one.py 调视觉参数，本文件用于上机。

【模式编号】（数码管显示，11 显示 A，12 显示 b）
    1  基本1  回程精度：在画板上点一个点 → 转 -360° → 回到原点
    2  基本2  正方形 80mm
    3  基本3  等边三角形 60mm
    4  基本4  梯形 上底100 / 下底60 / 高80 mm（画之前旋转 90° 竖起来）
    5  发挥1  转 180° 识别目标 → 回 0° 把图形画到画板
    6  发挥2  画板在 270° 方位
    7  发挥3  画板在 90° 方位
    8  发挥4  360° 扫描自动找画板与素材 → 闭环对准 → 画图
    9  空模式（占位，不执行任何动作）
   11  标定    WASD 粗调 1° / IJKL 微调 0.1°，C 保存零点偏移到 zero_offset.json
   12  摄像头调试：不控制云台，实时显示 A4 检测与图形识别，C 键现场算 K 值标定

  ⚠ 待核对：题目原文要求发挥1/2/3 的画板分别在 90°/180°/270°。本文件当前是
    5→0°、6→270°、7→90°，且没有画在 180° 的模式。上机前请对着实际场地核对
    m5 / m6 / m7 里的 draw_angle 取值。

【依赖的外部文件】
  calibration.json   测距标定 K1/K2，由 calibrate.py 或本文件模式 12 生成。
                     距离公式：D = (K1 / w_pixel + K2 / h_pixel) / 2
                     ⚠ K1/K2 与摄像头分辨率强相关，换摄像头或换分辨率必须重新标定。
  zero_offset.json   电机零点偏移 {pan, tilt}，由模式 11 生成；删除即恢复出厂零点。

【串口协议】
  见 other/云台串口通信说明.txt（Modbus-RTU：0x03 / 0x06 / 0x10，以及 PVT / PV 指令）

【同目录其它文件】
  my-cv.py        视觉算法母本（本文件的检测函数是从它复制后改造而来）
  my-cv-one.py    静态图片调参工具（不接摄像头）
  calibrate.py    A4 纸测距标定工具
"""

import serial
import serial.tools.list_ports
import threading
import json
import math
import struct
import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.font*=false"  # 静音 Qt 字体警告
import time

import cv2
import numpy as np

# ----- 激光继电器引脚（必须放在 import 之前）-----
LASER_PIN = 15                # GPIO15（物理 pin10），高电平=激光亮，低电平=激光灭

# ----- 激光继电器控制（gpiozero，树莓派）-----
# 如果不在树莓派上运行（比如在PC上测试），自动降级为无激光模式
try:
    from gpiozero import LED
    _laser = LED(LASER_PIN)
    _laser.off()
    _HAVE_LASER = True
    print(f"[激光] GPIO{LASER_PIN} 已初始化")
except ImportError:
    _HAVE_LASER = False
    print("[激光] gpiozero 未安装，激光控制已跳过")
except Exception as e:
    _HAVE_LASER = False
    print(f"[激光] GPIO 初始化失败: {e}")

# 显示辅助（直接显示，无压缩）
def _imshow(name, img):
    cv2.imshow(name, img)


# ============================================================
# 用户配置区 — 每个变量都有注释说明用途和生效模式
# ============================================================

# ----- 串口配置（模式1/2/3/4/5/11 需要正确设置）-----
SERIAL_PORT_PAN = "/dev/ttyACM0"      # 航偏角电机（Pan）的串口号，到设备管理器查看
SERIAL_PORT_TILT = "/dev/ttyACM1"     # 俯仰角电机（Tilt）的串口号，到设备管理器查看
BAUDRATE = 115200             # 串口波特率，与电机驱动板保持一致（默认115200）

# ----- 电机 Modbus 地址（每个串口独立，不同电机可用相同地址）-----
PAN_ID = 0x01                 # 航偏角电机的 Modbus 通信地址
TILT_ID = 0x01                # 俯仰角电机的 Modbus 通信地址

# ----- 零点标定文件（所有模式通用）-----
# 标定后的偏移数据保存在此文件，程序启动时自动加载。
# 删除此文件则恢复默认零位（使用电机出厂物理零点）。
ZERO_FILE = "zero_offset.json"

# ----- 电机控制模式（所有需要云台运动的模式）-----
# 0=力矩模式    1=速度模式    2=位置梯形轨迹(推荐,速度参数在此模式下才生效)
# 3=位置滤波    4=位置直通    5=低速扭矩模式
MOTOR_MODE = 2

# ----- 电机运动参数（所有需要云台运动的模式）-----
INIT_SPEED_RPM = 120          # 初始化归零时的运动速度，单位 rpm（值越大归零越快但越抖）
INIT_TORQUE_PCT = 100         # 初始化时的力矩百分比，范围 0~100（越大力量越大）
MOVE_SPEED_RPM = 360          # 绘制图形时的运动速度，单位 rpm（太大会抖动，太小会卡顿）
STEP_DELAY_S = 0.01           # 绘制时每两个点之间的延迟，单位秒（越小画越快但可能抖动）

# ----- 几何参数（画板到云台距离，模式1/2/3/4/5 固定 1m）-----
DISTANCE_MM = 900.0          # 激光发射点到画板的垂直距离，单位 mm（基础部分固定900mm）
TILT_OFFSET_MM = 10.0         # 俯仰旋转中心到激光头的物理偏移，单位 mm（需实测标定）

# ----- 基础模式形状尺寸（模式2/3/4 使用这些固定尺寸，单位 mm）-----
SQUARE_SIDE = 80              # 基本2：正方形边长，单位 mm（题目要求 8cm）
TRIANGLE_SIDE = 60            # 基本3：等边三角形边长，单位 mm（题目要求 6cm）
TRAPEZOID_TOP = 100           # 基本4：梯形上底，单位 mm（gen_trap 原版水平坐标系）
TRAPEZOID_BOTTOM = 60        # 基本4：梯形下底，单位 mm（m4 内会旋转 90° 竖起来画）
TRAPEZOID_HEIGHT = 80        # 基本4：梯形的高，单位 mm
                               # 注：m4 画图时会旋转 90°，平行边变竖直

# ----- 绘制遍数（所有画图形模式）-----
DRAW_REPEAT = 2               # 绘制重复遍数（≥1），2=画两遍更清晰但耗时翻倍

# ----- 形状插值精度（所有画图形模式）-----
CIRCLE_POINTS = 600           # 圆形插值点数，越多越圆滑但绘制时间越长
POINTS_PER_SIDE = 120          # 直线边每条边的插值点数，越多线条越平滑

# ============================================================
# ★★★ 闭环模式选择 — 数码管 + 按钮 ★★★
# 模式由按钮切换，数码管显示当前编号（11=A, 12=b）
# ============================================================

# ----- 数码管引脚（共阳极，低电平点亮）-----
SEG_A_PIN = 9                 # 7段数码管 — 段 a
SEG_B_PIN = 11                # 7段数码管 — 段 b
SEG_C_PIN = 0                 # 7段数码管 — 段 c
SEG_D_PIN = 5                 # 7段数码管 — 段 d
SEG_E_PIN = 6                 # 7段数码管 — 段 e
SEG_F_PIN = 13                # 7段数码管 — 段 f
SEG_G_PIN = 19                # 7段数码管 — 段 g

# ----- 按钮引脚（接 GND，启用内部上拉）-----
BTN_MODE_PIN = 21             # 模式切换按钮，按一次跳一个
BTN_START_PIN = 20            # 开始按钮，按下执行当前模式

# ----- 模式轮询列表（数码管 A=11, b=12）-----
MODE_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]

# ----- 7段码字型表（共阳极：0=点亮，1=熄灭）-----
# 引脚顺序：(a, b, c, d, e, f, g)
DIGIT_PATTERNS = {
    0: (0,0,0,0,0,0,1),
    1: (1,0,0,1,1,1,1),
    2: (0,0,1,0,0,1,0),
    3: (0,0,0,0,1,1,0),
    4: (1,0,0,1,1,0,0),
    5: (0,1,0,0,1,0,0),
    6: (0,1,0,0,0,0,0),
    7: (0,0,0,1,1,1,1),
    8: (0,0,0,0,0,0,0),
    9: (0,0,0,0,1,0,0),
    'A': (0,0,0,1,0,0,0),
    'b': (1,1,0,0,0,0,0),
}

# ============================================================
# ★★★ 摄像头 / 视觉参数调优区 ★★★
# 修改这些值来适配你的光照、距离、摄像头型号
# ============================================================

# ----- 摄像头编号与分辨率 -----
CAMERA_ID = 0                 # 0=笔记本内置摄像头  1=外接USB摄像头
CAMERA_WIDTH = 1080            # 采集宽度(像素)
CAMERA_HEIGHT = 640            # 采集高度(像素)
CAMERA_EXPOSURE = 200          # 手动曝光值（越大越亮/长曝光）
WARMUP_FRAMES = 15             # 暖帧帧数
WARMUP_DELAY = 0.1             # 暖帧每帧间隔(秒)
WARMUP_SETTLE = 0.3            # 暖帧后额外等待稳定(秒)

# ----- A4 纸检测参数（调参重点！）-----
# 检测流程：灰度化 → 高斯模糊 → 白色区域二值化 → 找轮廓 → 四边形逼近
A4_BRIGHT_THRESH = 170         # 白色纸面二值化阈值（0~255），高于此值为白
                               # 值越小→越容易把灰色背景当成白纸（弱光调低）
                               # 值越大→只把纯白当纸（强光调高）
                               # 典型范围：130~180

A4_GAUSSIAN_BLUR_SIZE = (3, 3)  # 模糊核大小：(3,3)保留细节适合远处，(5,5)去噪但细节少
A4_MIN_AREA = 7000              # A4纸最小轮廓面积(像素)
A4_CENTER_Y_RATIO = 1.0         # 全图搜索（1.0=不限制）
A4_CENTER_X_RATIO = 1.0         # 全图搜索（1.0=不限制）
A4_APPROX_EPSILON = 0.02        # 多边形逼近精度，越小顶点越多
A4_ANGLE_TOLERANCE = 8          # A4纸外框角点直角容差(度)

# ----- ROI 内部图形检测参数（如果找到A4纸但识别不出图形时改这里）-----
SHAPE_GAUSSIAN_BLUR_SIZE = (3, 3)  # 模糊核大小，同上
SHAPE_MIN_AREA = 1000              # 图形最小面积(像素)，太小会把噪声当图形
SHAPE_APPROX_EPSILON = 0.02
SHAPE_ANGLE_TOLERANCE = 5          # 直角判定容差(度)，越大越容易把梯形判成正方形
SHAPE_CIRCULARITY_THRESHOLD = 0.7  # 圆形度阈值(0~1)，越大圆越严格，非圆形不会被误判为圆
PARALLEL_TOLERANCE = 0.2          # 平行判定容差(归一化叉积=|sinθ|)，0.2≈11.5°，越大越宽松
SHAPE_DRAW_OFFSET_MM = 10         # 画图尺寸增量(mm)：检测值不变，画图时每个方向加此值

# ----- 测距参数（白色区域实际物理尺寸）-----
A4_WIDTH_MM = 170           # 白色区域宽度（mm），根据你的白框实际尺寸改
A4_HEIGHT_MM = 257          # 白色区域高度（mm）
CALIB_FILE = "calibration.json"
# K1/K2 标定方法：
#   1. 把 A4 纸放在画板位置，量实际距离 D（mm）
#   2. 运行模式 12（摄像头调试），看打印的 w_pixel / h_pixel
#   3. 计算 K1 = D × w_pixel, K2 = D × h_pixel
#   4. 写入 calibration.json：{"K1": 数值, "K2": 数值}
#   示例 (D=725mm)：{"K1": 246500, "K2": 362500}

# ----- 发挥4 扫描参数（模式8 生效）-----
SCAN_STEP_DEG = 30          # 每步转多少度（30°=12步扫一圈）
SCAN_DELAY_S = 1.0          # 每步等待云台稳定（秒）
MIN_SCAN_SEPARATION_DEG = 30  # 画板与素材的最小角度间距（度），小于此值说明是同一张纸的误识别
SERVO_THRESHOLD_PX = 2      # servo_to_board 的 X 轴像素容差（小于此值算对准）
SERVO_MAX_STEPS = 15        # servo_to_board 的最大调整步数
SERVO_ADJUST_GAIN = 0.01     # 每像素偏移对应云台调整角度（度/像素）
CAMERA_LASER_OFFSET_DEG = 0 # 摄像头中心到激光笔中心的水平偏差（度），正值=激光偏左需向右补偿
DIST_MIN_MM = 400           # 有效距离下限(mm)，低于此值视为测量错误，不补偿
DIST_MAX_MM = 1100          # 有效距离上限(mm)，高于此值视为测量错误，不补偿
SCAN_SHOW_DEBUG = True      # True=扫描时弹出窗口显示每步识别结果

# ============================================================
# 全局变量 — 零点偏移
# ============================================================
PAN_ZERO_OFFSET = 0.0
TILT_ZERO_OFFSET = 0.0

# ============================================================
# Modbus CRC16
# ============================================================
def crc16_modbus(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc

# ============================================================
# 串口操作
# ============================================================
def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    if not ports: return
    print("\n可用串口：")
    for p in sorted(ports, key=lambda x: x.device):
        print(f"  {p.device:>8}  {p.description}")

def open_serial(port, baudrate, label=""):
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1)
        print(f"{label}串口 {port} 已打开")
        return ser
    except serial.SerialException as e:
        print(f"错误：{port} {label} — {e}")
        return None

def send_modbus(ser, cmd_bytes, response_len=0, label=""):
    crc = crc16_modbus(cmd_bytes)
    full = cmd_bytes + bytes([crc & 0xFF, (crc >> 8) & 0xFF])
    try:
        ser.write(full)
        if response_len > 0:
            time.sleep(0.02)
            ser.read(response_len)
    except Exception as e:
        print(f"  错误：{label} — {e}")

def send_parallel(ser_a, ca, ser_b, cb, rl=0):
    def _w(s, c):
        crc = crc16_modbus(c)
        f = c + bytes([crc & 0xFF, (crc >> 8) & 0xFF])
        try:
            s.write(f)
            rl > 0 and (time.sleep(0.02), s.read(rl))
        except: pass
    t1 = threading.Thread(target=_w, args=(ser_a, ca))
    t2 = threading.Thread(target=_w, args=(ser_b, cb))
    t1.start(); t2.start(); t1.join(); t2.join()

# ============================================================
# 电机控制
# ============================================================
def motor_set_mode(ser, mid, mode):
    send_modbus(ser, bytes([mid, 6, 0, 0x60, 0, mode]), 8, f"mode={mode}")

def motor_close_loop(ser, mid):
    send_modbus(ser, bytes([mid, 6, 0, 0xA2, 0, 1]), 8, "close")

def motor_idle(ser, mid):
    send_modbus(ser, bytes([mid, 6, 0, 0xA0, 0, 1]), 8, "idle")

def motor_read_pos(ser, mid):
    response = send_modbus(ser, bytes([mid, 3, 0, 8, 0, 2]), 9, "pos")
    if response is not None and len(response) >= 7:
        return struct.unpack(">i", response[3:7])[0] / 100.0
    return 0.0

def _a2r(d): return int(round(d * 100))
def _s2r(r): return int(round(r))

def pvt_cmd(mid, deg, rpm, tor):
    return bytes([mid, 0x25]) + struct.pack(">i", _a2r(deg)) + struct.pack(">h", _s2r(rpm)) + bytes([max(0, min(100, int(tor))) & 0xFF])

def pv_cmd(mid, deg, rpm):
    return bytes([mid, 0x24]) + struct.pack(">i", _a2r(deg)) + struct.pack(">h", _s2r(rpm))

# ============================================================
# 零点偏移 — 文件持久化
# ============================================================
def save_zero(pan, tilt):
    with open(ZERO_FILE, "w") as f:
        json.dump({"pan": pan, "tilt": tilt}, f)

def load_zero():
    if os.path.exists(ZERO_FILE):
        with open(ZERO_FILE) as f:
            d = json.load(f)
            return d.get("pan", 0.0), d.get("tilt", 0.0)
    return 0.0, 0.0

# ============================================================
# 初始化与归零
# ============================================================
def laser_on():
    """打开激光笔"""
    if _HAVE_LASER:
        _laser.on()
    else:
        print("  [激光] 跳过（无硬件）")


def laser_off():
    """关闭激光笔"""
    if _HAVE_LASER:
        _laser.off()


def init_motors(sp, st):
    laser_off()
    print("\n初始化电机...")
    send_parallel(sp, bytes([PAN_ID, 6, 0, 0xA2, 0, 1]),
                  st, bytes([TILT_ID, 6, 0, 0xA2, 0, 1]), 8)
    time.sleep(0.05)
    send_parallel(sp, bytes([PAN_ID, 6, 0, 0x60, 0, MOTOR_MODE]),
                  st, bytes([TILT_ID, 6, 0, 0x60, 0, MOTOR_MODE]), 8)
    time.sleep(0.05)
    send_parallel(sp, pvt_cmd(PAN_ID, 0 + PAN_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT),
                  st, pvt_cmd(TILT_ID, 0 + TILT_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT), 14)
    time.sleep(0.5)
    print(f"  归零→ Pan:{motor_read_pos(sp,PAN_ID):+.2f}° Tilt:{motor_read_pos(st,TILT_ID):+.2f}°\n")

def home_motors(sp, st):
    laser_off()
    send_parallel(sp, pvt_cmd(PAN_ID, 0 + PAN_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT),
                  st, pvt_cmd(TILT_ID, 0 + TILT_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT), 14)
    time.sleep(0.5)

# ============================================================
# 坐标 → 角度
# ============================================================
def xy_to_angle(x, y):
    return math.degrees(math.atan2(x, DISTANCE_MM)), \
           math.degrees(math.atan2(y, DISTANCE_MM + TILT_OFFSET_MM))

# ============================================================
# 形状点集生成（画图用）
# ============================================================
def _ic(c):
    r = []
    for i in range(len(c)):
        x1, y1 = c[i]; x2, y2 = c[(i+1)%len(c)]
        for j in range(POINTS_PER_SIDE):
            t = j / POINTS_PER_SIDE
            r.append((x1+(x2-x1)*t, y1+(y2-y1)*t))
    return r

def gen_square(s): h=s/2; return _ic([(-h,-h),(h,-h),(h,h),(-h,h)])
def gen_tri(s): h=s*math.sqrt(3)/2; return _ic([(0,h*2/3),(-s/2,-h/3),(s/2,-h/3)])
def gen_trap(t,b,h): ht=t/2; hb=b/2; hh=h/2; return _ic([(-ht,-hh),(ht,-hh),(hb,hh),(-hb,hh)])
def rotate_points_90(pts):
    """点集顺时针旋转 90°（用于基础4梯形竖起来画）"""
    return [(-y, x) for x, y in pts]
def gen_circle(r):
    return [(r*math.cos(2*math.pi*i/CIRCLE_POINTS), r*math.sin(2*math.pi*i/CIRCLE_POINTS)) for i in range(CIRCLE_POINTS)]

# ============================================================
# 绘制（叠加零点偏移）
# ============================================================
def draw(sp, st, pts, label, tilt_shift=0):
    n = len(pts)
    print(f"绘制 {label}，{n} 点 × {DRAW_REPEAT} 遍 @ {MOVE_SPEED_RPM}rpm")
    # 预定位到第一个点再开激光，防止留影
    p0, t0 = xy_to_angle(pts[0][0], pts[0][1])
    send_parallel(sp, pv_cmd(PAN_ID, p0+PAN_ZERO_OFFSET, MOVE_SPEED_RPM),
                  st, pv_cmd(TILT_ID, t0+TILT_ZERO_OFFSET+tilt_shift, MOVE_SPEED_RPM), 14)
    time.sleep(0.3)
    laser_on()
    time.sleep(0.3)
    for r in range(DRAW_REPEAT):
        if DRAW_REPEAT > 1:
            print(f"  第 {r+1}/{DRAW_REPEAT} 遍")
        for i, (x, y) in enumerate(pts):
            p, t = xy_to_angle(x, y)
            send_parallel(sp, pv_cmd(PAN_ID, p+PAN_ZERO_OFFSET, MOVE_SPEED_RPM),
                          st, pv_cmd(TILT_ID, t+TILT_ZERO_OFFSET, MOVE_SPEED_RPM), 14)
            i % 50 == 0 and print(f"  进度 {i}/{n}")
            time.sleep(STEP_DELAY_S)
    laser_off()
    print("  完成")

def draw_at_angle(sp, st, pts, label, base_pan_deg, tilt_shift=0):
    """在指定航偏角方向绘制图形（用于画板不在0°的情况）"""
    n = len(pts)
    print(f"绘制 {label} 于 {base_pan_deg:.0f}°方向，{n} 点 × {DRAW_REPEAT} 遍 @ {MOVE_SPEED_RPM}rpm")
    # 预定位到第一个点再开激光
    p0, t0 = xy_to_angle(pts[0][0], pts[0][1])
    send_parallel(sp,
        pv_cmd(PAN_ID, base_pan_deg + p0 + PAN_ZERO_OFFSET, MOVE_SPEED_RPM),
        st, pv_cmd(TILT_ID, t0 + TILT_ZERO_OFFSET + tilt_shift, MOVE_SPEED_RPM), 14)
    time.sleep(0.3)
    laser_on()
    time.sleep(0.3)
    for r in range(DRAW_REPEAT):
        if DRAW_REPEAT > 1:
            print(f"  第 {r+1}/{DRAW_REPEAT} 遍")
        for i, (x, y) in enumerate(pts):
            p, t = xy_to_angle(x, y)
            send_parallel(sp,
                pv_cmd(PAN_ID, base_pan_deg + p + PAN_ZERO_OFFSET, MOVE_SPEED_RPM),
                st, pv_cmd(TILT_ID, t + TILT_ZERO_OFFSET + tilt_shift, MOVE_SPEED_RPM), 14)
            i % 50 == 0 and print(f"  进度 {i}/{n}")
            time.sleep(STEP_DELAY_S)
    laser_off()
    print("  完成")

# ============================================================
# 标定
# ============================================================
def calibrate(sp, st):
    global PAN_ZERO_OFFSET, TILT_ZERO_OFFSET
    PAN_ZERO_OFFSET, TILT_ZERO_OFFSET = load_zero()
    print("\n" + "="*50 + "\n标定模式  W/S/A/D=粗调1°  I/K/J/L=微调0.1°\nR=归零  P=读位置  C=保存  Q=退出\n" + "="*50)
    send_parallel(sp, pvt_cmd(PAN_ID, 0+PAN_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT),
                  st, pvt_cmd(TILT_ID, 0+TILT_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT), 14)
    time.sleep(0.5)
    cp, ct = motor_read_pos(sp, PAN_ID), motor_read_pos(st, TILT_ID)
    print(f"初始→ Pan:{cp:+.2f}° Tilt:{ct:+.2f}°")
    while True:
        print(f"\n>>> Pan:{cp:+.2f}° Tilt:{ct:+.2f}°")
        i = input("命令: ").strip().lower()
        if not i: continue
        m = False
        command_char = i[0]  # 取第一个字符作为命令
        for ch in i:
            if ch=='w': ct+=1; m=True
            elif ch=='s': ct-=1; m=True
            elif ch=='a': cp-=1; m=True
            elif ch=='d': cp+=1; m=True
            elif ch=='i': ct+=0.1; m=True
            elif ch=='k': ct-=0.1; m=True
            elif ch=='j': cp-=0.1; m=True
            elif ch=='l': cp+=0.1; m=True
            elif ch=='r': cp=0; ct=0; m=True
        if m:
            send_parallel(sp, pvt_cmd(PAN_ID, cp, MOVE_SPEED_RPM, INIT_TORQUE_PCT),
                          st, pvt_cmd(TILT_ID, ct, MOVE_SPEED_RPM, INIT_TORQUE_PCT), 14)
            time.sleep(0.15)
            'r' in i and print(f"  归零→ Pan:{motor_read_pos(sp,PAN_ID):+.2f}° Tilt:{motor_read_pos(st,TILT_ID):+.2f}°")
            continue
        if command_char == 'p':
            print(f"  位置→ Pan:{motor_read_pos(sp,PAN_ID):+.2f}° Tilt:{motor_read_pos(st,TILT_ID):+.2f}°")
        elif command_char == 'c':
            PAN_ZERO_OFFSET, TILT_ZERO_OFFSET = motor_read_pos(sp,PAN_ID), motor_read_pos(st,TILT_ID)
            save_zero(PAN_ZERO_OFFSET, TILT_ZERO_OFFSET)
            print(f"  已保存: Pan={PAN_ZERO_OFFSET:+.2f}° Tilt={TILT_ZERO_OFFSET:+.2f}°")
            break
        elif command_char == 'q': break

# ============================================================
# ===== 视觉识别（新 — 单帧检测）============================
# ============================================================

def angle_between(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    m = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
    if m == 0:
        return 0
    return math.degrees(math.acos(dot / m))


def classify_shape(approx):
    """按顶点数和圆形度分类图形，返回 (名称, 顶点数)"""
    n = len(approx)
    if n == 3:
        return ("triangle", n)
    elif n == 4:
        pts = approx.reshape(4, 2)
        angles = []
        for i in range(4):
            v1 = pts[(i - 1) % 4] - pts[i]
            v2 = pts[(i + 1) % 4] - pts[i]
            angles.append(angle_between(v1, v2))
        # 如果有一个角接近180°（多余顶点在直边上），去掉它当三角形处理
        if any(a > 170 for a in angles):
            return ("triangle", 3)
        is_square = all(abs(a - 90) < SHAPE_ANGLE_TOLERANCE for a in angles)
        return ("square", n) if is_square else ("trapezoid", n)
    elif n >= 5:
        # 只有圆形度达标的才算圆形，避免粗糙三角形/梯形误判为圆
        area = cv2.contourArea(approx)
        peri = cv2.arcLength(approx, True)
        if peri > 0:
            circularity = 4 * math.pi * area / (peri * peri)
            if circularity > SHAPE_CIRCULARITY_THRESHOLD:
                return ("circle", n)
        return ("unknown", n)
    return ("unknown", n)


def measure_shape(approx, name, mm_per_px):
    """测量图形实际尺寸（mm），返回参数字典"""
    pts = approx.reshape(len(approx), 2)

    if name == "circle":
        (_, _), radius = cv2.minEnclosingCircle(approx)
        return {"radius": radius * mm_per_px}

    elif name in ("square", "triangle"):
        sides = []
        for i in range(len(pts)):
            sides.append(math.sqrt((pts[i][0]-pts[(i+1)%len(pts)][0])**2 + (pts[i][1]-pts[(i+1)%len(pts)][1])**2) * mm_per_px)
        avg = sum(sides) / len(sides)
        return {"side": avg}

    elif name == "trapezoid":
        s = [math.sqrt((pts[i][0]-pts[(i+1)%4][0])**2 + (pts[i][1]-pts[(i+1)%4][1])**2) for i in range(4)]
        v = [pts[(i+1)%4] - pts[i] for i in range(4)]

        def is_par(v1, v2):
            cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
            norm = math.sqrt(v1[0]**2+v1[1]**2) * math.sqrt(v2[0]**2+v2[1]**2)
            return norm > 0 and cross / norm < PARALLEL_TOLERANCE

        if is_par(v[0], v[2]):
            # 用 y 坐标判断上下：v[0]连接pts[0]↔pts[1]，v[2]连接pts[2]↔pts[3]
            y0 = (pts[0][1] + pts[1][1]) / 2  # v[0] 的中间 y
            y2 = (pts[2][1] + pts[3][1]) / 2  # v[2] 的中间 y
            top_len = s[0] if y0 < y2 else s[2]
            bot_len = s[2] if y0 < y2 else s[0]
            top = top_len * mm_per_px
            bottom = bot_len * mm_per_px
            # 高 = 平行边v[0] × 侧边v[1] 的叉积 ÷ |v[0]|
            cross = abs(v[1][0]*v[0][1] - v[1][1]*v[0][0])
            h_px = cross / math.sqrt(v[0][0]**2 + v[0][1]**2) if math.sqrt(v[0][0]**2+v[0][1]**2) > 0 else 0
            height = h_px * mm_per_px
        elif is_par(v[1], v[3]):
            # 用 x 坐标判断左右：v[1]连接pts[1]↔pts[2]，v[3]连接pts[3]↔pts[0]
            x1 = (pts[1][0] + pts[2][0]) / 2  # v[1] 的中间 x
            x3 = (pts[3][0] + pts[0][0]) / 2  # v[3] 的中间 x
            top_len = s[1] if x1 < x3 else s[3]
            bot_len = s[3] if x1 < x3 else s[1]
            top = top_len * mm_per_px
            bottom = bot_len * mm_per_px
            # 高 = 平行边v[1] × 侧边v[2] 的叉积 ÷ |v[1]|
            cross = abs(v[2][0]*v[1][1] - v[2][1]*v[1][0])
            h_px = cross / math.sqrt(v[1][0]**2 + v[1][1]**2) if math.sqrt(v[1][0]**2+v[1][1]**2) > 0 else 0
            height = h_px * mm_per_px
        else:
            return {"top": 0, "bottom": 0, "height": 0}
        return {"top": top, "bottom": bottom, "height": height}

    return {}


def get_pixel_size(vertices):
    """计算 A4 纸四个顶点的平均像素宽和高"""
    pts = vertices.reshape(4, 2)
    w1 = math.sqrt((pts[0][0]-pts[1][0])**2 + (pts[0][1]-pts[1][1])**2)
    w2 = math.sqrt((pts[2][0]-pts[3][0])**2 + (pts[2][1]-pts[3][1])**2)
    h1 = math.sqrt((pts[1][0]-pts[2][0])**2 + (pts[1][1]-pts[2][1])**2)
    h2 = math.sqrt((pts[3][0]-pts[0][0])**2 + (pts[3][1]-pts[0][1])**2)
    return (w1 + w2) / 2, (h1 + h2) / 2


def load_calib():
    """加载标定文件，返回 K1, K2"""
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE) as f:
            data = json.load(f)
            return data.get("K1", 0), data.get("K2", 0)
    return 0, 0


def shape_to_points(shape_type, params):
    """视觉识别结果 → 画图点集"""
    if shape_type == "circle":
        return gen_circle(params["radius"])
    elif shape_type == "square":
        return gen_square(params["side"])
    elif shape_type == "triangle":
        return gen_tri(params["side"])
    elif shape_type == "trapezoid":
        # 发挥模式中视觉识别的上下底是反的，交换
        return gen_trap(params["bottom"], params["top"], params["height"])
    return []


# my-cv.py 兼容别名
load_calibration = load_calib


def detect_a4(frame):
    """
    检测画面中的白色矩形区域（A4 纸内部白面）。
    返回 (vertices, w_pixel, h_pixel, distance_mm, roi, mm_per_px) 或 None
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, A4_GAUSSIAN_BLUR_SIZE, 0)

    # 亮二值：白色纸面 → 白色，黑色背景/干扰 → 黑色
    # 白纸是单一外轮廓，纸内图案=内部孔洞，不干扰外轮廓
    _, bright_bin = cv2.threshold(blur, A4_BRIGHT_THRESH, 255, cv2.THRESH_BINARY)

    y_min = int(h * (1 - A4_CENTER_Y_RATIO) / 2)
    y_max = int(h * (1 + A4_CENTER_Y_RATIO) / 2)
    x_min = int(w * (1 - A4_CENTER_X_RATIO) / 2)
    x_max = int(w * (1 + A4_CENTER_X_RATIO) / 2)

    # RETR_TREE：只保留偶数层轮廓（外部/岛），排除奇数层（洞）
    contours, hierarchy = cv2.findContours(bright_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    def _depth(i):
        d = 0; p = hierarchy[0][i][3]
        while p != -1:
            d += 1; p = hierarchy[0][p][3]
        return d
    contours = [cnt for i, cnt in enumerate(contours) if _depth(i) % 2 == 0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    K1, K2 = load_calibration()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < A4_MIN_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, A4_APPROX_EPSILON * peri, True)
        if len(approx) != 4:
            continue

        # 矩形验证：4个角是否接近90°
        pts_check = approx.reshape(4, 2)
        rect_ok = True
        for i in range(4):
            v1 = pts_check[(i - 1) % 4] - pts_check[i]
            v2 = pts_check[(i + 1) % 4] - pts_check[i]
            a = angle_between(v1, v2)
            if abs(a - 90) > A4_ANGLE_TOLERANCE:
                rect_ok = False
                break
        if not rect_ok:
            continue

        w_pixel, h_pixel = get_pixel_size(approx)

        # 宽高比验证
        _aspect = max(w_pixel, h_pixel) / min(w_pixel, h_pixel)
        if abs(_aspect - A4_HEIGHT_MM / A4_WIDTH_MM) > 0.15:
            continue

        distance = 0
        if K1 > 0 and K2 > 0:
            distance = (K1 / w_pixel + K2 / h_pixel) / 2

        pts = approx.reshape(4, 2)
        x1 = max(0, int(min(pts[:, 0])) - 5)
        x2 = min(w, int(max(pts[:, 0])) + 5)
        y1 = max(0, int(min(pts[:, 1])) - 5)
        y2 = min(h, int(max(pts[:, 1])) + 5)
        roi = frame[y1:y2, x1:x2]

        roi_h, roi_w = roi.shape[:2]
        mm_per_px = (A4_WIDTH_MM / roi_w + A4_HEIGHT_MM / roi_h) / 2

        return (approx, w_pixel, h_pixel, distance, roi, mm_per_px)

    return None


def find_shapes_in_roi(roi, mm_per_px):
    """在 A4 纸 ROI 中检测内部图形，返回 [(shape_name, params_dict), ...]"""
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_blur = cv2.GaussianBlur(roi_gray, SHAPE_GAUSSIAN_BLUR_SIZE, 0)

    # 自适应阈值：根据局部亮度算阈值，不受光照变化影响
    # blockSize=31 保证在大面积黑色图形内部也能包含纸面白色像素
    roi_binary = cv2.adaptiveThreshold(roi_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 5)

    # 如果自适应阈值检测不到（大面积黑色图形），改用 OTSU
    if cv2.countNonZero(roi_binary) < 50:
        _, roi_binary = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 用中心掩膜排除 A4 纸的黑边框（只保留纸内部区域）
    roi_h, roi_w = roi.shape[:2]
    inset = 20
    center_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    center_mask[inset:roi_h - inset, inset:roi_w - inset] = 255
    roi_binary = cv2.bitwise_and(roi_binary, center_mask)

    roi_contours, _ = cv2.findContours(roi_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for rc in roi_contours:
        ra = cv2.contourArea(rc)
        if ra < SHAPE_MIN_AREA:
            continue
        rp = cv2.arcLength(rc, True)
        rap = cv2.approxPolyDP(rc, SHAPE_APPROX_EPSILON * rp, True)
        # 如果标准精度给恰好4个顶点，逐步放宽精度试能否降为3（三角形粗边多出的顶点）
        # 注意：≥5顶点的绝不放宽（那是圆形，放宽会坍缩成正方形）
        if len(rap) == 4:
            for _mult in (1.5, 2.0, 3.0):
                rap2 = cv2.approxPolyDP(rc, _mult * SHAPE_APPROX_EPSILON * rp, True)
                if len(rap2) == 3:
                    rap = rap2
                    break
        name, verts = classify_shape(rap)
        params = measure_shape(rap, name, mm_per_px)
        if params:
            # 如果尺寸接近A4纸大小（>150mm），不是内部图形，是A4纸边框残留
            too_big = False
            for k, v in params.items():
                if isinstance(v, (int, float)) and v > 150:
                    too_big = True
                    break
            if not too_big:
                results.append((name, params))

    return results


def process_frame(frame):
    """
    完整处理一帧（与 my-cv.py 完全一致）。
    返回 (ok, shape_name, shape_params, message, a4_info)
      a4_info = (vertices, distance_mm) 或 None
    """
    a4_result = detect_a4(frame)
    if a4_result is None:
        return (False, None, None, "no A4", None)

    vertices, w_pixel, h_pixel, distance, roi, mm_per_px = a4_result

    shapes = find_shapes_in_roi(roi, mm_per_px)

    if not shapes:
        return (False, None, None, "A4 found, no shape", (vertices, distance))

    shape_name, shape_params = shapes[0]
    a4_info = (vertices, distance)
    return (True, shape_name, shape_params, "OK", a4_info)


# ============================================================
# 发挥4 辅助函数：360° 扫描 + 闭环对准
# ============================================================

def scan_for_papers(sp, st):
    """
    旋转一圈，找到画板和素材各自所在的方位角。
    返回 (board_azimuth, material_info) 或 None。
      material_info = (shape_type, params_dict) 或 None
    """
    print("\n===== 开始 360° 扫描 =====")
    home_motors(sp, st)
    time.sleep(0.5)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)
    # 长暖帧让摄像头自动曝光稳定（跳过15帧）
    for _ in range(15):
        cap.read()
        time.sleep(0.1)
    time.sleep(0.3)
    cap.read()

    found_board = None
    found_material = None

    angle = 0
    step = 0
    while angle < 360:
        step += 1
        print(f"\n步骤 {step}: {angle:.0f}°")

        send_parallel(sp, pvt_cmd(PAN_ID, angle + PAN_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT),
                      st, pvt_cmd(TILT_ID, 0 + TILT_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT), 14)
        time.sleep(SCAN_DELAY_S)

        # 等曝光稳定（云台转动后场景亮度变化，需要暖帧）
        for _ in range(10):
            cap.read()
            time.sleep(0.05)
        ret, frame = cap.read()
        if not ret:
            angle += SCAN_STEP_DEG
            continue

        a4_result = detect_a4(frame)
        if a4_result is None:
            print("  无 A4 纸")
            if SCAN_SHOW_DEBUG:
                _imshow("Scan Debug", frame)
                cv2.waitKey(1)
            angle += SCAN_STEP_DEG
            continue

        vertices, w_pixel, h_pixel, distance, roi, mm_per_px = a4_result
        shapes = find_shapes_in_roi(roi, mm_per_px)

        if SCAN_SHOW_DEBUG:
            debug_frame = frame.copy()
            cv2.drawContours(debug_frame, [vertices], -1, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"{angle:.0f}deg", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _imshow("Scan Debug", debug_frame)
            cv2.waitKey(1)

        if not shapes:
            if found_board is None:
                # 再拍一张确认（避免单帧误判）
                time.sleep(0.3)
                cap.read()
                ret2, frame2 = cap.read()
                if ret2:
                    a4_retry = detect_a4(frame2)
                    if a4_retry:
                        _, _, _, _, roi2, mm2 = a4_retry
                        shapes2 = find_shapes_in_roi(roi2, mm2)
                        if not shapes2:
                            # 如果已经找到素材且角度太近，说明是同一张纸的误识别
                            if found_material is not None and abs(angle - found_material[0]) < MIN_SCAN_SEPARATION_DEG:
                                print(f"  画板候选@ {angle:.0f}° 与素材({found_material[0]:.0f}°)太近，视为同一张纸")
                            else:
                                found_board = angle
                                print(f"  ★ 找到画板(空白) → 方位 {angle:.0f}°")
                                if SCAN_SHOW_DEBUG:
                                    cv2.imshow("Board ROI", roi2)
                                    cv2.waitKey(1)
                        else:
                            print("  画板候选但二次确认有图形 → 跳过")
                    else:
                        print("  画板候选但二次确认无A4纸 → 跳过")
                else:
                    print("  画板候选但二次确认拍照失败 → 跳过")
        else:
            stype, sparams = shapes[0]
            if found_material is None:
                # 再拍一张确认
                time.sleep(0.3)
                cap.read()
                ret2, frame2 = cap.read()
                if ret2:
                    a4_retry = detect_a4(frame2)
                    if a4_retry:
                        _, _, _, _, roi2, mm2 = a4_retry
                        shapes2 = find_shapes_in_roi(roi2, mm2)
                        if shapes2:
                            if found_board is not None and abs(angle - found_board) < MIN_SCAN_SEPARATION_DEG:
                                print(f"  素材候选@ {angle:.0f}° 与画板({found_board:.0f}°)太近，视为同一张纸")
                            else:
                                stype2, sparams2 = shapes2[0]
                                found_material = (angle, stype2, sparams2, mm2)
                                print(f"  ★ 找到素材({stype2}) → 方位 {angle:.0f}°")
                                for k, v in sparams2.items():
                                    print(f"    {k}={v:.1f}mm")
                            if SCAN_SHOW_DEBUG:
                                cv2.imshow("Material ROI", roi2)
                                cv2.waitKey(1)
                        else:
                            print("  素材候选但二次确认无图形 → 跳过")
                    else:
                        print("  素材候选但二次确认无A4纸 → 跳过")
                else:
                    print("  素材候选但二次确认拍照失败 → 跳过")

        if found_board is not None and found_material is not None:
            print("\n画板和素材均已找到，提前结束扫描")
            break

        angle += SCAN_STEP_DEG

    cap.release()
    cv2.destroyAllWindows()
    return found_board, found_material


def servo_to_board(sp, st, initial_azimuth):
    """
    快速对准画板 X 轴中心（只调 Pan，不动 Tilt）。
    只看 X 轴像素偏移，容差 SERVO_THRESHOLD_PX，每步 0.2s，最多 SERVO_MAX_STEPS 步。
    返回 (aligned_azimuth, board_mm_per_px, board_distance_mm)
    """
    print(f"  对准画板 (初始 {initial_azimuth:.0f}°)")

    send_parallel(sp, pvt_cmd(PAN_ID, initial_azimuth + PAN_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT),
                  st, pvt_cmd(TILT_ID, 0 + TILT_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT), 14)
    time.sleep(0.5)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)
    for _ in range(15):
        cap.read()
        time.sleep(0.1)
    time.sleep(0.3)
    cap.read()

    pan_angle = initial_azimuth
    board_mm_per_px = 0
    board_distance = 0

    for s in range(SERVO_MAX_STEPS):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.2)
            continue

        if s == 0:
            image_center_x = frame.shape[1] / 2

        a4 = detect_a4(frame)
        # 显示实时画面
        show = frame.copy()
        if a4:
            cv2.drawContours(show, [a4[0]], -1, (0, 255, 0), 3)
            cx = int(a4[0][:, 0, 0].mean())
            cv2.line(show, (int(image_center_x), 0), (int(image_center_x), show.shape[0]), (0, 0, 255), 1)
            cv2.line(show, (cx, 0), (cx, show.shape[0]), (0, 255, 0), 1)
        _imshow("Servo Align", show)
        cv2.waitKey(1)

        if a4 is None:
            print(f"  第{s+1}步: 未检测到画板")
            time.sleep(0.2)
            continue

        vertices, _, _, board_distance, _, board_mm_per_px = a4
        cx = int(vertices[:, 0, 0].mean())
        offset_x = cx - image_center_x

        print(f"  第{s+1}步: 偏移{offset_x:+.0f}px", end="")

        if abs(offset_x) < SERVO_THRESHOLD_PX:
            pan_angle += CAMERA_LASER_OFFSET_DEG
            print(f" → ✓ 对准 Pan={pan_angle:.1f}°")
            cap.release()
            cv2.destroyWindow("Servo Align")
            return pan_angle, board_mm_per_px, board_distance

        adjust = -offset_x * SERVO_ADJUST_GAIN
        pan_angle += adjust
        pan_angle = max(0, min(360, pan_angle))
        print(f" → 调{adjust:+.2f}°")
        send_parallel(sp, pvt_cmd(PAN_ID, pan_angle + PAN_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT),
                      st, pvt_cmd(TILT_ID, 0 + TILT_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT), 14)
        time.sleep(0.2)

    pan_angle += CAMERA_LASER_OFFSET_DEG
    print(f"  对准结束(最大步数) Pan={pan_angle:.1f}°")
    cap.release()
    cv2.destroyWindow("Servo Align")
    return pan_angle, board_mm_per_px, board_distance


def camera_detect_shape(show_debug=False):
    """
    单帧拍照识别：打开摄像头 → 拍一张 → 找A4纸 → 找图形 → 返回。
    show_debug=True 时显示处理过程的窗口（按任意键继续）。
    返回 (ok, shape_type, params, message)
    """
    print("打开摄像头...")
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)

    for _ in range(15):
        cap.read()
        time.sleep(0.1)
    time.sleep(0.3)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return (False, None, None, "拍照失败")

    # ---- A4 纸检测（白色区域）----
    a4_result = detect_a4(frame)
    if a4_result is None:
        if show_debug:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, A4_GAUSSIAN_BLUR_SIZE, 0)
            _, binary = cv2.threshold(blur, A4_BRIGHT_THRESH, 255, cv2.THRESH_BINARY)
            _imshow("Camera Raw", frame)
            _imshow("Binary", binary)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return (False, None, None, "no A4 paper")

    vertices, w_pixel, h_pixel, distance, roi, mm_per_px = a4_result

    if distance > 0:
        print(f"  距离: {distance:.0f}mm ({distance/10:.1f}cm)")
    else:
        print("  未标定，不显示距离")

    # ---- 图形检测（复刻 my-cv.py）----
    shapes = find_shapes_in_roi(roi, mm_per_px)

    shape_name = "unknown"
    shape_params = {}

    if shapes:
        shape_name, shape_params = shapes[0]
        print(f"  识别到: {shape_name}")
        for k, v in shape_params.items():
            print(f"    {k}={v:.0f}mm")

    # ---- 调试显示 ----
    if show_debug:
        frame_show = frame.copy()
        cv2.drawContours(frame_show, [vertices], -1, (0, 255, 0), 3)
        _imshow("A4 Detected", frame_show)
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not shape_params:
        return (False, None, None, "A4 found, no shape")

    return (True, shape_name, shape_params, "OK")


# ============================================================
# 竞赛模式
# ============================================================
# ============================================================
# 距离检测工具 —— 在画板位拍一张，返回 detect_a4 测出的距离
# ============================================================

def _get_board_dist(sp, st, pan_deg):
    """
    转到 pan_deg，检测 A4，X 轴一步对准（最多 5 步），返回距离(mm)。
    返回 (distance_mm, 检测到标志)
        distance_mm=0 且 检测到=False → 没检测到 A4
        distance_mm>0 → 测距成功
    """
    try:
        tgt = pan_deg + PAN_ZERO_OFFSET
        send_parallel(sp, pvt_cmd(PAN_ID, tgt, INIT_SPEED_RPM, INIT_TORQUE_PCT),
                      st, pvt_cmd(TILT_ID, 0 + TILT_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT), 14)
        time.sleep(0.5)
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            print("  摄像头打开失败，跳过测距")
            cap.release()
            return 0, False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)
    except Exception as e:
        print(f"  测距异常(开摄像头): {e}，跳过")
        return 0, False

    try:
        for _ in range(15):
            cap.read()
            time.sleep(0.1)
        time.sleep(0.3)
        ret, frame = cap.read()
        if not ret:
            print("  拍照失败，跳过测距")
            return 0, False

        pan_angle = pan_deg
        found = False
        final_dist = 0

        for step in range(6):  # 首次检测 + 最多5次微调
            a4 = detect_a4(frame)
            if a4 is None:
                if step == 0:
                    print("  未检测到A4纸，跳过测距")
                    return 0, False
                print(f"  第{step}步: A4 丢失，保持位置")
                break

            vertices, w_pixel, h_pixel, distance, roi, mm_per_px = a4
            image_center_x = frame.shape[1] / 2
            cx = int(vertices[:, 0, 0].mean())
            offset_x = cx - image_center_x

            # 算距离（用 K1/K2）
            K1, K2 = load_calib()
            d = 0
            if K1 > 0 and K2 > 0:
                d = (K1 / w_pixel + K2 / h_pixel) / 2
            final_dist = d

            if abs(offset_x) <= 15 or step >= 5:
                print(f"  测距: {d:.0f}mm 偏移={offset_x:+.0f}px", end="")
                if d <= 0:
                    print("（无K1/K2标定）")
                elif d < DIST_MIN_MM or d > DIST_MAX_MM:
                    print(f"  ⚠ 超出范围({DIST_MIN_MM}~{DIST_MAX_MM}mm)")
                    return 0, True
                else:
                    print("")
                found = True
                break

            # 调 X 轴
            adjust = -offset_x * SERVO_ADJUST_GAIN
            pan_angle += adjust
            pan_angle = max(0, min(360, pan_angle))
            print(f"  第{step+1}步: 偏移{offset_x:+.0f}px → 调{adjust:+.2f}°")
            send_parallel(sp, pvt_cmd(PAN_ID, pan_angle + PAN_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT),
                          st, pvt_cmd(TILT_ID, 0 + TILT_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT), 14)
            time.sleep(0.2)
            ret, frame = cap.read()
            if not ret:
                break

        if found:
            return final_dist, True
        return 0, True
    except Exception as e:
        print(f"  测距异常: {e}，跳过")
        return 0, False
    finally:
        cap.release()


def scale_params(sparams, dist_mm):
    """根据实际距离缩放图形参数，返回新 dict"""
    if dist_mm <= 0:
        return sparams
    s = DISTANCE_MM / dist_mm
    if abs(s - 1.0) < 0.01:
        return sparams
    print(f"  距离补偿: DISTANCE_MM/DIST={DISTANCE_MM:.0f}/{dist_mm:.0f}={s:.3f}")
    return {k: v * s if isinstance(v, (int, float)) else v for k, v in sparams.items()}


def m1(sp, st):
    """
    基本1：回程精度测试
    激光时序：归零 → ON(测起点) → 等1s → OFF → 转-360° → ON(测回程) → 等1s → OFF → 归零
    """
    print("\n===== 基本1 回程 =====")
    home_motors(sp, st)
    laser_on()                          # ★ 点亮激光，显示起始点
    time.sleep(1)
    laser_off()                         # ★ 转动前关闭激光
    t = -360 + PAN_ZERO_OFFSET
    print(f"转 -360° → {t:.1f}°")
    send_parallel(sp, pvt_cmd(PAN_ID, t, INIT_SPEED_RPM, INIT_TORQUE_PCT),
                  st, pvt_cmd(TILT_ID, 0+TILT_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT), 14)
    time.sleep(2); time.sleep(1)
    laser_on()                          # ★ 回零后点亮，显示回程点
    time.sleep(1)
    laser_off()                         # ★ 关闭激光
    home_motors(sp, st)
    print(f"回零→ Pan:{motor_read_pos(sp,PAN_ID):+.2f}° Tilt:{motor_read_pos(st,TILT_ID):+.2f}°")
    print("===== 基本1 完成 =====\n")

def m2(sp, st):
    print(f"\n===== 基本2 正方形 {SQUARE_SIDE}mm =====")
    home_motors(sp, st)
    aligned_az, _, board_dist = servo_to_board(sp, st, 0)
    d = board_dist if board_dist > 0 else 800
    if board_dist <= 0:
        show_digit(0)
        print("  未检测到画板，按 800mm 硬画")
        time.sleep(0.5)
    sparams = scale_params({"side": SQUARE_SIDE}, d)
    draw_at_angle(sp, st, gen_square(sparams["side"]), "square", aligned_az - 1.0)
    home_motors(sp, st)
    print("===== 基本2 完成 =====\n")

def m3(sp, st):
    print(f"\n===== 基本3 三角形 {TRIANGLE_SIDE}mm =====")
    home_motors(sp, st)
    aligned_az, _, board_dist = servo_to_board(sp, st, 0)
    d = board_dist if board_dist > 0 else 800
    if board_dist <= 0:
        show_digit(0)
        print("  未检测到画板，按 800mm 硬画")
        time.sleep(0.5)
    sparams = scale_params({"side": TRIANGLE_SIDE}, d)
    draw_at_angle(sp, st, gen_tri(sparams["side"]), "triangle", aligned_az - 1.0)
    home_motors(sp, st)
    print("===== 基本3 完成 =====\n")

def m4(sp, st):
    print(f"\n===== 基本4 梯形 {TRAPEZOID_TOP}/{TRAPEZOID_BOTTOM}/{TRAPEZOID_HEIGHT}mm =====")
    home_motors(sp, st)
    aligned_az, _, board_dist = servo_to_board(sp, st, 0)
    d = board_dist if board_dist > 0 else 800
    if board_dist <= 0:
        show_digit(0)
        print("  未检测到画板，按 800mm 硬画")
        time.sleep(0.5)
    sparams = scale_params({"top": TRAPEZOID_TOP, "bottom": TRAPEZOID_BOTTOM, "height": TRAPEZOID_HEIGHT}, d)
    pts = gen_trap(sparams["top"], sparams["bottom"], sparams["height"])
    pts = rotate_points_90(pts)  # 竖起来画
    # 梯形：无偏移（仅沿画板方位角画）
    pan_shift = aligned_az
    tilt_shift = 0
    p0, t0 = xy_to_angle(pts[0][0], pts[0][1])
    send_parallel(sp,
        pv_cmd(PAN_ID, pan_shift + p0 + PAN_ZERO_OFFSET, MOVE_SPEED_RPM),
        st, pv_cmd(TILT_ID, tilt_shift + t0 + TILT_ZERO_OFFSET, MOVE_SPEED_RPM), 14)
    time.sleep(0.3)
    laser_on()
    time.sleep(0.3)
    for _ in range(DRAW_REPEAT):
        for x, y in pts:
            p, t = xy_to_angle(x, y)
            send_parallel(sp,
                pv_cmd(PAN_ID, pan_shift + p + PAN_ZERO_OFFSET, MOVE_SPEED_RPM),
                st, pv_cmd(TILT_ID, tilt_shift + t + TILT_ZERO_OFFSET, MOVE_SPEED_RPM), 14)
            time.sleep(STEP_DELAY_S)
    laser_off()
    home_motors(sp, st)
    print("===== 基本4 完成 =====\n")

def m5(sp, st):
    """
    发挥1：视觉识别→画图
    流程：
      1. 归零到画板(0°) → 停顿1s
      2. 航偏角转180°到背面(相机对准目标)
      3. 调用 camera_detect_shape() 识别图形
      4. 航偏角回0°
      5. 用识别到的图形和尺寸画图
    """
    print("\n===== 发挥1 视觉识别+画图 =====")

    # 1. 归零到画板
    print("归零到画板(0°)...")
    home_motors(sp, st); time.sleep(1.0)

    # 2. 转180°到背面
    target_pan = 180.0 + PAN_ZERO_OFFSET
    print("航偏角转180°(相机对准目标)...")
    send_parallel(sp, pvt_cmd(PAN_ID, target_pan, INIT_SPEED_RPM, INIT_TORQUE_PCT),
                  st, pvt_cmd(TILT_ID, 0+TILT_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT), 14)
    time.sleep(2.0)

    # 3. 摄像头检测（调用可复用函数）
    ok, stype, params, msg = camera_detect_shape(show_debug=False)
    if not ok:
        print(f"识别失败: {msg}")
        home_motors(sp, st)
        return

    # 4. 回0°，检测画板距离
    print("归零到画板...")
    home_motors(sp, st)
    d, _found = _get_board_dist(sp, st, 0)
    if not _found:
        print("  未检测到画板，按 800mm 硬画")
        d = 800
    sparams = scale_params(params, d)
    sparams = {k: v + SHAPE_DRAW_OFFSET_MM if isinstance(v, (int, float)) else v for k, v in sparams.items()}

    # 5. 画图（draw 内已含 laser_on/off）
    pts = shape_to_points(stype, sparams)
    if not pts:
        print(f"不支持的图形: {stype}")
        return
    draw(sp, st, pts, stype, tilt_shift=3.0)
    home_motors(sp, st)
    print("===== 发挥1 完成 =====\n")


def m6(sp, st):
    """
    发挥2：画板在 270° 方向
    流程：
      1. 归零(0°)
      2. 转180°到背面(相机识别图形)
      3. 转90°(画板位置) → 画图
      4. 归零
    """
    print("\n===== 发挥2 画板270° =====")
    home_motors(sp, st); time.sleep(1.0)

    # 转180°识别
    send_parallel(sp, pvt_cmd(PAN_ID, 180.0+PAN_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT),
                  st, pvt_cmd(TILT_ID, 0+TILT_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT), 14)
    time.sleep(2.0)

    ok, stype, params, msg = camera_detect_shape(show_debug=False)
    if not ok:
        print(f"识别失败: {msg}")
        home_motors(sp, st)
        return

    # 转到270°画板并测距画图
    print("转到270°画板位置并测距...")
    draw_angle = 270.0
    d, _found = _get_board_dist(sp, st, draw_angle)
    if not _found:
        print("  未检测到画板，按 800mm 硬画")
        d = 800
    sparams = scale_params(params, d)
    sparams = {k: v + SHAPE_DRAW_OFFSET_MM if isinstance(v, (int, float)) else v for k, v in sparams.items()}

    pts = shape_to_points(stype, sparams)
    if pts:
        draw_at_angle(sp, st, pts, stype, draw_angle, tilt_shift=3.0)
    home_motors(sp, st)
    print("===== 发挥2 完成 =====\n")


def m7(sp, st):
    """
    发挥3：画板在 90° 方向
    流程：
      1. 归零(0°)
      2. 转180°到背面(相机识别图形)
      3. 转270°(画板位置) → 画图
      4. 归零
    """
    print("\n===== 发挥3 画板90° =====")
    home_motors(sp, st); time.sleep(1.0)

    # 转180°识别
    send_parallel(sp, pvt_cmd(PAN_ID, 180.0+PAN_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT),
                  st, pvt_cmd(TILT_ID, 0+TILT_ZERO_OFFSET, INIT_SPEED_RPM, INIT_TORQUE_PCT), 14)
    time.sleep(2.0)

    ok, stype, params, msg = camera_detect_shape(show_debug=False)
    if not ok:
        print(f"识别失败: {msg}")
        home_motors(sp, st)
        return

    # 转到90°画板并测距画图
    print("转到90°画板位置并测距...")
    draw_angle = 90.0
    d, _found = _get_board_dist(sp, st, draw_angle)
    if not _found:
        print("  未检测到画板，按 800mm 硬画")
        d = 800
    sparams = scale_params(params, d)
    sparams = {k: v + SHAPE_DRAW_OFFSET_MM if isinstance(v, (int, float)) else v for k, v in sparams.items()}

    pts = shape_to_points(stype, sparams)
    if pts:
        draw_at_angle(sp, st, pts, stype, draw_angle, tilt_shift=3.0)
    home_motors(sp, st)
    print("===== 发挥3 完成 =====\n")


def m8(sp, st):
    """
    发挥4：360° 扫描 → 找画板+素材 → 闭环对准 → 画图。
    流程：
      1. 360° 扫描，找空白画板（无图形）和素材（有图形）
      2. 摄像头闭环对准画板中心
      3. 将素材的图形画到画板上
      4. 归零
    """
    print("\n===== 发挥4 360°扫描+画图 =====")

    # 1. 扫描
    board_az, material = scan_for_papers(sp, st)

    if board_az is None:
        print("未找到画板，终止")
        home_motors(sp, st)
        return

    if material is None:
        print("未找到素材，终止")
        home_motors(sp, st)
        return

    m_az, stype, sparams = material[:3]
    print(f"\n素材: {stype} @ {m_az:.0f}°")
    for k, v in sparams.items():
        print(f"  {k}={v:.1f}mm")

    # 2. 闭环对准画板
    aligned_az, _, board_distance = servo_to_board(sp, st, board_az)
    print(f"\n画板基准方位: {aligned_az:.1f}°")

    # 距离补偿：用 detect_a4 测出的画板实际距离修正 xy_to_angle 的固定 DISTANCE_MM
    # detect_a4 返回的 distance_mm 来自 calibration.json 的 K1/w_pixel
    print(f"  画板距离: {board_distance:.0f}mm", end="")
    if board_distance > 0 and (board_distance < DIST_MIN_MM or board_distance > DIST_MAX_MM):
        print(f"  ⚠ 超出有效范围({DIST_MIN_MM}~{DIST_MAX_MM}mm)，舍弃，不补偿")
        board_distance = 0
    elif board_distance > 0:
        print("")
    else:
        print("（无K1/K2标定）")

    if board_distance > 0:
        scale = DISTANCE_MM / board_distance
        print(f"  缩放系数: {scale:.3f}")
        if abs(scale - 1.0) > 0.01:
            print(f"  → 图形尺寸 × {scale:.3f}")
            sparams = {k: v * scale if isinstance(v, (int, float)) else v
                       for k, v in sparams.items()}
            for k, v in sparams.items():
                if isinstance(v, (int, float)):
                    print(f"    {k}={v:.1f}mm（缩放后）")
        else:
            print("  → 距离与DISTANCE_MM相近，不缩放")
    else:
        print("  → 无距离数据（K1/K2为0），不缩放")

    # 画图尺寸偏移
    sparams = {k: v + SHAPE_DRAW_OFFSET_MM if isinstance(v, (int, float)) else v for k, v in sparams.items()}

    # 3. 画图
    pts = shape_to_points(stype, sparams)
    if not pts:
        print(f"不支持的图形: {stype}")
        home_motors(sp, st)
        return

    draw_at_angle(sp, st, pts, stype, aligned_az, tilt_shift=3.0)

    # 4. 归零
    home_motors(sp, st)
    print("===== 发挥4 完成 =====\n")


def m21_camera_test():
    """
    摄像头调试模式（不控制云台）。
    连续循环显示画面和检测结果。
    按 Q 退出，按 C 标定（输入实际距离自动算K值）。
    """
    print("\n===== 摄像头调试模式 =====")
    print("  画面窗口按 Q=退出  C=标定")

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)
    for _ in range(WARMUP_FRAMES):
        cap.read()
        time.sleep(WARMUP_DELAY)
    time.sleep(WARMUP_SETTLE)

    _last_calib_print = ""
    a4 = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        a4 = detect_a4(frame)
        if a4:
            _v, _wp, _hp, _dist, _roi, _mpp = a4
            cv2.drawContours(frame, [_v], -1, (0, 255, 0), 2)
            cv2.putText(frame, f"W:{_wp:.0f} H:{_hp:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"dist:{_dist:.0f}mm  mpp:{_mpp:.3f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Q=quit  C=calibrate", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            _cp = f"{_wp:.1f}_{_hp:.1f}_{_dist:.0f}"
            if _cp != _last_calib_print:
                _last_calib_print = _cp
                print(f"\nA4  W:{_wp:.1f}px  H:{_hp:.1f}px  dist:{_dist:.0f}mm")
                print(f"     K1=D×{_wp:.1f}  K2=D×{_hp:.1f}")
        else:
            cv2.putText(frame, "no A4", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        _imshow("mode12 - camera debug", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and a4:
            try:
                d = float(input(f"\n>>> 输入实际距离(mm) [当前W={_wp:.1f}px]: "))
                if d <= 0:
                    print("  距离必须 > 0")
                    continue
                new_k1 = d * _wp
                new_k2 = d * _hp
                print(f"  新 K1 = {d:.0f} × {_wp:.1f} = {new_k1:.0f}")
                print(f"  新 K2 = {d:.0f} × {_hp:.1f} = {new_k2:.0f}")
                old_k1, old_k2 = load_calibration()
                choice = input("  覆盖(o) 平均(a) 取消(回车): ").strip().lower()
                if choice == 'o':
                    k1, k2 = new_k1, new_k2
                    print("  ✓ 覆盖")
                elif choice == 'a':
                    if old_k1 > 0 and old_k2 > 0:
                        k1 = (old_k1 + new_k1) / 2
                        k2 = (old_k2 + new_k2) / 2
                        print(f"  平均: K1={k1:.0f}  K2={k2:.0f}")
                    else:
                        print("  无旧标定数据，按覆盖")
                        k1, k2 = new_k1, new_k2
                else:
                    print("  取消")
                    continue
                with open(CALIB_FILE, 'w') as f:
                    json.dump({"K1": k1, "K2": k2, "count": 1}, f, indent=2)
                print(f"  ✓ 已写入 {CALIB_FILE}")
                _last_calib_print = ""
            except ValueError:
                print("  输入无效，请输入数字")
            except Exception as e:
                print(f"  标定失败: {e}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n===== 摄像头调试模式 退出 =====\n")


# ============================================================
# 闭环模式选择 — UI 函数
# ============================================================

_HAVE_UI = False
_seg_leds = None
_btn_mode = None
_btn_start = None


def init_ui():
    """初始化数码管和按钮，失败时自动降级"""
    global _HAVE_UI, _seg_leds, _btn_mode, _btn_start
    if not _HAVE_LASER:
        print("[UI] gpiozero 不可用，跳过")
        return
    try:
        from gpiozero import LED, Button
        pins = [SEG_A_PIN, SEG_B_PIN, SEG_C_PIN, SEG_D_PIN, SEG_E_PIN, SEG_F_PIN, SEG_G_PIN]
        leds = []
        for i, pin in enumerate(pins):
            try:
                leds.append(LED(pin, active_high=False))
            except Exception as e:
                print(f"  [UI] 段{chr(97+i)}(GPIO{pin}) 初始化失败: {e}")
        _seg_leds = leds
        if len(_seg_leds) < 7:
            print(f"  [UI] 警告：只有 {len(_seg_leds)}/7 段可用，数码管显示可能不完整")
        _btn_mode = Button(BTN_MODE_PIN, pull_up=True, bounce_time=0.05)
        _btn_start = Button(BTN_START_PIN, pull_up=True, bounce_time=0.05)
        _HAVE_UI = True
        print(f"[UI] 数码管({len(_seg_leds)}段) + 按钮已初始化")
    except Exception as e:
        print(f"[UI] 按钮初始化失败: {e}")


def show_digit(digit):
    """在数码管上显示一个数字或字母"""
    if not _HAVE_UI or _seg_leds is None:
        return
    pattern = DIGIT_PATTERNS.get(digit)
    if pattern is None:
        return
    for i, (led, val) in enumerate(zip(_seg_leds, pattern)):
        if val == 0:    # 低电平 = 点亮（共阳极）
            led.on()
        else:
            led.off()


def all_seg_off():
    """熄灭所有段"""
    if not _HAVE_UI or _seg_leds is None:
        return
    for led in _seg_leds:
        led.off()


def mode_to_digit(mode):
    """模式编号 → 数码管显示值"""
    if mode == 11:
        return 'A'
    elif mode == 12:
        return 'b'
    return mode  # 1~9


# ============================================================
# 模式调度
# ============================================================

MODE_DISPATCH = {
    1: m1, 2: m2, 3: m3, 4: m4, 5: m5,
    6: m6, 7: m7, 8: m8, 11: calibrate,
}


def execute_mode(sp, st, mode):
    """执行指定模式，执行完返回（不退出程序）"""
    if mode == 9:
        print("完成")
        return
    if mode == 12:
        m21_camera_test()
        return
    if not sp or not st:
        print(f"错误：模式 {mode} 需要串口但未打开")
        return
    fn = MODE_DISPATCH.get(mode)
    if fn:
        fn(sp, st)
    else:
        print(f"未知模式: {mode}")


def cleanup(sp, st):
    """统一清理：串口 + 激光 + GPIO"""
    if sp and st:
        laser_off()
        motor_idle(sp, PAN_ID)
        motor_idle(st, TILT_ID)
        sp.close()
        st.close()
    if _HAVE_LASER:
        try:
            _laser.close()
        except:
            pass
    if _HAVE_UI:
        try:
            all_seg_off()
            _btn_mode.close()
            _btn_start.close()
            for led in _seg_leds:
                led.close()
        except:
            pass
    print("完成")


# ============================================================
# 主程序
# ============================================================
def main():
    global PAN_ZERO_OFFSET, TILT_ZERO_OFFSET
    print("="*60 + "\n竞赛执行器（闭环模式选择）\n" + "="*60)

    # 加载零点偏移
    PAN_ZERO_OFFSET, TILT_ZERO_OFFSET = load_zero()
    if PAN_ZERO_OFFSET != 0 or TILT_ZERO_OFFSET != 0:
        print(f"零点偏移: Pan={PAN_ZERO_OFFSET:+.2f}° Tilt={TILT_ZERO_OFFSET:+.2f}°")

    # 初始化 UI（数码管 + 按钮）
    init_ui()

    # 打开串口（所有需要云台的模式共用，模式 12 用不到但不影响）
    sp = None
    st = None
    try:
        list_serial_ports()
        sp = open_serial(SERIAL_PORT_PAN, BAUDRATE, "[Pan]")
        st = open_serial(SERIAL_PORT_TILT, BAUDRATE, "[Tilt]")
        if not sp or not st:
            print("[串口] 未连接串口，需要串口的模式无法运行")
        else:
            init_motors(sp, st)
    except Exception as e:
        print(f"[串口] 初始化错误: {e}")

    # ---- 无 UI 硬件 → 无法选择模式，退出 ----
    if not _HAVE_UI:
        print("\n[错误] 未检测到数码管/按钮硬件，无法选择模式")
        print("  请检查 GPIO 连接和权限")
        cleanup(sp, st)
        return

    # ---- 闭环模式选择循环 ----
    mode_idx = 0
    show_digit(mode_to_digit(MODE_LIST[mode_idx]))
    print("\n按模式按钮切换模式，按开始按钮执行")
    print("（Ctrl+C 退出）\n")

    try:
        while True:
            # 模式切换按钮
            if _btn_mode.is_pressed:
                mode_idx = (mode_idx + 1) % len(MODE_LIST)
                show_digit(mode_to_digit(MODE_LIST[mode_idx]))
                time.sleep(0.3)  # 防抖

            # 开始按钮
            if _btn_start.is_pressed:
                time.sleep(0.05)
                if not _btn_start.is_pressed:
                    continue  # 抖动，不执行
                selected = MODE_LIST[mode_idx]
                print(f"\n{'='*50}")
                print(f"执行模式 {selected}")
                print(f"{'='*50}")
                all_seg_off()
                execute_mode(sp, st, selected)
                print(f"\n模式 {selected} 执行完毕，返回模式选择\n")
                show_digit(mode_to_digit(MODE_LIST[mode_idx]))
                time.sleep(0.5)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        cleanup(sp, st)

if __name__ == "__main__":
    main()
