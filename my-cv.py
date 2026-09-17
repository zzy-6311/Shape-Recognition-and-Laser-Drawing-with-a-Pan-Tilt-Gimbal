"""
my-cv.py — A4 纸图形识别（视觉算法母本 + 摄像头实时调试）
=========================================================

【做什么】
  从摄像头画面里找出 A4 纸（白色矩形），再识别纸上的图形（正方形 / 三角形 /
  梯形 / 圆形），并估算图形的实际物理尺寸(mm)以及纸到摄像头的距离。

  这是整个项目的视觉算法母本。work-pi.py 里的检测函数就是从本文件复制过去、
  再把返回类型从"给人看的字符串"改成"给机器算的字典"而来。改视觉算法时建议
  先在本文件调好，再同步到 work-pi.py，避免两边行为不一致。

【识别流程】
  1. detect_a4(frame)
       灰度 → 高斯模糊 → 按亮度阈值二值化（白纸=白）
       → RETR_TREE 取偶数层轮廓（排除纸内图案形成的"洞"）
       → 四边形逼近 + 四角直角校验 + 宽高比校验 → 得到 A4 轮廓与 ROI
  2. find_shapes(roi, mm_per_px)
       在 ROI 内自适应阈值（检测不到则回退 OTSU）
       → 中心掩膜排除纸边框 → 逐个轮廓逼近 → classify_shape 分类
       → measure_shape 换算实际尺寸
  3. process_frame(frame)  把上面两步串起来，供外部一次调用

【对外接口】
  detect_a4(frame)       -> [(vertices, w_px, h_px, distance, roi, mm_per_px), ...]
                            注意返回的是**列表**：一张画面里可以有多张 A4 纸
  find_shapes(roi, mpp)  -> [(图形名, 尺寸字符串, 轮廓, 顶点数), ...]
  process_frame(frame)   -> (ok, 图形名, 尺寸字符串, 消息, a4_info)

【测距原理】
  距离 D = (K1 / w_pixel + K2 / h_pixel) / 2，K1、K2 由 calibrate.py 标定后
  写入 calibration.json（K = 实际距离 × 像素宽）。
  ⚠ K1/K2 与摄像头分辨率强相关：换摄像头或换分辨率必须重新标定，否则算出的
    距离是错的。本文件按 CAMERA_ID=0 / 1080×640 工作。

【怎么用】
  直接运行本文件 → 打开摄像头实时检测，结果打印在终端，画面里画出 A4 外框，
  按 Q 退出。若想用固定图片反复调参数，请用 my-cv-one.py（更快且可复现）。

【依赖】
  opencv-python、numpy；calibration.json 可选（缺失时只识别不测距）
"""
import cv2
import math
import numpy as np
import json
import os
import re

# ============================================================
# 摄像头参数（main() 用）
# ============================================================
CAMERA_ID = 0
FRAME_WIDTH = 1080
FRAME_HEIGHT = 640


# 显示辅助：统一封装 cv2.imshow，方便以后替换成缩放显示
def _imshow(name, img):
    cv2.imshow(name, img)


# ============================================================
# 外层 A4 纸检测参数
# ============================================================
A4_GAUSSIAN_BLUR_SIZE = (3, 3)
A4_BRIGHT_THRESH = 160         # 白色纸面二值化阈值，高于此值为白
A4_MIN_AREA = 10000
A4_CENTER_Y_RATIO = 1
A4_CENTER_X_RATIO = 1
A4_APPROX_EPSILON = 0.02
A4_ANGLE_TOLERANCE = 2

# ============================================================
# 内部图形检测参数
# ============================================================
SHAPE_GAUSSIAN_BLUR_SIZE = (3, 3)
SHAPE_MIN_AREA = 1000
SHAPE_APPROX_EPSILON = 0.02
SHAPE_CIRCULARITY_THRESHOLD = 0.7
SHAPE_ANGLE_TOLERANCE = 5

# ============================================================
# 测距参数（白色区域实际物理尺寸）
# ============================================================
A4_WIDTH_MM = 170
A4_HEIGHT_MM = 257
CALIB_FILE = "calibration.json"


# ============================================================
# 辅助函数
# ============================================================
def angle_between(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    m = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
    if m == 0:
        return 0
    return math.degrees(math.acos(dot / m))


def classify_shape(approx):
    """按顶点数和圆形度分类图形"""
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
    """
    根据图形类型计算实际尺寸（mm）。
    返回格式化的尺寸字符串。
    """
    pts = approx.reshape(len(approx), 2)

    if name == "circle":
        (_, _), radius = cv2.minEnclosingCircle(approx)
        d = 2 * radius * mm_per_px
        return f"D={d:.0f}mm"

    elif name == "square":
        s1 = math.sqrt((pts[0][0]-pts[1][0])**2 + (pts[0][1]-pts[1][1])**2) * mm_per_px
        s2 = math.sqrt((pts[1][0]-pts[2][0])**2 + (pts[1][1]-pts[2][1])**2) * mm_per_px
        s3 = math.sqrt((pts[2][0]-pts[3][0])**2 + (pts[2][1]-pts[3][1])**2) * mm_per_px
        s4 = math.sqrt((pts[3][0]-pts[0][0])**2 + (pts[3][1]-pts[0][1])**2) * mm_per_px
        avg = (s1 + s2 + s3 + s4) / 4
        return f"side={avg:.0f}mm"

    elif name == "triangle":
        s = []
        for i in range(3):
            s.append(math.sqrt((pts[i][0]-pts[(i+1)%3][0])**2 + (pts[i][1]-pts[(i+1)%3][1])**2) * mm_per_px)
        avg = sum(s) / 3
        return f"side={avg:.0f}mm"

    elif name == "trapezoid":
        s = [math.sqrt((pts[i][0]-pts[(i+1)%4][0])**2 + (pts[i][1]-pts[(i+1)%4][1])**2) for i in range(4)]
        v = [pts[(i+1)%4] - pts[i] for i in range(4)]

        def is_par(v1, v2):
            cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
            norm = math.sqrt(v1[0]**2+v1[1]**2) * math.sqrt(v2[0]**2+v2[1]**2)
            return norm > 0 and cross / norm < 0.2

        if is_par(v[0], v[2]):
            y0 = (pts[0][1] + pts[1][1]) / 2
            y2 = (pts[2][1] + pts[3][1]) / 2
            top_len = s[0] if y0 < y2 else s[2]
            bot_len = s[2] if y0 < y2 else s[0]
            top = top_len * mm_per_px
            bottom = bot_len * mm_per_px
            cross = abs(v[1][0]*v[0][1] - v[1][1]*v[0][0])
            h_px = cross / math.sqrt(v[0][0]**2+v[0][1]**2) if math.sqrt(v[0][0]**2+v[0][1]**2) > 0 else 0
            height = h_px * mm_per_px
        elif is_par(v[1], v[3]):
            x1 = (pts[1][0] + pts[2][0]) / 2
            x3 = (pts[3][0] + pts[0][0]) / 2
            top_len = s[1] if x1 < x3 else s[3]
            bot_len = s[3] if x1 < x3 else s[1]
            top = top_len * mm_per_px
            bottom = bot_len * mm_per_px
            cross = abs(v[2][0]*v[1][1] - v[2][1]*v[1][0])
            h_px = cross / math.sqrt(v[1][0]**2+v[1][1]**2) if math.sqrt(v[1][0]**2+v[1][1]**2) > 0 else 0
            height = h_px * mm_per_px
        else:
            return "trap?"
        return f"top={top:.0f} bot={bottom:.0f} h={height:.0f}mm"

    return ""


def get_pixel_size(vertices):
    """计算 A4 纸四个顶点的平均像素宽和高"""
    pts = vertices.reshape(4, 2)
    w1 = math.sqrt((pts[0][0]-pts[1][0])**2 + (pts[0][1]-pts[1][1])**2)
    w2 = math.sqrt((pts[2][0]-pts[3][0])**2 + (pts[2][1]-pts[3][1])**2)
    h1 = math.sqrt((pts[1][0]-pts[2][0])**2 + (pts[1][1]-pts[2][1])**2)
    h2 = math.sqrt((pts[3][0]-pts[0][0])**2 + (pts[3][1]-pts[0][1])**2)
    return (w1 + w2) / 2, (h1 + h2) / 2


def load_calibration():
    """加载标定文件，返回 K1, K2"""
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE) as f:
            data = json.load(f)
            return data.get("K1", 0), data.get("K2", 0)
    return 0, 0


# ============================================================
# 封装函数 —— 供外部调用
# ============================================================

def detect_a4(frame):
    """
    检测画面中所有白色矩形区域（一张或多张 A4 纸的白面）。

    参数:
        frame: BGR 图像 (numpy array)

    返回:
        [(vertices, w_pixel, h_pixel, distance_mm, roi, mm_per_px), ...] 或 []
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, A4_GAUSSIAN_BLUR_SIZE, 0)

    # 亮二值：白色纸面 → 白色，白纸是单一外轮廓，纸内图案=内部孔洞不干扰
    _, bright_bin = cv2.threshold(blur, A4_BRIGHT_THRESH, 255, cv2.THRESH_BINARY)
    _imshow("A4_binary", bright_bin)

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
    _imshow("A4_contours", frame)

    K1, K2 = load_calibration()
    a4_list = []

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
        _aspect = max(w_pixel, h_pixel) / min(w_pixel, h_pixel)
        target_aspect = A4_HEIGHT_MM / A4_WIDTH_MM
        if abs(_aspect - target_aspect) > 0.15:
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

        a4_list.append((approx, w_pixel, h_pixel, distance, roi, mm_per_px))
        print(f"  ✓ 找到A4: {w_pixel:.0f}×{h_pixel:.0f}px 距离={distance:.0f}mm")

    if not a4_list:
        print(f"  → 未找到A4纸（共{len(contours)}个白色轮廓）")
    return a4_list


def find_shapes(roi, mm_per_px, debug=True):
    """
    在 A4 纸 ROI 中检测内部图形。
    debug=True 时显示二值化图和轮廓检测图。
    """
    roi_h, roi_w = roi.shape[:2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_blur = cv2.GaussianBlur(roi_gray, SHAPE_GAUSSIAN_BLUR_SIZE, 0)

    # ---- 自适应阈值 ----
    roi_binary = cv2.adaptiveThreshold(roi_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 5)

    # ---- OTSU 回退 ----
    if cv2.countNonZero(roi_binary) < 50:
        _, roi_binary = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # ---- 中心掩膜 ----
    inset = 20
    center_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    center_mask[inset:roi_h - inset, inset:roi_w - inset] = 255
    roi_binary_masked = cv2.bitwise_and(roi_binary, center_mask)

    # ---- 找轮廓 ----
    roi_contours, _ = cv2.findContours(roi_binary_masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # ========== 调试可视化 ==========
    if debug:
        cv2.imshow("binary", roi_binary_masked)
        contour_debug = roi.copy()
        cv2.drawContours(contour_debug, roi_contours, -1, (0, 255, 0), 2)
        for i, c in enumerate(roi_contours):
            ra = cv2.contourArea(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(contour_debug, f"#{i}A={ra:.0f}", (cx-25, cy+2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow("contours", contour_debug)

    # ---- 逐个轮廓分析 ----
    results = []
    roi_area = roi_h * roi_w
    for rc in roi_contours:
        ra = cv2.contourArea(rc)
        if ra < SHAPE_MIN_AREA or ra > roi_area * 0.6:
            continue

        rp = cv2.arcLength(rc, True)
        rap = cv2.approxPolyDP(rc, SHAPE_APPROX_EPSILON * rp, True)
        if len(rap) == 4:
            for _mult in (1.5, 2.0, 3.0):
                rap2 = cv2.approxPolyDP(rc, _mult * SHAPE_APPROX_EPSILON * rp, True)
                if len(rap2) == 3:
                    rap = rap2
                    break
        name, verts = classify_shape(rap)
        params = measure_shape(rap, name, mm_per_px)
        if params:
            # 过滤尺寸接近 A4 纸(>150mm)的轮廓——那是纸张边框残留，不是纸内图形。
            # measure_shape 返回的是 "side=80mm" 这类字符串，所以取出其中的数值来判断。
            dims = [float(v) for v in re.findall(r"-?\d+\.?\d*", params)]
            if any(abs(d) > 150 for d in dims):
                continue
            results.append((name, params, rap, verts))

    return results


def process_frame(frame):
    """
    完整处理一帧：检测 A4 纸 → 识别内部图形 → 返回第一个结果。

    参数:
        frame: BGR 图像

    返回:
        (ok, shape_name, shape_info_str, message, a4_info)
        a4_info = (vertices, distance_mm) 或 None
    """
    a4_list = detect_a4(frame)
    if not a4_list:
        return (False, None, None, "未检测到A4纸", None)

    # 只处理第一个（最大）A4 纸
    vertices, w_pixel, h_pixel, distance, roi, mm_per_px = a4_list[0]
    shapes = find_shapes(roi, mm_per_px)

    if not shapes:
        return (False, None, None, "A4 found, no shape", (vertices, distance))

    shape_name, shape_str, _, _ = shapes[0]
    return (True, shape_name, shape_str, "OK", (vertices, distance))


# ============================================================
# 主程序：循环显示摄像头画面，实时检测
# ============================================================
def main():
    K1, K2 = load_calibration()
    if K1 > 0 and K2 > 0:
        print(f"标定数据已加载 K1={K1:.1f} K2={K2:.1f}")
    else:
        print("未找到标定文件，不显示距离")

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("拍照失败")
            break

        ok, shape_name, shape_str, msg, a4_info = process_frame(frame)

        # 结果显示在终端（画面压缩后文字看不清）
        if ok:
            print(f"  {shape_name}: {shape_str}")
        elif a4_info:
            print(f"  A4已找到: {msg}")
        else:
            print(f"  {msg}")

        if a4_info:
            cv2.drawContours(frame, [a4_info[0]], -1, (0, 255, 0), 3)
        _imshow("my-cv", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
