"""
my-cv-one.py — 单张图片离线调参工具
===================================

【做什么】
  对 sucai/ 目录下的一张固定照片跑一遍完整的"找 A4 纸 → 识别纸内图形"流程，
  并把中间的轮廓判定过程打印出来、把关键二值图弹窗显示。

【为什么需要它】
  举着 A4 纸对着摄像头调参数时，改一个阈值就要重新摆一次纸，无法复现。
  本脚本用一张固定照片反复跑，改一次参数看一次结果，便于快速对比。

  它复制了 my-cv.py 的全部检测函数，但针对静态图片做了两点调整：
    · 参数按照片重新调过（亮度阈值更低、直角容差更宽，见下面参数区注释）
    · 补了大量 "[调试]" 打印，会说明"最大轮廓为什么被跳过"

【和 my-cv.py 的分工】
  my-cv.py       接摄像头实时跑，参数面向现场光照
  my-cv-one.py   读固定照片跑，参数面向该照片，调试输出更详细

【怎么用】
  1. 把要测的图片放进同目录的 sucai/
  2. 修改下面的 IMAGE_NAME 指向该文件名
  3. python my-cv-one.py   （按任意键关闭窗口）

【依赖】
  opencv-python、numpy；calibration.json 可选（缺失时只识别不测距）
"""
import cv2
import math
import numpy as np
import json
import os
import re

# 要识别的素材文件名（位于同目录的 sucai/ 下）
IMAGE_NAME = "9.jpg"

# ============================================================
# 外层 A4 纸检测参数
# ============================================================
A4_GAUSSIAN_BLUR_SIZE = (3, 3)
A4_MIN_AREA = 6500
A4_CENTER_Y_RATIO = 1
A4_CENTER_X_RATIO = 1
A4_APPROX_EPSILON = 0.02
A4_ANGLE_TOLERANCE = 15
A4_BRIGHT_THRESH = 130      # 白色区域二值化阈值，高于此值为白（目标）

# ============================================================
# 内部图形检测参数
# ============================================================
SHAPE_GAUSSIAN_BLUR_SIZE = (3, 3)
SHAPE_MIN_AREA = 1000
SHAPE_APPROX_EPSILON = 0.02
SHAPE_CIRCULARITY_THRESHOLD = 0.7
SHAPE_ANGLE_TOLERANCE = 5

# ============================================================
# 测距参数（白色区域实际物理尺寸，根据你的白框量了改）
# ============================================================
A4_WIDTH_MM = 170          # 白色区域宽度（mm），原黑框A4纸约185mm
A4_HEIGHT_MM = 257         # 白色区域高度（mm），原黑框A4纸约270mm
CALIB_FILE = "calibration.json"


# ============================================================
# 辅助函数
# ============================================================
def angle_between(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    m = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
    if m == 0: return 0
    return math.degrees(math.acos(dot / m))


def classify_shape(approx):
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
        if any(a > 170 for a in angles):
            return ("triangle", 3)
        is_square = all(abs(a - 90) < SHAPE_ANGLE_TOLERANCE for a in angles)
        return ("square", n) if is_square else ("trapezoid", n)
    elif n >= 5:
        area = cv2.contourArea(approx)
        peri = cv2.arcLength(approx, True)
        if peri > 0:
            circularity = 4 * math.pi * area / (peri * peri)
            if circularity > SHAPE_CIRCULARITY_THRESHOLD:
                return ("circle", n)
        return ("unknown", n)
    return ("unknown", n)


def measure_shape(approx, name, mm_per_px):
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
            h_px = cross / math.sqrt(v[0][0]**2+v[0][1]**2)
            height = h_px * mm_per_px
        elif is_par(v[1], v[3]):
            x1 = (pts[1][0] + pts[2][0]) / 2
            x3 = (pts[3][0] + pts[0][0]) / 2
            top_len = s[1] if x1 < x3 else s[3]
            bot_len = s[3] if x1 < x3 else s[1]
            top = top_len * mm_per_px
            bottom = bot_len * mm_per_px
            cross = abs(v[2][0]*v[1][1] - v[2][1]*v[1][0])
            h_px = cross / math.sqrt(v[1][0]**2+v[1][1]**2)
            height = h_px * mm_per_px
        else:
            return "trap?"
        return f"top={top:.0f} bot={bottom:.0f} h={height:.0f}mm"
    return ""


def get_pixel_size(vertices):
    pts = vertices.reshape(4, 2)
    w1 = math.sqrt((pts[0][0]-pts[1][0])**2 + (pts[0][1]-pts[1][1])**2)
    w2 = math.sqrt((pts[2][0]-pts[3][0])**2 + (pts[2][1]-pts[3][1])**2)
    h1 = math.sqrt((pts[1][0]-pts[2][0])**2 + (pts[1][1]-pts[2][1])**2)
    h2 = math.sqrt((pts[3][0]-pts[0][0])**2 + (pts[3][1]-pts[0][1])**2)
    return (w1 + w2) / 2, (h1 + h2) / 2


def load_calibration():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE) as f:
            data = json.load(f)
            return data.get("K1", 0), data.get("K2", 0)
    return 0, 0


# ============================================================
# 检测函数
# ============================================================

def detect_a4(frame):
    """
    检测画面中所有白色矩形区域（一张或多张 A4 纸的白面）。
    返回 [(vertices, w_pixel, h_pixel, distance_mm, roi, mm_per_px), ...] 或 []
    说明：A4_WIDTH_MM / A4_HEIGHT_MM 应设为白色区域的实际物理尺寸。
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, A4_GAUSSIAN_BLUR_SIZE, 0)

    # 亮二值：白色纸面 → 白色，黑色背景/干扰 → 黑色
    # 白色纸在 dark 背景上是单一外轮廓，纸内图案=内部孔洞，不干扰外轮廓
    _, bright_bin = cv2.threshold(blur, A4_BRIGHT_THRESH, 255, cv2.THRESH_BINARY)

    # cv2.imshow("bright_bin (白色目标)", bright_bin)

    y_min = int(h * (1 - A4_CENTER_Y_RATIO) / 2)
    y_max = int(h * (1 + A4_CENTER_Y_RATIO) / 2)
    x_min = int(w * (1 - A4_CENTER_X_RATIO) / 2)
    x_max = int(w * (1 + A4_CENTER_X_RATIO) / 2)

    # 在亮二值中找白色外轮廓（RETR_EXTERNAL=忽略内部孔洞），按面积排序
    contours, _ = cv2.findContours(bright_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    print(f"  [调试] 白色轮廓数: {len(contours)}", end="")
    if contours:
        print(f"  最大面积: {cv2.contourArea(contours[0]):.0f}px")
    else:
        print("")

    K1, K2 = load_calibration()
    a4_list = []

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < A4_MIN_AREA:
            if idx == 0: print(f"  [调试] 最大面积 {area:.0f} < {A4_MIN_AREA}，跳过")
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
            if idx == 0: print(f"  [调试] 最大轮廓质心({cx},{cy})不在画面中心，跳过")
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, A4_APPROX_EPSILON * peri, True)
        if len(approx) != 4:
            continue

        # 矩形验证
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

        a4_list.append((approx, w_pixel, h_pixel, distance, roi, mm_per_px))

    print(f"  [调试] 合格 A4 候选: {len(a4_list)}")
    return a4_list


def find_shapes(roi, mm_per_px):
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_blur = cv2.GaussianBlur(roi_gray, SHAPE_GAUSSIAN_BLUR_SIZE, 0)
    # 自适应阈值：根据局部亮度算阈值，不受光照变化影响
    # blockSize=31 保证在大面积黑色图形内部也能包含纸面白色像素
    roi_binary = cv2.adaptiveThreshold(roi_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 5)
    # 如果自适应阈值检测不到（大面积黑色图形），改用 OTSU
    if cv2.countNonZero(roi_binary) < 50:
        _, roi_binary = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow("shape_binary", roi_binary)

    # 中心掩膜排除 A4 纸黑边框
    roi_h, roi_w = roi.shape[:2]
    inset = 20
    center_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    center_mask[inset:roi_h - inset, inset:roi_w - inset] = 255
    roi_binary = cv2.bitwise_and(roi_binary, center_mask)

    roi_contours, _ = cv2.findContours(roi_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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


# ============================================================
# 主程序：识别单张图片
# ============================================================

def main():
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sucai", IMAGE_NAME)
    print(f"读取图片: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        print("✗ 图片读取失败")
        return

    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    # ---- 1. 检测所有 A4 纸 ----
    a4_list = detect_a4(img)
    if not a4_list:
        print("✗ 未检测到 A4 纸")
        cv2.imshow("img", img)
        cv2.waitKey(0)
        return

    debug = img.copy()
    print(f"\n✓ 检测到 {len(a4_list)} 张 A4 纸")
    for n, (vertices, w_pixel, h_pixel, distance, roi, mm_per_px) in enumerate(a4_list):
        shapes = find_shapes(roi, mm_per_px)
        if shapes:
            name, info_str, _, _ = shapes[0]
            print(f"\n  --- A4 #{n+1} --- {name} {info_str}")
            for sn, (sname, sinfo, approx, _) in enumerate(shapes):
                cv2.drawContours(roi, [approx], -1, (0, 0, 255), 2)
        else:
            name = "board"
            print(f"\n  --- A4 #{n+1} --- board")
        print(f"  {w_pixel:.0f}x{h_pixel:.0f}px  dist:{distance:.0f}mm")
        cv2.drawContours(debug, [vertices], -1, (0, 255, 0), 2)

        if len(a4_list) == 1:
            # 单张 → 左上角显示
            cv2.putText(debug, name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # 多张同框 → 在每个纸中心附近用小字显示
            cx = int(vertices[:, 0, 0].mean())
            cy = int(vertices[:, 0, 1].mean())
            cv2.putText(debug, name, (cx-30, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---- 2. 显示 ----
    cv2.imshow("A4", debug)
    # 显示每个检测到的 A4 纸 ROI
    for n, (_, _, _, _, roi, _) in enumerate(a4_list):
        if roi is not None:
            cv2.imshow(f"ROI_{(n+1)}", roi)
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
