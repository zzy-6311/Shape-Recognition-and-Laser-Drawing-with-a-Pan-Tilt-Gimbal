"""
calibrate.py — A4 纸测距标定工具
================================

【做什么】
  标定"像素尺寸 → 实际距离"的换算系数 K1/K2，写入 calibration.json。

  原理：摄像头焦距固定时，同一物体在画面中的像素尺寸与距离成反比。
        把标定物放在已知距离处，量出它的像素宽 w，则 K = 实际距离 × w。
        多组数据取平均得到 K1（按宽度）和 K2（按高度）。

  测距公式（项目内各脚本统一）：
        D = (K1 / w_pixel + K2 / h_pixel) / 2

【怎么用】
  1. python calibrate.py
  2. 把标定物正对摄像头，放在已知距离处
  3. 按 S，输入当前标定物到摄像头的实际距离(mm)，记录一组 K
  4. 换几个不同距离重复第 3 步；K 自动取平均，右下角表格显示历史记录与误差
  5. 按 Q 退出（数据已实时写入 calibration.json）
  按 R 清除全部记录并重新开始。

【⚠ 重要：K 值不能跨配置复用】
  K1/K2 与摄像头分辨率强相关。
    本脚本        CAMERA_ID=1 / 640×480
    work-pi.py    CAMERA_ID=0 / 1080×640
  两者的 K 值**不能通用**。给哪个程序标定，就要用哪个程序的摄像头编号和分辨率，
  否则算出来的距离是错的。

【检测方式与其它脚本不同，请注意】
  本脚本用 THRESH_BINARY_INV 找**暗色**轮廓（针对带黑边框的标定纸），
  而 my-cv.py / work-pi.py 用 THRESH_BINARY 找**白色**纸面区域。
  两者框出的物理范围可能不同（黑框外沿 vs 白面内沿），
  上机前请确认标定出来的 K 与主程序检测的是同一块区域。

【依赖】
  opencv-python、numpy
"""
import cv2
import numpy as np
import math
import json
import os

# === 标定物尺寸（仅作参考，本脚本用实际距离标定，不依赖物理尺寸）===
# 注意区分：整张 A4 纸是 210×297mm，而各脚本真正检测的是纸内白色区域，
# 项目里按 170×257mm 处理（见 my-cv.py / work-pi.py 的 A4_WIDTH_MM、A4_HEIGHT_MM）。
A4_PAPER_WIDTH_MM = 210
A4_PAPER_HEIGHT_MM = 297

# === 图像处理参数（面向标定场景，与 my-cv.py 不同，不要直接照搬）===
A4_GAUSSIAN_BLUR_SIZE = (3, 3)
A4_BINARY_THRESHOLD = 50     # 固定阈值 + INV：找出暗色轮廓
A4_MAX_THRESHOLD_VALUE = 255
A4_MIN_AREA = 3000           # 最小轮廓面积（标定时放宽一些更灵敏）
A4_CENTER_Y_RATIO = 0.7      # 只在画面中心 70% 区域搜索，排除背景干扰
A4_CENTER_X_RATIO = 0.7
A4_APPROX_EPSILON = 0.02

# === 摄像头参数 ===
CAMERA_ID = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# === 保存路径 ===
CALIB_FILE = "calibration.json"


def preprocess(image):
    """预处理：灰度化、高斯模糊、固定阈值二值化（反色）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, A4_GAUSSIAN_BLUR_SIZE, 0)
    _, binary = cv2.threshold(blurred, A4_BINARY_THRESHOLD, A4_MAX_THRESHOLD_VALUE, cv2.THRESH_BINARY_INV)
    return binary


def find_a4_contour(binary):
    """只在画面中心区域搜索A4纸的轮廓，返还四个顶点"""
    h, w = binary.shape

    # 创建中心区域掩码，只保留画面中间部分，排除背景干扰
    x_start = int(w * (1 - A4_CENTER_X_RATIO) / 2)
    x_end = int(w * (1 + A4_CENTER_X_RATIO) / 2)
    y_start = int(h * (1 - A4_CENTER_Y_RATIO) / 2)
    y_end = int(h * (1 + A4_CENTER_Y_RATIO) / 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y_start:y_end, x_start:x_end] = 255
    masked = cv2.bitwise_and(binary, mask)

    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 按面积降序，取最大轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]

    area = cv2.contourArea(cnt)
    if area < A4_MIN_AREA:
        return None

    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, A4_APPROX_EPSILON * perimeter, True)

    if len(approx) != 4:
        return None

    return approx


def get_pixel_size(vertices):
    """计算A4纸四条边的平均像素宽和像素高"""
    pts = vertices.reshape(4, 2)

    w_top = math.sqrt((pts[0][0] - pts[1][0]) ** 2 + (pts[0][1] - pts[1][1]) ** 2)
    w_bottom = math.sqrt((pts[2][0] - pts[3][0]) ** 2 + (pts[2][1] - pts[3][1]) ** 2)
    h_left = math.sqrt((pts[1][0] - pts[2][0]) ** 2 + (pts[1][1] - pts[2][1]) ** 2)
    h_right = math.sqrt((pts[3][0] - pts[0][0]) ** 2 + (pts[3][1] - pts[0][1]) ** 2)

    w_pixel = (w_top + w_bottom) / 2
    h_pixel = (h_left + h_right) / 2

    return w_pixel, h_pixel


def save_calibration(records):
    """保存所有标定记录和平均值到文件"""
    avg_k1 = sum(r["K1"] for r in records) / len(records)
    avg_k2 = sum(r["K2"] for r in records) / len(records)
    data = {
        "K1": avg_k1,
        "K2": avg_k2,
        "records": records,
        "count": len(records)
    }
    with open(CALIB_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n标定数据已保存到 {CALIB_FILE}（共 {len(records)} 组，取平均值）")


def load_calibration():
    """从文件读取标定结果和记录"""
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, "r") as f:
            data = json.load(f)
            return data.get("K1", 0), data.get("K2", 0), data.get("records", [])
    return 0, 0, []


def main():
    K1, K2, records = load_calibration()
    if K1 > 0 and K2 > 0:
        print(f"已加载标定数据：K1={K1:.2f}, K2={K2:.2f}（{len(records)} 组记录）")
    else:
        print("未找到标定数据，请进行标定")
        records = []

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("\n=== A4纸测距标定工具 ===")
    print("操作说明：")
    print("  将A4纸正对摄像头，按 S 键标定当前距离")
    print("  按 R 键：清除所有标定记录重新开始")
    print("  按 Q 键：退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        binary = preprocess(frame)
        vertices = find_a4_contour(binary)

        if vertices is not None:
            cv2.drawContours(display, [vertices], -1, (0, 255, 0), 3)

            for pt in vertices.reshape(4, 2):
                cv2.circle(display, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)

            w_pixel, h_pixel = get_pixel_size(vertices)
            cv2.putText(display, f"W: {w_pixel:.0f}px  H: {h_pixel:.0f}px",
                       (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 显示当前标定信息
            cv2.putText(display, f"Records: {len(records)}  S:calibrate  R:reset  Q:quit",
                       (30, FRAME_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # 如果有标定数据，显示测距结果
            if K1 > 0 and K2 > 0:
                distance = (K1 / w_pixel + K2 / h_pixel) / 2
                cv2.putText(display, f"Distance: {distance:.0f}mm ({distance/10:.1f}cm)",
                           (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 在画面右上角显示标定记录表
            if records:
                y = 40
                cv2.putText(display, "--- Calibration Records ---", (FRAME_WIDTH - 420, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                y += 25
                cv2.putText(display, f"{'No':>3}  {'Actual':>7}  {'Pred':>7}  {'Err%':>7}",
                           (FRAME_WIDTH - 420, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                for i, rec in enumerate(records):
                    y += 20
                    pred = rec.get("predicted_dist", 0)
                    if pred > 0 and i > 0:
                        err_pct = (rec["dist"] - pred) / rec["dist"] * 100
                        err_str = f"{err_pct:+6.1f}%"
                    else:
                        err_str = "    -"
                    cv2.putText(display,
                               f"{i+1:>3}  {rec['dist']:>7.0f}  {pred:>7.0f}  {err_str:>7}",
                               (FRAME_WIDTH - 420, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                    if y > FRAME_HEIGHT - 80:
                        break
                # 显示平均值
                y += 25
                avg_k1 = sum(r["K1"] for r in records) / len(records)
                avg_k2 = sum(r["K2"] for r in records) / len(records)
                cv2.putText(display, f"Avg K1={avg_k1:.1f}  K2={avg_k2:.1f}",
                           (FRAME_WIDTH - 420, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Calibrate", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            records = []
            K1, K2 = 0, 0
            if os.path.exists(CALIB_FILE):
                os.remove(CALIB_FILE)
            print("\n已清除所有标定记录")
        elif key == ord('s') and vertices is not None:
            try:
                dist_mm = float(input(f"\n请输入当前A4纸到摄像头的距离(mm) [当前记录数:{len(records)}]: "))
                if dist_mm <= 0:
                    print("距离必须大于0")
                    continue

                w_pixel, h_pixel = get_pixel_size(vertices)

                # 用旧平均值预测当前距离（仅当已有记录时）
                predicted_dist = 0
                if K1 > 0 and K2 > 0:
                    predicted_dist = (K1 / w_pixel + K2 / h_pixel) / 2

                # 用实际距离计算本次K值
                k1 = dist_mm * w_pixel
                k2 = dist_mm * h_pixel

                # 加入记录（包含预测值和实际值的对比）
                records.append({
                    "dist": dist_mm,
                    "w_pixel": w_pixel,
                    "h_pixel": h_pixel,
                    "K1": k1,
                    "K2": k2,
                    "predicted_dist": predicted_dist
                })

                # 重新计算平均值作为最终K值
                K1 = sum(r["K1"] for r in records) / len(records)
                K2 = sum(r["K2"] for r in records) / len(records)

                # 打印本次和汇总
                print(f"\n第 {len(records)} 组：实际距离={dist_mm:.0f}mm, W={w_pixel:.1f}px, H={h_pixel:.1f}px")
                if predicted_dist > 0:
                    error = dist_mm - predicted_dist
                    print(f"  旧平均预测={predicted_dist:.0f}mm, 误差={error:+.0f}mm ({error/dist_mm*100:+.1f}%)")
                print(f"  本次 K1={k1:.2f}, K2={k2:.2f}")
                print(f"  平均 K1={K1:.2f}, K2={K2:.2f}")

                save_calibration(records)

            except ValueError:
                print("输入无效，请输入数字")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
