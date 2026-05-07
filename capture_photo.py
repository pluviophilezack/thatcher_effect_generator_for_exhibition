import cv2
import numpy as np
import time
import subprocess
import os
from PIL import Image, ImageDraw, ImageFont


CAPTURE_WINDOW_NAME = "Face Alignment"
OUTPUT_FILE_PATH = "input_images/user_face.jpeg"
COUNTDOWN_SECONDS = 3
KEY_ESC = 27
KEY_SPACE = 32
KEY_F7_CODES = {63242}
KEY_F9_CODES = {63244}
KEY_LEFT_ARROW_CODES = {81, 2424832, 63234}
KEY_RIGHT_ARROW_CODES = {83, 2555904, 63235}
LEFT_BUTTON_COLOR = (52, 118, 212)  # #D47634 in BGR
RIGHT_BUTTON_COLOR = (72, 182, 238)  # #EEB648 in BGR
BUTTON_TEXT_COLOR = (0, 0, 0)
COUNTDOWN_COLOR = (72, 182, 238)  
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/RoundGothic.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]


def _is_key_pressed(key, expected):
    if isinstance(expected, int):
        return key == expected
    return key in expected


def _load_font(size):
    for font_path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_rounded_square(frame, x, y, size, color, radius):
    radius = max(1, min(radius, size // 2))
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.rounded_rectangle(
        (x, y, x + size, y + size),
        radius=radius,
        fill=(color[2], color[1], color[0]),
        outline=(0, 0, 0),
        width=2,
    )
    frame[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _draw_centered_text_block(draw, lines, left, top, size, font, color, line_gap):
    metrics = []
    total_height = 0
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        line_w = box[2] - box[0]
        line_h = box[3] - box[1]
        metrics.append((line, box, line_w, line_h))
        total_height += line_h
    total_height += line_gap * (len(lines) - 1)

    current_y = top + (size - total_height) // 2
    for line, box, line_w, line_h in metrics:
        line_x = left + (size - line_w) // 2
        line_y = current_y - box[1]
        draw.text((line_x, line_y), line, font=font, fill=color)
        current_y += line_h + line_gap


def _draw_button(frame, x, y, size, color, lines, font_size=30):
    corner_radius = max(8, size // 10)
    _draw_rounded_square(frame, x, y, size, color, corner_radius)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = _load_font(font_size)
    _draw_centered_text_block(
        draw,
        lines,
        x,
        y,
        size,
        font,
        (BUTTON_TEXT_COLOR[2], BUTTON_TEXT_COLOR[1], BUTTON_TEXT_COLOR[0]),
        line_gap=6,
    )
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _prepare_display_frame(frame, show_capture_hint=False):
    display_frame = frame.copy()
    h, w, _ = display_frame.shape

    center = (w // 2, h // 2)
    axes = (int(w * 0.22), int(h * 0.38))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    overlay = display_frame.copy()
    overlay[:] = (40, 40, 40)

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    alpha = 0.7
    background_part = cv2.addWeighted(display_frame, 1 - alpha, overlay, alpha, 0)
    display_frame = np.where(mask_3ch == 255, display_frame, background_part)

    for i in range(0, 360, 10):
        cv2.ellipse(display_frame, center, axes, 0, i, i + 5, RIGHT_BUTTON_COLOR, 2)

    if show_capture_hint:
        square_size = max(120, min(h, w) // 5)
        margin = 16
        hint_x = w - square_size - margin
        hint_y = margin
        display_frame = _draw_button(
            display_frame,
            hint_x,
            hint_y,
            square_size,
            RIGHT_BUTTON_COLOR,
            ["按右鍵", "拍照"],
            font_size=28,
        )

    return display_frame, center, axes


def _draw_countdown(frame, seconds_left, center, axes):
    countdown_text = str(seconds_left)
    text_size, _ = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
    text_x = center[0] - text_size[0] // 2
    text_y = max(50, center[1] - axes[1] - 20)
    cv2.putText(
        frame,
        countdown_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        COUNTDOWN_COLOR,
        4,
        cv2.LINE_AA,
    )


def _play_countdown_beep():
    print("\a", end="", flush=True)
    try:
        subprocess.Popen(
            ["afplay", "/System/Library/Sounds/Ping.aiff"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def _play_capture_beep():
    print("\a", end="", flush=True)
    try:
        subprocess.Popen(
            ["afplay", "/System/Library/Sounds/Glass.aiff"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def _play_button_press_beep():
    try:
        subprocess.Popen(
            ["afplay", "-t", "0.08", "/System/Library/Sounds/Basso.aiff"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def _draw_confirm_prompt(frame):
    preview = frame.copy()
    h, w, _ = preview.shape
    center_x, center_y = w // 2, h // 2
    head_half_w = int(w * 0.22)
    head_half_h = int(h * 0.38)

    square_size = max(120, min(h, w) // 5)
    top_y = max(10, center_y - head_half_h - square_size - 20)
    left_x = max(10, center_x - head_half_w - square_size - 20)
    right_x = min(w - square_size - 10, center_x + head_half_w + 20)
    preview = _draw_button(preview, left_x, top_y, square_size, LEFT_BUTTON_COLOR, ["再拍一張"])
    preview = _draw_button(preview, right_x, top_y, square_size, RIGHT_BUTTON_COLOR, ["確定使用", "此照片"])
    return preview


def capture_photo():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("error: Cannot activate camera")
        return False
    print("--- Camera is ready ---")
    print("- 按下 [F9] 或 [右鍵] 倒數 3 秒後拍照")
    print("- 拍照後按 [空白鍵] 或 [右鍵] 確認使用，按 [F7] 或 [左鍵] 重拍")
    print("- 按下 [Esc] 結束程式")

    while True:
        # preview setting
        ret, frame = camera.read()
        if not ret:
            print("error: Receive ret Fail")
            break

        # --- 修正左右顛倒 (Mirror Effect) ---
        # flipCode = 1 代表水平翻轉，讓預覽畫面像照鏡子一樣自然
        frame = cv2.flip(frame, 1)

        display_frame, _, _ = _prepare_display_frame(frame, show_capture_hint=True)

        # 顯示預覽
        cv2.imshow(CAPTURE_WINDOW_NAME, display_frame)
        
        # cv2.imshow('Camera preview', frame)
        
        # Listening for keyin
        key = cv2.waitKeyEx(1)
        if _is_key_pressed(key, KEY_ESC):
            break
        elif _is_key_pressed(key, KEY_F9_CODES) or _is_key_pressed(key, KEY_RIGHT_ARROW_CODES):
            captured_frame = None
            countdown_aborted = False
            esc_during_countdown = False

            for seconds_left in range(COUNTDOWN_SECONDS, 0, -1):
                _play_countdown_beep()
                second_start = time.monotonic()
                while time.monotonic() - second_start < 1.0:
                    ret, frame = camera.read()
                    if not ret:
                        print("error: Receive ret Fail")
                        countdown_aborted = True
                        break

                    frame = cv2.flip(frame, 1)
                    captured_frame = frame
                    countdown_frame, center, axes = _prepare_display_frame(frame)
                    _draw_countdown(countdown_frame, seconds_left, center, axes)
                    cv2.imshow(CAPTURE_WINDOW_NAME, countdown_frame)

                    countdown_key = cv2.waitKeyEx(1)
                    if _is_key_pressed(countdown_key, KEY_ESC):
                        countdown_aborted = True
                        esc_during_countdown = True
                        break

                if countdown_aborted:
                    break

            if countdown_aborted or captured_frame is None:
                if esc_during_countdown:
                    break
                continue
            _play_capture_beep()

            while True:
                cv2.imshow(CAPTURE_WINDOW_NAME, _draw_confirm_prompt(captured_frame))
                confirm_key = cv2.waitKeyEx(1)

                if _is_key_pressed(confirm_key, KEY_ESC):
                    camera.release()
                    cv2.destroyAllWindows()
                    return False

                if (
                    _is_key_pressed(confirm_key, KEY_SPACE)
                    or _is_key_pressed(confirm_key, KEY_F9_CODES)
                    or _is_key_pressed(confirm_key, KEY_RIGHT_ARROW_CODES)
                ):
                    if _is_key_pressed(confirm_key, KEY_F9_CODES):
                        _play_button_press_beep()
                    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
                    if not cv2.imwrite(OUTPUT_FILE_PATH, captured_frame):
                        print(f"error: Failed to save image to {OUTPUT_FILE_PATH}")
                        break
                    print(f"{OUTPUT_FILE_PATH} is already saved.")
                    print("開始處理照片...")
                    camera.release()
                    cv2.destroyAllWindows()
                    return OUTPUT_FILE_PATH

                if _is_key_pressed(confirm_key, KEY_F7_CODES) or _is_key_pressed(confirm_key, KEY_LEFT_ARROW_CODES):
                    if _is_key_pressed(confirm_key, KEY_F7_CODES):
                        _play_button_press_beep()
                    break

    camera.release()
    cv2.destroyAllWindows()
    return False

if __name__ == "__main__":
    capture_photo()
