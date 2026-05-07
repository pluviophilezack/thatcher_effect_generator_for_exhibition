import os
from os import listdir, mkdir
from os.path import isfile, join
from math import inf
import subprocess
from capture_photo import capture_photo


from facial_landmark_detection import get_image_facial_landmarks
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

INPUT_IMAGES_DIRECTORY_PATH = "input_images"
OUTPUT_IMAGES_DIRECTORY_PATH = "output_images"
OUTPUT_NONFLIP_IMAGE_PATH = "output_nonflip_images"
CAPTURED_IMAGE_FILENAME = "user_face.jpeg"
OUTPUT_CAPTURED_IMAGE_PATH = join(OUTPUT_IMAGES_DIRECTORY_PATH, CAPTURED_IMAGE_FILENAME)
OUTPUT_CAPTURED_NONFLIP_IMAGE_PATH = join(OUTPUT_NONFLIP_IMAGE_PATH, CAPTURED_IMAGE_FILENAME)
PROCESSED_PREVIEW_WINDOW_NAME = "Processed Result"
KEY_F9_CODES = {63244, 120, 0x780000}
KEY_F7_CODES = {63242, 118, 0x760000}
ROTATION_ANIMATION_FRAMES = 60
ROTATION_ANIMATION_DELAY_MS = 60
LEFT_BUTTON_COLOR = (52, 118, 212)  # #D47634 in BGR
RIGHT_BUTTON_COLOR = (72, 182, 238)  # #EEB648 in BGR
BUTTON_TEXT_COLOR = (0, 0, 0)
FONT_CANDIDATES = [
	"/System/Library/Fonts/Supplemental/RoundGothic.ttc",
	"/System/Library/Fonts/PingFang.ttc",
	"/System/Library/Fonts/STHeiti Light.ttc",
	"/Library/Fonts/Arial Unicode.ttf",
]
ATTRIBUTES_CSV_PATH = "attributes.csv"
ATTRIBUTES_CSV_DELIMITER = ","
ATTRIBUTES_CSV_MAX_ROWS = 1000
PRINT_LOG = True
PRINT_LOG_PERIOD = 1


def get_bounding_rectangle(points):
	top_left = [inf, inf]
	bottom_right = [-inf, -inf]
	for point in points:
		top_left[0] = min(top_left[0], point[1])
		top_left[1] = min(top_left[1], point[0])
		bottom_right[0] = max(bottom_right[0], point[1])
		bottom_right[1] = max(bottom_right[1], point[0])
	return [top_left, bottom_right]


def flip_subimage_vertically(image, x1, y1, x2, y2):
	mid_x = (x1 + x2) // 2
	for x in range(x1, mid_x):
		for y in range(y1, y2 + 1):
			image[x][y], image[x1 + x2 - x][y] = image[x1 + x2 - x][y].copy(), image[x][y].copy()


def flip_subimage_ellipse_vertically(image, x1, y1, x2, y2):
	mid_x = (x1 + x2) / 2.0
	mid_y = (y1 + y2) / 2.0
	b = (y2 - y1) / 2.0
	a = (x2 - x1) / 2.0
	for x in range(x1, x2 + 1):
		for y in range(y1, y2 + 1):
			dx = x - mid_x
			dy = y - mid_y
			if (dx * dx) / (a * a) + (dy * dy) / (b * b) <= 1 and x1 + x2 - x > x:
				image[x][y], image[x1 + x2 - x][y] = image[x1 + x2 - x][y].copy(), image[x][y].copy()


def gradient_subimage(image, x1, y1, x2, y2):
	final_distance = (x2 - x1) ** 2 + (y2 - y1) ** 2
	start_color = image[x1][y1].copy()
	final_color = image[x2][y2].copy()
	for x in range(x1, x2 + 1):
		for y in range(y1, y2 + 1):
			current_distance = (x - x1) ** 2 + (y - y1) ** 2
			k = current_distance / final_distance
			current_color = start_color * (1 - k) + final_color * k
			image[x][y] = current_color


def blur_ellipse_border(image, x1, y1, x2, y2):
	blurred_image = cv2.GaussianBlur(image, (5,5), 0)
	mid_x = (x1 + x2) / 2.0
	mid_y = (y1 + y2) / 2.0
	b = (y2 - y1) / 2.0
	a = (x2 - x1) / 2.0
	for x in range(x1, x2 + 1):
		for y in range(y1, y2 + 1):
			dx = x - mid_x
			dy = y - mid_y
			if (dx * dx) / (a * a) + (dy * dy) / (b * b) <= 1.25 and (dx * dx) / (a * a) + (dy * dy) / (b * b) >= 0.75:
				image[x][y] = blurred_image[x][y]


def blur_orthogonal_border(image, blurred_image, x1, y1, x2, y2, border_size):
	if x1 == x2:
		for x in range(x1 - border_size, x1 + border_size + 1):
			for y in range(y1, y2 + 1):
				image[x][y] = blurred_image[x][y]
	if y1 == y2:
		for y in range(y1 - border_size, y1 + border_size + 1):
			for x in range(x1, x2 + 1):
				image[x][y] = blurred_image[x][y]


def blur_rectangle_border(image, x1, y1, x2, y2, border_size=2):
	blurred_image = cv2.GaussianBlur(image, (5,5), 0)
	blur_orthogonal_border(image, blurred_image, x1, y1, x2, y1, border_size)
	blur_orthogonal_border(image, blurred_image, x1, y2, x2, y2, border_size)
	blur_orthogonal_border(image, blurred_image, x1, y1, x1, y2, border_size)
	blur_orthogonal_border(image, blurred_image, x2, y1, x2, y2, border_size)


def flip_subimage_vertically_with_border_softening(image, x1, y1, x2, y2):
	flip_subimage_vertically(image, x1, y1, x2, y2)
	blur_rectangle_border(image, x1, y1, x2, y2)


def flip_subimage_ellipse_vertically_with_border_softening(image, x1, y1, x2, y2):
	flip_subimage_ellipse_vertically(image, x1, y1, x2, y2)
	blur_ellipse_border(image, x1, y1, x2, y2)


def apply_thatcher_effect_on_image(input_image_path, output_image_path, output_nonflip_image_path, left_eye_rectangle, right_eye_rectangle, mouth_rectangle):
	image = cv2.imread(input_image_path)
	flip_subimage_ellipse_vertically_with_border_softening(image, left_eye_rectangle[0][0] - 5, left_eye_rectangle[0][1] - 6, left_eye_rectangle[1][0] + 7, left_eye_rectangle[1][1] + 3)
	flip_subimage_ellipse_vertically_with_border_softening(image, right_eye_rectangle[0][0] - 5, right_eye_rectangle[0][1] - 3, right_eye_rectangle[1][0] + 7, right_eye_rectangle[1][1] + 6)
	flip_subimage_ellipse_vertically_with_border_softening(image, mouth_rectangle[0][0] - 4, mouth_rectangle[0][1] - 5, mouth_rectangle[1][0] + 3, mouth_rectangle[1][1] + 5)
	cv2.imwrite(output_nonflip_image_path, image)
	image = cv2.flip(image, 0)
	cv2.imwrite(output_image_path, image)


def rotate_image(image, angle):
	height, width = image.shape[:2]
	center = (width / 2.0, height / 2.0)
	matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
	return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR)


def play_button_press_beep():
	try:
		subprocess.Popen(
			["afplay", "-t", "0.08", "/System/Library/Sounds/Basso.aiff"],
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
		)
	except OSError:
		pass


def load_font(size):
	for font_path in FONT_CANDIDATES:
		try:
			return ImageFont.truetype(font_path, size=size)
		except OSError:
			continue
	return ImageFont.load_default()


def draw_rounded_square(frame, x, y, size, color, radius):
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


def draw_centered_text_block(draw, lines, left, top, size, font, color, line_gap):
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


def draw_button(frame, x, y, size, color, lines, font_size=30):
	corner_radius = max(8, size // 10)
	draw_rounded_square(frame, x, y, size, color, corner_radius)
	pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	draw = ImageDraw.Draw(pil_image)
	font = load_font(font_size)
	draw_centered_text_block(
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


def add_processed_preview_buttons(frame, left_lines, right_lines):
	with_buttons = frame.copy()
	h, w, _ = with_buttons.shape
	center_x, center_y = w // 2, h // 2
	head_half_w = int(w * 0.22)
	head_half_h = int(h * 0.38)
	square_size = max(120, min(h, w) // 5)
	top_y = max(10, center_y - head_half_h - square_size - 20)
	left_x = max(10, center_x - head_half_w - square_size - 20)
	right_x = min(w - square_size - 10, center_x + head_half_w + 20)
	with_buttons = draw_button(with_buttons, left_x, top_y, square_size, LEFT_BUTTON_COLOR, left_lines)
	with_buttons = draw_button(with_buttons, right_x, top_y, square_size, RIGHT_BUTTON_COLOR, right_lines)
	return with_buttons


def show_processed_image(image_path):
	image = cv2.imread(image_path)
	if image is None:
		print(f"❌ error: Cannot open processed image: {image_path}")
		return "close"
	cv2.destroyAllWindows()
	cv2.namedWindow(PROCESSED_PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
	print("已顯示處理後照片，按 F9 旋轉 180 度，按 Esc/Enter/Space 關閉視窗。")
	post_f7_mode = False
	while True:
		if post_f7_mode:
			left_lines = ["列印成", "紀念品"]
			right_lines = ["刪除照片", "並重新啟動", "相機"]
		else:
			left_lines = ["結束旋轉"]
			right_lines = ["旋轉照片"]
		cv2.imshow(PROCESSED_PREVIEW_WINDOW_NAME, add_processed_preview_buttons(image, left_lines, right_lines))
		key = cv2.waitKeyEx(30)
		if key in KEY_F9_CODES:
			play_button_press_beep()
			if post_f7_mode:
				cv2.destroyAllWindows()
				return "restart_camera"
			start_image = image.copy()
			for frame_idx in range(1, ROTATION_ANIMATION_FRAMES + 1):
				angle = (180.0 * frame_idx) / ROTATION_ANIMATION_FRAMES
				animated_frame = rotate_image(start_image, angle)
				cv2.imshow(PROCESSED_PREVIEW_WINDOW_NAME, animated_frame)
				cv2.waitKey(ROTATION_ANIMATION_DELAY_MS)
			image = rotate_image(start_image, 180)
			while cv2.waitKeyEx(1) != -1:
				pass
			continue
		if key in KEY_F7_CODES:
			play_button_press_beep()
			post_f7_mode = True
			continue
		if key in (27, 13, 32):
			break
	cv2.destroyAllWindows()
	return "close"


def main():
	while True:
		captured_image_path = capture_photo()
		if not captured_image_path:
			print("Capture cancelled.")
			return

		os.makedirs(OUTPUT_IMAGES_DIRECTORY_PATH, exist_ok=True)
		os.makedirs(OUTPUT_NONFLIP_IMAGE_PATH, exist_ok=True)
		input_file_path = captured_image_path
		if not isfile(input_file_path):
			print(f"❌ error: Not found captured image: {input_file_path}")
			return

		output_file_path = OUTPUT_CAPTURED_IMAGE_PATH
		output_nonflip_image_path = OUTPUT_CAPTURED_NONFLIP_IMAGE_PATH
		if isfile(output_file_path):
			os.remove(output_file_path)
		if isfile(output_nonflip_image_path):
			os.remove(output_nonflip_image_path)

		image_facial_landmarks = get_image_facial_landmarks(input_file_path)
		if not image_facial_landmarks or len(image_facial_landmarks) == 0:
			print(f"❌ error: Not detected face in {CAPTURED_IMAGE_FILENAME}")
			return

		if len(image_facial_landmarks) != 68:
			faces = len(image_facial_landmarks) / 68
			print(f"⚠️ warning: Get {len(image_facial_landmarks)} points in {CAPTURED_IMAGE_FILENAME}. Get {faces} faces).")

		left_eye_rectangle = get_bounding_rectangle(image_facial_landmarks[36:42])
		right_eye_rectangle = get_bounding_rectangle(image_facial_landmarks[42:48])
		mouth_rectangle = get_bounding_rectangle(image_facial_landmarks[48:68])
		apply_thatcher_effect_on_image(input_file_path, output_file_path, output_nonflip_image_path, left_eye_rectangle, right_eye_rectangle, mouth_rectangle)
		if not isfile(output_file_path) or not isfile(output_nonflip_image_path):
			print(f"❌ error: Processed image not generated: {output_file_path} / {output_nonflip_image_path}")
			return
		print(f"Done: {output_file_path}")
		print(f"Done: {output_nonflip_image_path}")
		next_action = show_processed_image(output_file_path)
		if next_action != "restart_camera":
			return


if __name__ == "__main__":
	main()
