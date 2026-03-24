import cv2
import numpy as np

def capture_photo():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("error: Cannot activate camera")
        return
    print("--- Camera is ready ---")
    print("- 按下 [空白鍵] (Space) 拍照並存檔")
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

        # 為了不破壞原始影像，複製一份來製作預覽畫面
        display_frame = frame.copy()
        h, w, _ = display_frame.shape
        
        # --- 定義橢圓參數 ---
        center = (w // 2, h // 2)
        axes = (int(w * 0.18), int(h * 0.38))
        
        # ==========================================
        # 製作「空心洞」半透明遮罩
        # ==========================================
        # 1. 建立一個與畫面大小相同的黑色遮罩 (全 0)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 2. 在遮罩上畫一個「實心白色」橢圓 (代表我們要留下的空心區域)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # 3. 建立一個半透明的灰色層 (數值越小越暗)
        overlay = display_frame.copy()
        overlay[:] = (40, 40, 40) # 深灰色 BGR
        
        # 4. 根據遮罩進行合成：
        #    - 遮罩為 0 (黑色) 的地方，顯示變暗的 overlay
        #    - 遮罩為 255 (白色) 的地方，顯示原始的 display_frame
        # 使用 np.where 進行快速像素替換
        # 將 mask 擴展到 3 通道以便運算
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # 合成：如果 mask 是白色，用原始圖；否則用灰色層，並設定透明度
        alpha = 0.7 # 遮罩區的透明度
        background_part = cv2.addWeighted(display_frame, 1 - alpha, overlay, alpha, 0)
        
        # 最終顯示畫面：空心部分用原始影格，其餘部分用合成後的背景
        display_frame = np.where(mask_3ch == 255, display_frame, background_part)

        # ==========================================
        # 繪製對齊虛線 (讓空心邊緣更明顯)
        # ==========================================
        for i in range(0, 360, 10):
            cv2.ellipse(display_frame, center, axes, 0, i, i+5, (0, 255, 0), 2)

        # 顯示預覽
        cv2.imshow('Face Alignment', display_frame)
        
        # cv2.imshow('Camera preview', frame)
        
        # Listening for keyin
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            file_path = f"input_images/original_face.jpeg"
            cv2.imwrite(file_path, frame)
            print(f"{file_path} is already saved.")
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_photo()