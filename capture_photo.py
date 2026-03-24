import cv2

def capture_photo():
    camera = cv2.VideoCapture()
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
        
        cv2.imshow('Camera preview', frame)
        
        # Listening for keyin
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            file_path = f"input_images/original_face.jpeg"
            cv2.imwrite(file_path, frame)
            print(f"{file_name} is already saved.")
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_photo()