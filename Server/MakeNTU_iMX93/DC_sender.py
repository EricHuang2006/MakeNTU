import requests

def send_image_to_discord(img, webhook_url):
    # return
    files = {
        "file": ("capture.jpg", img)
    }
    response = requests.post(webhook_url, files=files)
    
    if response.status_code in [200, 204]:
        print("Discord 傳送成功！")
    else:
        print(f"傳送失敗，錯誤碼：{response.status_code}")

    time.sleep(3)

import cv2
import requests


def send_frame_to_discord(frame, webhook_url):
    return
    if not webhook_url:
        print("DISCORD_WEBHOOK_URL not set. Skipping Discord upload.")
        return False

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        print("Failed to encode frame for Discord.")
        return False

    try:
        response = requests.post(
            webhook_url,
            data={"content": "📷 系統偵測到舉手，自動拍照！"},
            files={"file": ("capture.jpg", buffer.tobytes())},
            timeout=5,
        )

        if response.status_code in [200, 204]:
            print("照片傳送成功！")
            return True

        print(f"Discord 傳送失敗，狀態碼: {response.status_code}")
        return False

    except Exception as e:
        print(f"Discord 傳送失敗: {e}")
        return False