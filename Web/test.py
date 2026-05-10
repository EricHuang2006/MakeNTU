from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def upload_to_drive(service, image_path):
    file_metadata = {
        "name": "capture.jpg",
        "parents": ["16tXR0g4jbL5se5FfhPufIubPPNevpqY6"]
        # "parents": ["你的資料夾ID"]  # 可選，指定上傳到某個資料夾
    }

    media = MediaFileUpload(image_path, mimetype="image/jpeg")

    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()

    print("Uploaded file ID:", file.get("id"))

service = build('drive', 'v3', credentials=creds)
upload_to_drive("", "./capture.jpg")

