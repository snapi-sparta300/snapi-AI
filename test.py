from ultralytics import YOLO
import cv2

# 1. 모델 로드
model = YOLO("my_best.pt")  # pt 파일 경로

# 2. 이미지 불러오기
image_path = "handimage.jpg"  # 인식할 이미지 경로
img = cv2.imread(image_path)

# 3. 객체 인식 수행
results = model(img)

# 4. 결과 출력
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        print(f"{label} ({conf:.2f})")

# 5. 결과 이미지 저장
results[0].save(filename="/home/parkjinseo/Snapi/result2.jpg")
print("결과 이미지 저장됨: Snapi/result.jpg")

# 6. 결과 이미지 보기
result_img = cv2.imread("/home/parkjinseo/Snapi/result2.jpg")
cv2.imshow("Detection Result", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
