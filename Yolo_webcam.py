from ultralytics import YOLO
import cv2
import cvzone
import math

# Function to calculate distance
def calculate_distance(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width

# Function to get focal length based on object width and known distance
def get_focal_length(known_width, known_distance, per_width):
    return (per_width * known_distance) / known_width

cap = cv2.VideoCapture(0)
cap.set(3, 2500)
cap.set(4, 1200)

model = YOLO("../Yolo_Weights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Set known width and known distance (calibrate based on your camera and setup)
known_width = 20.0  # Example: width of a standard object in inches
known_distance = 24.0  # Example: distance from the camera to the object in inches
focal_length = -1  # Initialize focal length, to be calculated during the first iteration

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

            # Calculate distance based on known width and focal length
            if focal_length == -1:
                # Calculate focal length during the first iteration
                focal_length = get_focal_length(known_width, known_distance, w)

            distance = calculate_distance(known_width, focal_length, w)
            cv2.putText(img, f'Distance: {distance:.2f} inches', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
