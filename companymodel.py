import cv2
from yolov5 import YOLOv5
import pandas

model_path = 'companymodel.pt'

model = YOLOv5(model_path)
# model.model.eval()  # Set the model to evaluation mode
count = 0

rtsp_url = 'rtsp://admin:admin_123@122.180.245.190:1024/unicast/c3/s0/live'
cap = cv2.VideoCapture(rtsp_url)

output_path = "output/video/output.mp4"
output_writer = None

# Set the desired width and height for resizing
resized_width = 1080
resized_height = 720

# Set the confidence threshold
confidence_threshold = 0.5

while cap.isOpened():
    ret, frame = cap.read()

    count += 1
    print(count)

    if not ret:
        break

    frame = cv2.resize(frame, (resized_width, resized_height))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.model(frame_rgb)
    df = results.pandas().xyxy[0]

    boxes = results.pred[0][:, :4]
    labels = results.pred[0][:, -1]
    scores = results.pred[0][:, 4]

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        class_id = int(label.item())
        confidence = score.item()

        if confidence > confidence_threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'Class: {class_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

    # Set the frame rate to 60
    frame_rate = 60
    cap.set(cv2.CAP_PROP_FPS, frame_rate)

    cv2.imshow('Detection-YOLOv5', frame)

    if output_writer is None:
        output_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 60, (resized_width, resized_height))
    output_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
