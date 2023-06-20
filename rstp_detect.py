import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

confidence_threshold = 0.75

# RTSP stream URL
rtsp_url = 'rtsp://ishujainbrt@gmail.com:Ishu@1234@192.168.1.41:1935'


cap = cv2.VideoCapture(rtsp_url)

# Get original frame width, height, and frame rate
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

resized_width = 640
resized_height = 480

output_path = "output/video/output.mp4"
output_writer = None


if not cap.isOpened():
    print("Failed to open the RTSP stream")
    exit()

# Read and display frames from the stream
while True:
    ret, frame = cap.read()

    # Check if a frame is successfully read
    if not ret:
        print("Failed to read the frame from the stream")
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    resized_frame = cv2.resize(frame, (resized_width, resized_height))

    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    results = model(frame_rgb)

    # Parse detection results
    detections = results.pandas().xyxy[0]

    person_detections = detections[(detections["name"] == "person") & (detections["confidence"] > confidence_threshold)]

    for _, detection in person_detections.iterrows():
        x, y, w, h = int(detection["xmin"]), int(detection["ymin"]), int(detection["xmax"] - detection["xmin"]), int(
            detection["ymax"] - detection["ymin"])
        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(resized_frame, "person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('RTSP Stream', resized_frame)

    # Write frame to the output video
    if output_writer is None:
        output_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                        (resized_width, resized_height))
    output_writer.write(resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
if output_writer is not None:
    output_writer.release()
cv2.destroyAllWindows()
