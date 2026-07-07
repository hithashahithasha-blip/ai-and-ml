from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Read the image
img = cv2.imread("kitchen.jpg")

# Detect ONLY bottles
results = model(img, classes=[39])

# Draw boxes around detected bottles
annotated_img = results[0].plot()

# Resize image for display
annotated_img = cv2.resize(annotated_img, (800, 600))

# Show image

cv2.imshow("Bottle Detection", annotated_img)

# Wait until a key is pressed
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()