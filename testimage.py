import json
import requests
import cv2
import matplotlib.pyplot as plt

# Run inference on an image
url = "https://api.ultralytics.com/v1/predict/coZThRvFC3t2brXOu2ES"
headers = {"x-api-key": "34c4ac86a3e4756731a2a6c473798a595bd1fb795f"}
data = {"size": 640, "confidence": 0.25, "iou": 0.45}
with open("test.jpg", "rb") as f:
    response = requests.post(url, headers=headers, data=data, files={"image": f})

# Check for successful response
response.raise_for_status()

# Print inference results
inference_results = response.json()
print(json.dumps(inference_results, indent=2))

# Load the original image
image = cv2.imread("test.jpg")

# Convert color style from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the predictions from the response
predictions = inference_results.get('data', [])

# Check if predictions exist
if not predictions:
    print("No predictions found in the response.")
else:
    # Draw the bounding boxes on the image
    for prediction in predictions:
        xcenter = prediction.get('xcenter')
        ycenter = prediction.get('ycenter')
        width = prediction.get('width')
        height = prediction.get('height')
        label = prediction.get('name')
        confidence = prediction.get('confidence')

        # Check if bbox, label, and confidence exist
        if xcenter is not None and ycenter is not None and width is not None and height is not None and label and confidence:
            # Convert center coordinates, width and height to top-left and bottom-right coordinates
            x1 = int((xcenter - width / 2) * image.shape[1])
            y1 = int((ycenter - height / 2) * image.shape[0])
            x2 = int((xcenter + width / 2) * image.shape[1])
            y2 = int((ycenter + height / 2) * image.shape[0])

            # Draw rectangle (bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display label and confidence
            cv2.putText(image, f"{label}: {confidence}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        else:
            print("bbox, label, or confidence not found in prediction.")

# Display the image
plt.imshow(image)
plt.show()