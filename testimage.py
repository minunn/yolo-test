import json
import requests
import cv2
import matplotlib.pyplot as plt

url = "https://api.ultralytics.com/v1/predict/coZThRvFC3t2brXOu2ES"
headers = {"x-api-key": "34c4ac86a3e4756731a2a6c473798a595bd1fb795f"} # you can use my api key i don't need it anymore since it was just for testing have fun
data = {"size": 640, "confidence": 0.25, "iou": 0.45}
with open("test.jpg", "rb") as f:
    response = requests.post(url, headers=headers, data=data, files={"image": f})

response.raise_for_status()

inference_results = response.json()
print(json.dumps(inference_results, indent=2))

image = cv2.imread("test.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictions = inference_results.get('data', [])
if not predictions:
    print("Pas de prédiction trouvé pour cette réponse.")
else:
    for prediction in predictions:
        xcenter = prediction.get('xcenter')
        ycenter = prediction.get('ycenter')
        width = prediction.get('width')
        height = prediction.get('height')
        label = prediction.get('name')
        confidence = prediction.get('confidence')

        if xcenter is not None and ycenter is not None and width is not None and height is not None and label and confidence:
            x1 = int((xcenter - width / 2) * image.shape[1])
            y1 = int((ycenter - height / 2) * image.shape[0])
            x2 = int((xcenter + width / 2) * image.shape[1])
            y2 = int((ycenter + height / 2) * image.shape[0])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(image, f"{label}: {confidence}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        else:
            print("Prédiction invalide")

plt.imshow(image)
plt.show()
