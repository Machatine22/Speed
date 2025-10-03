from flask import Flask, request, jsonify
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load your custom license plate model
plate_model = YOLO("license-plate-detection.pt")

# OCR engine
reader = easyocr.Reader(['en'])

@app.route("/create_ticket", methods=["POST"])
def create_ticket():
    # Get uploaded image
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Run detection
    results = plate_model(image)

    plates = []
    for r in results:
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            crop = image[int(y1):int(y2), int(x1):int(x2)]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # OCR
            text = reader.readtext(gray, detail=0)
            plates.append({"bbox": [x1, y1, x2, y2], "text": text})

    return jsonify({"plates": plates})


@app.route("/list_tickets", methods=["GET"])
def list_tickets():
    # (For now just dummy response)
    return jsonify({"tickets": ["Demo ticket 1", "Demo ticket 2"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
