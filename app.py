import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# 1. Load model YOLOv8 bawaan (COCO)
model = YOLO('yolov8n.pt')
model.conf = 0.5  # Confidence threshold

# 2. Dictionary sederhana untuk memetakan nama kelas -> kategori (sampah)
trash_category = {
    # Organik
    "banana": "Organic",
    "apple": "Organic",
    "orange": "Organic",
    "broccoli": "Organic",
    "carrot": "Organic",
    "sandwich": "Organic",
    "pizza": "Organic",
    "hot dog": "Organic",
    "donut": "Organic",
    "cake": "Organic",

    # Anorganik
    "bottle": "Unorganic",
    "cup": "Unorganic",
    "fork": "Unorganic",
    "knife": "Unorganic",
    "spoon": "Unorganic",
    "bowl": "Unorganic",
    "wine glass": "Unorganic",
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "Harap sertakan key 'image' berisi Base64 string"}), 400

    try:
        base64_str = data['image']
        # Kalau masih ada prefix "data:image/png;base64," → buang dulu
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",")[1]
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Gagal mendekode gambar: {str(e)}"}), 400

    if frame is None:
        return jsonify({"error": "Gagal memproses gambar, kemungkinan format Base64 tidak valid."}), 400

    # Jalankan prediksi YOLO
    results = model.predict(frame, imgsz=640)

    # Hanya akan mengirimkan satu prediksi dengan confidence tertinggi
    best_detection = None
    best_conf = -1.0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls_index = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_index]

            # Hanya cek jika class_name ada di dictionary
            if class_name in trash_category:
                if conf > best_conf:
                    best_conf = conf
                    best_detection = {
                        "class_name": class_name,
                        "category": trash_category[class_name],
                        "confidence": round(conf, 3)
                    }

    # Siapkan struktur JSON final
    predictions = []

    # Jika tidak ada deteksi yang relevan, kembalikan "unknown"
    if best_detection:
        predictions.append(best_detection)
    else:
        predictions.append({
            "class_name": "Unknown",
            "category": "Unknown",
            "confidence": 0
        })

    return jsonify({"predictions": predictions}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
