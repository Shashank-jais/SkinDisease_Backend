from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from PIL import Image, ImageDraw, ImageFont
import base64
from ultralytics import YOLO
import shutil
import io

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://skin-disease-model.vercel.app"}})


UPLOAD_FOLDER = "./uploads"
RESULT_FOLDER = "./result"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
segmentation_model = YOLO("model_weight/segment/best.pt")
detection_model = YOLO("model_weight/detect/best.pt")
classify_model = YOLO("model_weight/classify/best.pt")


# Function to encode image to base64
def encode_image_from_pil(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# Function to add text to an image
def classify_and_add_text(img, class_name, probability):
    font = ImageFont.truetype("arial.ttf", size=20)  # Use custom font
    text = f"Class: {class_name} | Confidence: {probability * 100:.2f}%"

    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    position = ((img_width - text_width) // 2, img_height - text_height - 10)

    draw.rectangle(
        [0, img_height - text_height - 20, img_width, img_height],
        fill=(0, 0, 0, 128)
    )
    draw.text(position, text, font=font, fill=(255, 255, 255))
    return img


@app.route("/api/")
def home():
    return "Server is running superb!"


@app.route("/api/upload", methods=['POST'])
def upload():
    if "image" not in request.files:
        return jsonify({'error': "No such file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Open the image
    img = Image.open(filepath)
    images_base64 = []

    try:
        # Segmentation
        result = segmentation_model.predict(img, save=True)[0]
        segmented_image = Image.open(os.path.join(result.save_dir, file.filename))
        images_base64.append(encode_image_from_pil(segmented_image))

        # Detection
        result = detection_model.predict(img, save=True)[0]
        detected_image = Image.open(os.path.join(result.save_dir, file.filename))
        images_base64.append(encode_image_from_pil(detected_image))

        # Classification
        results = classify_model(img)
        class_name = results[0].names[results[0].probs.top1]
        probability = results[0].probs.top1conf.item()
        classified_img = classify_and_add_text(img, class_name, probability)
        images_base64.append(encode_image_from_pil(classified_img))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Cleanup
    shutil.rmtree('runs', ignore_errors=True)
    os.remove(filepath)

    return jsonify({"processedResults": images_base64})


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)
