import cv2
import numpy as np
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model_path = 'spoof_detection_model.h5'  
model = load_model(model_path)

# Parameters
img_height = 224
img_width = 224

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (img_width, img_height))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def detect_spoof(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_image(frame)

    # Predict using the model
    prediction = model.predict(preprocessed_frame)
    predicted_class = int(np.round(prediction[0][0]))
    label = 'Genuine' if predicted_class == 0 else 'Spoof'

    return label

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label = detect_spoof(frame)

        # Display the result on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if label == 'Genuine' else (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
