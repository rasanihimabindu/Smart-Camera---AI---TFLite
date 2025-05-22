import cv2
import numpy as np
import tensorflow as tf

# Load labels
def load_labels(path='labelmap.txt'):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='detect.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

labels = load_labels()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare input image
    input_image = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(input_image, axis=0)
    input_data = np.uint8(input_data)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract output
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    height, width, _ = frame.shape

    # Draw results
    for i in range(len(scores)):
        if scores[i] > 0.5:
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1 = int(xmin * width), int(ymin * height)
            x2, y2 = int(xmax * width), int(ymax * height)

            class_id = int(class_ids[i])
            label = labels[class_id] if class_id < len(labels) else 'Unknown'
            score_text = f'{label}: {int(scores[i] * 100)}%'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, score_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Smart Camera AI', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
