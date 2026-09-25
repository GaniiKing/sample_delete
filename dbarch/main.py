import cv2
import ollama

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Press SPACE to capture", frame)

    key = cv2.waitKey(1)

    if key == 32:  # SPACE key
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        break

cap.release()
cv2.destroyAllWindows()

response = ollama.chat(
    model='llava-llama3:8b',
    messages=[
        {
            'role': 'user',
            'content': 'Describe this scene briefly',
            'images': [image_bytes]
        }
    ]
)

print(response['message']['content'])