import cv2
from inference_sdk import InferenceHTTPClient

# Initialisation du client Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="lPkTaCAqUye82G05U53F"
)

# Chemin de la vidéo (remplace par le chemin de ta vidéo)
video_path = "test.mp4"

# Ouvrir la vidéo avec OpenCV
cap = cv2.VideoCapture(video_path)

# Vérifier si la vidéo s'ouvre correctement
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo")
    exit()

while cap.isOpened():
    ret, frame = cap.read()  # Lire une image de la vidéo
    if not ret:
        break  # Fin de la vidéo

    # Sauvegarder temporairement l'image pour l'envoyer à l'API
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, frame)

    # Effectuer l'inférence avec Roboflow
    result = CLIENT.infer(temp_image_path, model_id="new-dataset-xjgtk/5")

    # Parcourir les résultats et dessiner les boîtes de détection
    for item in result['predictions']:
        x, y, width, height = item['x'], item['y'], item['width'], item['height']
        x1, y1 = int(x - width / 2), int(y - height / 2)
        x2, y2 = int(x + width / 2), int(y + height / 2)

        label = item['class']
        confidence = item['confidence']

        # Dessiner la boîte englobante
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence*100:.2f}%", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2)

    # Afficher la vidéo avec les détections
    cv2.imshow("Video with Detections", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
