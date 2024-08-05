import cv2
import mediapipe as mp

mp_face_mesh =  mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Configuración de MediaPipe Hands
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_mesh.FaceMesh(
    static_image_mode = False,
    max_num_faces=2,
    min_detection_confidence=0.5
) as face_mesh, mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    
    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # Procesa el frame para detección de manos
        results_hands = hands.process(frame_rgb)

        if results_hands.multi_hand_landmarks is not None:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Dibuja los puntos de referencia de la mano
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                )

            """  # Lista de índices de puntos para los dedos
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Determinar qué dedo está levantado comparando con el punto base
            def is_finger_raised(tip_point, base_point):
                return tip_point.y < base_point.y

            # Definir los puntos base para comparación
            thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            ring_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
            pinky_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

            # Verificar qué dedo está levantado
            raised_fingers = []
            if is_finger_raised(thumb_tip, thumb_base):
                raised_fingers.append("Thumb")
            if is_finger_raised(index_tip, index_base):
                raised_fingers.append("Index")
            if is_finger_raised(middle_tip, middle_base):
                raised_fingers.append("Middle")
            if is_finger_raised(ring_tip, ring_base):
                raised_fingers.append("Ring")
            if is_finger_raised(pinky_tip, pinky_base):
                raised_fingers.append("Pinky")

            # Mostrar en la pantalla qué dedos están levantados
            if raised_fingers:
                cv2.putText(frame, f"Raised Fingers: {', '.join(raised_fingers)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) """

        if results.multi_face_landmarks is not None:
            """ for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                frame, 
                face_landmarks, 
                mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1)
            ) """
                
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    face_landmarks, 
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1)
                )
                
            # Itera sobre todos los puntos de referencia
            """ for landmark in face_landmarks.landmark:
                # Obtén el punto de referencia específico
                landmark = face_landmarks.landmark[POINT_INDEX]

                # Coordenadas normalizadas (0-1)
                x = landmark.x
                y = landmark.y
                z = landmark.z

                # Convierte las coordenadas normalizadas a coordenadas de imagen
                ih, iw, _ = frame.shape
                x_px = int(x * iw)
                y_px = int(y * ih)

                # Dibuja un círculo en el punto de referencia
                cv2.circle(frame, (x_px, y_px), 2, (0, 255, 0), -1)

                # Imprime las coordenadas para depuración
                print(f"Point {POINT_INDEX} position: ({x_px}, {y_px})") """

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()