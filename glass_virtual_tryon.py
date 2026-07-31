import cv2
import numpy as np
import sys
import os
from datetime import datetime
import time


print("=" * 50)
print("Running file:", os.path.abspath(__file__))
print("=" * 50)

# ======================= CONSTANTS ======================= #

FACE_SCALE = 1.3
FACE_NEIGHBORS = 5
FACE_MIN_SIZE = (200, 200)

EYE_SCALE = 1.1
EYE_NEIGHBORS = 5
EYE_MIN_SIZE = (100, 100)

GLASS_SCALE = 2.5
GLASS_X_OFFSET = 0.28
GLASS_Y_OFFSET = 0.8


WINDOW_NAME = "AI Virtual Eyewear Studio"

# =================== LOAD HAAR CASCADES =================== #

face_detection_model = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_alt.xml"
)

eye_detection_model = cv2.CascadeClassifier(
    "models/haarcascade_eye.xml"
)

if face_detection_model.empty():
    print("Error: Face cascade not found!")
    sys.exit()

if eye_detection_model.empty():
    print("Error: Eye cascade not found!")
    sys.exit()

# =================== LOAD GLASSES =================== #
glass_images = [
    cv2.imread("assets/glasses/aviator.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("assets/glasses/round.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("assets/glasses/rectangle.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("assets/glasses/cateye.png", cv2.IMREAD_UNCHANGED)
]

glass_names = [
    "Aviator",
    "Round",
    "Rectangle",
    "CatEye"
]

for img in glass_images:
    if img is None:
        print("Error: Could not load one or more glasses images.")
        sys.exit()

current_glass = 0
# =================== POSITION SMOOTHING =================== #

prev_x = None
prev_y = None



SMOOTHING = 0.7
prev_angle = 0
ANGLE_SMOOTH = 0.8
# Screenshot notification
show_saved = False
saved_time = 0

# =================== OPEN CAMERA =================== #

vid = cv2.VideoCapture(0)

if not vid.isOpened():
    print("Error: Cannot access webcam!")
    sys.exit()

cv2.namedWindow(WINDOW_NAME)

# =================== MAIN LOOP =================== #

while True:

    ret, image = vid.read()

    if not ret:
        print("Failed to capture frame.")
        break

    final_image = image.copy()

    # Current glasses
    glass_image = glass_images[current_glass]

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = face_detection_model.detectMultiScale(
        gray_image,
        scaleFactor=FACE_SCALE,
        minNeighbors=FACE_NEIGHBORS,
        minSize=FACE_MIN_SIZE
    )

    for (face_x, face_y, face_w, face_h) in faces:

        eye_centers = []

        face_roi = gray_image[
            face_y:face_y + face_h,
            face_x:face_x + face_w
        ]

        eyes = eye_detection_model.detectMultiScale(
            face_roi,
            scaleFactor=EYE_SCALE,
            minNeighbors=EYE_NEIGHBORS,
            minSize=EYE_MIN_SIZE
        )

        for (eye_x, eye_y, eye_w, eye_h) in eyes:

            eye_centers.append((
                face_x + eye_x + eye_w // 2,
                face_y + eye_y + eye_h // 2
            ))

        if len(eye_centers) >= 2:

            # ---------------- Eye Based Position ---------------- #

            eye_centers = sorted(eye_centers)

            left_eye = eye_centers[0]
            right_eye = eye_centers[1]

            eye_distance = right_eye[0] - left_eye[0]

            # Automatic scaling using both eyes and face width
            #---------------- Resize Glasses ---------------- #

            glass_width = max(
                int(eye_distance * 2.8),
                int(face_w * 0.95)
            )

            scale = glass_width / glass_image.shape[1]

            resized_glasses = cv2.resize(
                glass_image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR
            )

            # ---------------- Calculate Head Tilt ---------------- #

            angle = np.degrees(
                np.arctan2(
                    right_eye[1] - left_eye[1],
                    right_eye[0] - left_eye[0]
                )
            )

            angle = prev_angle * ANGLE_SMOOTH + angle * (1 - ANGLE_SMOOTH)
            prev_angle = angle

            glass_h, glass_w = resized_glasses.shape[:2]

            center = (glass_w // 2, glass_h // 2)

            rotation_matrix = cv2.getRotationMatrix2D(
                center,
                angle,
                1.0
            )

            # Compute new image size after rotation
            cos = abs(rotation_matrix[0, 0])
            sin = abs(rotation_matrix[0, 1])

            new_w = int((glass_h * sin) + (glass_w * cos))
            new_h = int((glass_h * cos) + (glass_w * sin))

            # Shift image to keep it centered
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]

            # Rotate without cropping
            resized_glasses = cv2.warpAffine(
                resized_glasses,
                rotation_matrix,
                (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )

            glass_h, glass_w = resized_glasses.shape[:2]
            # ---------------- Midpoint Between Eyes ---------------- #

            center_x = (left_eye[0] + right_eye[0]) // 2
            center_y = (left_eye[1] + right_eye[1]) // 2

            # Position glasses
            glass_x = int(center_x - glass_w / 2)
            glass_y = int(center_y - glass_h * 0.40)

            # # -------- Smooth Position -------- #

            if prev_x is None:
                prev_x = glass_x
                prev_y = glass_y

            glass_x = int(prev_x * SMOOTHING + glass_x * (1 - SMOOTHING))
            glass_y = int(prev_y * SMOOTHING + glass_y * (1 - SMOOTHING))

            prev_x = glass_x
            prev_y = glass_y
            # Keep inside image
            if (
                glass_x >= 0 and
                glass_y >= 0 and
                glass_x + glass_w <= final_image.shape[1] and
                glass_y + glass_h <= final_image.shape[0]
            ):

                if resized_glasses.shape[2] == 4:

                    overlay = resized_glasses[:, :, :3]
                    alpha = resized_glasses[:, :, 3] / 255.0

                else:

                    overlay = resized_glasses
                    alpha = np.ones((glass_h, glass_w), dtype=np.float32)

                roi = final_image[
                    glass_y:glass_y+glass_h,
                    glass_x:glass_x+glass_w
                ]

                for c in range(3):

                    roi[:, :, c] = (
                        alpha * overlay[:, :, c] +
                        (1 - alpha) * roi[:, :, c]
                    )

                final_image[
                    glass_y:glass_y+glass_h,
                    glass_x:glass_x+glass_w
                ] = roi

                
    # =================== UI =================== #

    # Transparent background panel
    overlay = final_image.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (540, 175),
        (35, 35, 35),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.55,
        final_image,
        0.45,
        0,
        final_image
    )

    # Title
    cv2.putText(
        final_image,
        "AI VIRTUAL EYEWEAR STUDIO",
        (25, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (0,255,255),
        2
    )

    # Current time
    current_time = datetime.now().strftime("%H:%M:%S")

    cv2.putText(
        final_image,
        current_time,
        (430,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    # Selected glasses
    cv2.putText(
        final_image,
        f"Selected : {glass_names[current_glass]}",
        (25,75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255,255,255),
        2
    )

    # Controls
    cv2.putText(
        final_image,
        "[1] Aviator",
        (25,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )

    cv2.putText(
        final_image,
        "[2] Round",
        (170,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )

    cv2.putText(
        final_image,
        "[3] Rectangle",
        (305,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )

    cv2.putText(
        final_image,
        "[4] CatEye",
        (25,145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )

    cv2.putText(
        final_image,
        "[S] Save",
        (170,145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,0),
        2
    )

    cv2.putText(
        final_image,
        "[Q] Quit",
        (305,145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,165,255),
        2
    )

    # Screenshot notification
    if show_saved:

        if time.time() - saved_time < 1.5:

            cv2.putText(
                final_image,
                "Screenshot Saved!",
                (170,210),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0,255,0),
                2
            )

        else:

            show_saved = False

    cv2.imshow(WINDOW_NAME, final_image)

    # =================== KEYBOARD =================== #

    key = cv2.waitKey(10) & 0xFF

    if key == ord('1'):
        current_glass = 0

    elif key == ord('2'):
        current_glass = 1

    elif key == ord('3'):
        current_glass = 2
    elif key == ord('4'):
        current_glass = 3
    elif key == ord('s'):

        os.makedirs("screenshots", exist_ok=True)

        filename = os.path.join(
            "screenshots",
            datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
        )

        cv2.imwrite(filename, final_image)

        print("Saved:", os.path.abspath(filename))

        show_saved = True
        saved_time = time.time()

    elif key == ord('q') or key == 27:
        break

    # Window closed
    try:
        if cv2.getWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_VISIBLE
        ) < 1:
            break
    except cv2.error:
        break

# =================== CLEANUP =================== #

vid.release()
cv2.destroyAllWindows()