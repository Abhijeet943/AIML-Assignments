import cv2
import os
from ultralytics import YOLO


# ---------------------------------------------------------
# 1. Define Paths
# ---------------------------------------------------------

# Current assignment folder
assignment_folder = os.path.dirname(os.path.abspath(__file__))

# Main AIML-Assignments folder
main_folder = os.path.dirname(assignment_folder)


# ---------------------------------------------------------
# 2. Load YOLO Model
# ---------------------------------------------------------

model_path = os.path.join(
    main_folder,
    "yolo11n.pt"
)

print("\nLoading YOLO model...")

model = YOLO(model_path)

print("YOLO model loaded successfully.")


# ---------------------------------------------------------
# 3. Load CCTV Video
# ---------------------------------------------------------

video_path = os.path.join(
    assignment_folder,
    "opencv.mp4"
)

print("\nVideo Path:")
print(video_path)


cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("\nError: Could not open CCTV video.")
    print("Please check that opencv.mp4 exists in the assignment folder.")
    exit()


# ---------------------------------------------------------
# 4. Get Video Information
# ---------------------------------------------------------

fps = int(cap.get(cv2.CAP_PROP_FPS))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print("\nCCTV Video Information")
print("----------------------")

print("Width:", width)
print("Height:", height)
print("FPS:", fps)
print("Total Frames:", total_frames)


# ---------------------------------------------------------
# 5. Create Output Video
# ---------------------------------------------------------

output_path = os.path.join(
    assignment_folder,
    "detected_cctv.mp4"
)


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)


# ---------------------------------------------------------
# 6. Process CCTV Frames
# ---------------------------------------------------------

frame_count = 0

print("\nStarting object detection...")
print("Press Q to stop the video.\n")


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # -----------------------------------------------------
    # YOLO Object Detection
    # -----------------------------------------------------

    results = model(
        frame,
        verbose=False
    )


    # -----------------------------------------------------
    # Draw Bounding Boxes and Labels
    # -----------------------------------------------------

    annotated_frame = results[0].plot()


    # -----------------------------------------------------
    # Save Processed Frame
    # -----------------------------------------------------

    out.write(annotated_frame)


    # -----------------------------------------------------
    # Display Frame
    # -----------------------------------------------------

    cv2.imshow(
        "CCTV Object Detection - YOLO",
        annotated_frame
    )


    # -----------------------------------------------------
    # Press Q to Stop
    # -----------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------------------------------------
# 7. Release Resources
# ---------------------------------------------------------

cap.release()

out.release()

cv2.destroyAllWindows()


# ---------------------------------------------------------
# 8. Completion Message
# ---------------------------------------------------------

print("\n===================================")
print("OBJECT DETECTION COMPLETED")
print("===================================")

print("Frames processed:", frame_count)

print("\nOutput video saved at:")

print(output_path)