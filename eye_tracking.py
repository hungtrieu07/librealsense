import cv2
import numpy as np
import dlib
import pyrealsense2 as rs

face_detector = dlib.get_frontal_face_detector()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == "L500":
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    align_to = rs.stream.color
    align = rs.align(align_to)
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        faces = face_detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            roi_left_eye = gray[y + int(0.1 * h):y + int(0.4 * h), x + int(0.1 * w):x + int(0.5 * w)]
            roi_right_eye = gray[y + int(0.1 * h):y + int(0.4 * h), x + int(0.5 * w): x + int(0.9 * w)]

            _, thresh_left_eye = cv2.threshold(roi_left_eye, 30, 255, cv2.THRESH_BINARY)
            _, thresh_right_eye = cv2.threshold(roi_right_eye, 30, 255, cv2.THRESH_BINARY)
            
            contours_left, _ = cv2.findContours(thresh_left_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_right, _ = cv2.findContours(thresh_right_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            for contour in contours_left:
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                center_x = x + int(0.1 * w) + x_c + w_c // 2 
                center_y = y + int(0.1 * h) + y_c + h_c // 2 
                radius = max(w_c, h_c) // 3 
                cv2.circle(color_image, (center_x, center_y), radius, (0, 255, 0), 2)            

            for contour in contours_right:
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                center_x = x + int(0.5 * w) + x_c + w_c // 2 
                center_y = y + int(0.1 * h) + y_c + h_c // 2 
                radius = max(w_c, h_c) // 3
                cv2.circle(color_image, (center_x, center_y), radius, (0, 255, 0), 2)

        cv2.imshow('Eye Tracking', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

finally:
    # Stop streaming
    pipeline.stop()