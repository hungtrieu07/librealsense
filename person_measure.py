## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import math
import pyrealsense2 as rs
import numpy as np
import cv2

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


def calculate_distance(color_intrin, depth_frame, ix, iy, x, y):
    udist = depth_frame.get_distance(ix, iy)
    vdist = depth_frame.get_distance(x, y)

    point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [ix, iy], udist)
    point2 = rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], vdist)

    dist = math.sqrt(
        math.pow(point1[0] - point2[0], 2)
        + math.pow(point1[1] - point2[1], 2)
        + math.pow(point1[2] - point2[2], 2)
    )
    # print 'distance: '+ str(dist)
    return dist


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
        
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        # print(color_intrin)

        result = model.predict(color_image)
        list_class = result[0].boxes.cls
        names = model.names
        bboxes = result[0].boxes.xyxy.numpy()

        for box, name in zip(bboxes, list_class):
            name = names[int(name)]
            # print(name)

            if name == "mouse":
                box = list(map(int, box))
                cv2.rectangle(
                    color_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2
                )

                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                x_a = int(box[0])
                y_a = center_y

                dist = depth_frame.get_distance(center_x, center_y)
                
                cv2.putText(color_image, str(f'{dist:.2f}'), (box[0], box[1] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)

                distance_2_point = calculate_distance(color_intrin, depth_frame, x_a, y_a, center_x, center_y)
                print(distance_2_point)
                
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # depth_colormap_dim = depth_colormap.shape
        # color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        # if depth_colormap_dim != color_colormap_dim:
        #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        #     images = np.hstack((resized_color_image, depth_colormap))
        # else:
        #     images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RealSense", color_image)
        
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline.stop()

