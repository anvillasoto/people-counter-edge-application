import os
import tempfile
import sys
import numpy as np
import tensorflow as tf
from six.moves.urllib.request import urlopen
from six import BytesIO
import cv2
import time
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps


os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#path to the saved tesnorflow model
model_path = "models/faster_rcnn_inception_v2_coco_2018_01_28/saved_model"


# labels mapping for the ssd mobilenet v2 coco 
labels = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

# video frames intrevals with positive detections
VIDEO_FRAMES = [(62,197),(229,447),(504,696),(747,867),(925,1196),(1237,1360)]

# number of the frames with possitive detections
true_detection =  sum([interval[1] - interval[0] + 1 for interval in VIDEO_FRAMES])


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=1,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        label = labels[int(float(class_names[i].decode("ascii")))]
        if scores[i] >= min_score and label=='person':
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(label, int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
              image_pil,
              ymin,
              xmin,
              ymax,
              xmax,
              color,
              font,
              display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image


if __name__=="__main__":
    loaded = tf.compat.v2.saved_model.load(str(model_path ), None)

    infer = loaded.signatures["serving_default"]

    video_path = sys.argv[1]

    prob_tresh = float(sys.argv[2])
    
    cap = cv2.VideoCapture(video_path)
    cap.open(video_path)
   
    i = 0 
    avg_time = 0
    model_detections = 0

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        while cap.isOpened():
            flag, frame = cap.read()
            i += 1
            if not flag:
                break

            key_pressed = cv2.waitKey(60)
            
            rgb_img  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x  = tf.image.convert_image_dtype(rgb_img, tf.uint8)[tf.newaxis, ...]
            
            infer_start = time.time()

            result = infer(x)

            # print('result here: ')
            # print(result)
            
            infer_duration = time.time() - infer_start
            #print("Inference time: %s" % infer_duration)

            if i == 1:
                avg_time  = infer_duration
            else:
                avg_time  = (infer_duration + avg_time) / 2

            detections = int(result['num_detections'].eval()[0])
            boxes = np.array(result['detection_boxes'].eval()[0])[:detections]

            class_names = np.array(result['detection_classes'].eval()[0]).astype("bytes")[:detections]
            scores = np.array(result['detection_scores'].eval()[0])[:detections]
            
            #calculate the number of persons detected
            s = class_names[scores >= prob_tresh]
            d = sum(s == b'1.0')

            #print(d)
            if d != 0:
                for interval in VIDEO_FRAMES:
                    if interval[0] <= i <= interval[1]:
                        model_detections += 1
                        break

            if key_pressed == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    print("Average time: {:.3f}ms".format(avg_time*1000))
    print("Accuracy: {:.3f}".format(model_detections/true_detection))