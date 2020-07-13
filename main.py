"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import argparse
import time
import socket
import json
import cv2
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
cpu_extension = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

# count of the number of frames to count before it is deemed as a new person
COUNTER_THRESHOLD = 30
REQUEST_ID = 0
TOPIC_DURATION = "person/duration"
TOPIC_PERSON = "person"
ESC_KEY = 27

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    return parser


def connect_mqtt():
    ### DONE: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def preprocess_image(input_image, h, w):
    '''
    Given an input image, height and width:
    - Resize to height and width
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start
    '''
    image = cv2.resize(input_image, (w, h))
    image = image.transpose((2, 0, 1))
    image = image.reshape(1, *image.shape)

    return image

# inspiration from this https://stackoverflow.com/questions/61537074/bounding-boxes-not-showing-for-fasterrcnn-model
# for output shape, the author referred to here:
# https://stackoverflow.com/questions/59471526/input-and-output-format-of-tensorflow-models
def create_bounding_box(frame, result, width, height, prob_threshold):
    '''
    Draw bounding boxes onto the frame.
    '''
    global frame_count, location, bounding_box
    current_count = 0
    current_count_total = 0
    for box in result[0][0]:
        conf = box[2]

        if conf >= prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

            x1 = (xmax - xmin) / 2
            y1 = (ymax - ymin) / 2
            cx = xmin + x1
            cy = ymin + y1
            cv2.circle(frame, (int(cx), int(cy)), 2, (255, 0, 0), -1)
            location.append([cx, cy])
            current_count += 1
            bd = np.asarray(location)
            if len(bd) > 1:
                diff = bd[-1] - bd[-2]
                if abs(diff[0]) > COUNTER_THRESHOLD and abs(diff[1]) > COUNTER_THRESHOLD:
                    current_count_total += 1
        else:
            bounding_box.append(0)
    return len(location), current_count, frame, current_count_total


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    inference_network = Network()

    # variable declarations
    last_count_total = 0
    total_count = 0
    last_duration = 0

    # extract arguments from running to console
    model = args.model
    DEVICE = args.device
    CPU_EXTENSION = args.cpu_extension

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### DONE: Load the model through `infer_network` ###
    inference_network.load_model(model, DEVICE, CPU_EXTENSION)

    net_input_shape = inference_network.get_input_shape()

    ### DONE: Handle the input stream ###
    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_stream = args.input
    elif (not args.input.endswith('.jpg')) or (not (args.input.endswith('.bmp'))):
        input_stream = args.input
        assert os.path.isfile(args.input), "Input file does not exist"
    else:
        input_stream = args.input
        log.error("The file is unsupported.please pass a supported file")

    ### DONE: Handle the input stream ###
    cap = cv2.VideoCapture(input_stream)

    if input_stream:
        cap.open(input_stream)

    if not cap.isOpened():
        log.error('Input error. Check the image or video feed.')

    width = int(cap.get(3))
    height = int(cap.get(4))

    ### DONE: Loop until stream is over ###
    while cap.isOpened():
        ### DONE: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        key_pressed = cv2.waitKey(60)

        inference_start = time.time()

        ### DONE: Pre-process the image as needed ###

        image_processed = preprocess_image(frame, net_input_shape[2], net_input_shape[3])

        ### DONE: Start asynchronous inference for specified request ###
        inference_network.exec_net(REQUEST_ID, image_processed)

        ### DONE: Wait for the result ###
        if inference_network.wait(REQUEST_ID) == 0:
            latency = time.time() - inference_start
            ### DONE: Get the results of the inference request ###
            inference_output = inference_network.get_output(REQUEST_ID)

            # add the message to the frame
            bounding_box_message = "Latency: {:.2f}ms".format(latency * 1000)
            cv2.putText(frame, bounding_box_message, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 10, 0), 1)

            ### DONE: Extract any desired stats from the results ###
            centroid, current_count, out_frame, current_count_total = \
                create_bounding_box(frame, inference_output, width, height, prob_threshold)

            ### DONE: Calculate and send relevant information on ###
            # this is to ensure that the total count is not affected by new counts by a certain threshold that will
            if current_count_total > last_count_total:
                total_count = total_count + current_count_total - last_count_total

            if total_count > 0:
                # Publish messages to the MQTT server (assuming taht there is a new person detected carried by the total
                # count above and its duration recorded)
                if last_duration < total_count:
                    duration = float(time.time())
                    client.publish(TOPIC_DURATION, json.dumps({"duration": duration}))

            client.publish(TOPIC_PERSON, json.dumps({"count": current_count, "total": total_count}))
            last_count_total = current_count_total
            last_duration = total_count

        ### DONE: Send the frame to the FFMPEG server ###
        #  Resize the frame
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        # check for escape
        if key_pressed == ESC_KEY:
            break

    cap.release()
    cv2.destroyAllWindows()

    ### DONE: Disconnect from MQTT
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()