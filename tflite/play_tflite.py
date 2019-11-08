# -*- coding: UTF-8 -*-

import cv2
import imutils
import io
import re
import time

from annotation import Annotator

import numpy as np
import picamera
import multiprocessing as mp
import traceback

from PIL import Image
from tflite_runtime.interpreter import Interpreter

#----------------------------------------------------

cam_id = 1
write_output = False
output_video_path = "output.avi"
input_width, input_height = 300, 300  #x,y
video_rate = 24.0
interval_inference = 3  #speed up the frame

threshold = 0.65
label_path = "12_palms/labels.txt"
tflite_path = "12_palms/12_palms_training_aware.tflite"

#multi-process
pool_result = []
pool_inference = mp.Pool(processes = 3)

#----------------------------------------------------

def fps_count(total_frames):
    global last_time, last_frames, fps

    timenow = time.time()
    if(timenow - last_time)>30:
        fps  = (total_frames - last_frames) / (timenow - last_time)
        print("FPS: {0}".format(fps))

        last_time  = timenow
        last_frames = total_frames

    #print("FPS: {0}".format(fps))

def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()

    return labels

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

'''
def detect_objects(interpreter, image, threshold):
    print("Inference start")
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)

    print(results)
    return results
'''

def annotate_objects(annotator, results, labels):
    """Draws the bounding box and label for each object in the results."""
    for obj in results:
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)

        # Overlay the box, label, and score on the camera preview
        annotator.bounding_box([xmin, ymin, xmax, ymax])
        annotator.text([xmin, ymin],
                   '%s\n%.2f' % (labels[obj['class_id']], obj['score']))

def handle_error(e):
    traceback.print_exception(type(e), e, e.__traceback__)

camera = cv2.VideoCapture(cam_id)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
print("This webcam's resolution is: %d x %d" % (width, height))

camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#fps count
start = time.time()
last_time = time.time()
last_frames = 0
fps = 0

if(write_output is True):
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, video_rate, (int(width),int(height)))


grabbed = True
frame_id = 0
if __name__ == '__main__':
    labels = load_labels(label_path)
    interpreter = Interpreter(tflite_path)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']


    def detect_objects(interpreter, image, threshold):
        print("Inference start")
        """Returns a list of detection results, each a dictionary of object info."""
        set_input_tensor(interpreter, image)
        interpreter.invoke()

        # Get all output details
        boxes = get_output_tensor(interpreter, 0)
        classes = get_output_tensor(interpreter, 1)
        scores = get_output_tensor(interpreter, 2)
        count = int(get_output_tensor(interpreter, 3))

        results = []
        for i in range(count):
            if scores[i] >= threshold:
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
                }
                results.append(result)

        print(results)
        return results


    while grabbed:
        frame_id += 1
        (grabbed, frame) = camera.read()

        #stream = io.BytesIO()
        #stream.seek(0)
        #image = Image.open(frame).convert('RGB').resize(
        #    (input_width, input_height), Image.ANTIALIAS)
        frame2 = cv2.resize(frame, (input_width, input_height))

        results = []
        if(frame_id % interval_inference == 0):
            #results = detect_objects(interpreter, frame2[...,::-1], threshold)
            proc_inference = pool_inference.apply_async(detect_objects, ((interpreter, frame2[...,::-1], threshold), ), error_callback=handle_error)
            print("Return:", proc_inference)

        if(len(results)>0):
            print(results[0]["bounding_box"], results[0]["class_id"], results[0]["score"])

        if(write_output is True):
            print("write frame id #{} ({}x{}) to file".format(i, frame.shape[1], frame.shape[0]))
            out.write(frame2)

        cv2.imshow("webcam id:"+str(cam_id), imutils.resize(frame, height=300))
        cv2.waitKey(1)

        fps_count(frame_id)

    out.release()
    camera.release()

