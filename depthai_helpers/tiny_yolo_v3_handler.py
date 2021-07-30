from math import exp as exp
import cv2
import numpy as np
from time import time
import json
import math

from io import BytesIO
import depthai
import gtts
import pyttsx3 #offline
import speech_recognition as sr
from playsound import playsound

import sys
import keyboard

from openal import *
import time
import msvcrt
import os

class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    def __init__(self, side, mask, coords, classes, anchors):
        self.num = 3 
        self.coords = coords 
        self.classes = classes
        self.anchors = anchors

        self.num = len(mask)

        maskedAnchors = []
        for idx in mask:
            maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
        self.anchors = maskedAnchors
        self.side = side
   
    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)



def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    #replace side , params.side with out_blob_h
    side_square = out_blob_h **2



    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // out_blob_h
        col = i % out_blob_h
        for n in range(params.num):
            obj_index = entry_index(out_blob_h, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            
            if scale < threshold:
                continue
            box_index = entry_index(out_blob_h, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / out_blob_h
            y = (row + predictions[box_index + 1 * side_square]) / out_blob_h

            
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w) #if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h)# if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(out_blob_h, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
    return objects


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


def decode_tiny_yolo(nnet_packet, **kwargs):
    NN_metadata = kwargs['NN_json']
    output_format = NN_metadata['NN_config']['output_format']

    in_layers = nnet_packet.getInputLayersInfo()
    # print(in_layers)
    input_width  = in_layers[0].get_dimension(depthai.TensorInfo.Dimension.W)
    input_height = in_layers[0].get_dimension(depthai.TensorInfo.Dimension.H)

    if output_format == "detection":
        detections = nnet_packet.getDetectedObjects()
        objects = list()
        # detection_nr = detections.size
        # for i in range(detection_nr):
        #     detection =detections[i]
        for detection in detections:
            confidence = detection.confidence
            class_id = detection.label
            xmin = int(detection.x_min * input_width)
            xmax = int(detection.x_max * input_width)
            ymin = int(detection.y_min * input_height)
            ymax = int(detection.y_max * input_height)
            depth_x = detection.depth_x
            depth_y = detection.depth_y
            depth_z = detection.depth_z
            scaled_object = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence, depth_x=depth_x, depth_y=depth_y, depth_z=depth_z)
            objects.append(scaled_object)

        return objects
    else:
        
        output_list = nnet_packet.getOutputsList()

        objects = list()
        resized_image_shape =[input_width,input_height]
        original_image_shape =[input_width,input_height]
        iou_threshold = NN_metadata['NN_config']['NN_specific_metadata']['iou_threshold']


        start_time = time()
        for out_blob in output_list:
            side = out_blob.shape[2]
            side_str = 'side' + str(side)
            mask = NN_metadata['NN_config']['NN_specific_metadata']['anchor_masks'][side_str]
            coords = NN_metadata['NN_config']['NN_specific_metadata']['coordinates']
            classes = NN_metadata['NN_config']['NN_specific_metadata']['classes']
            anchors = NN_metadata['NN_config']['NN_specific_metadata']['anchors']
            detection_threshold = NN_metadata['NN_config']['NN_specific_metadata']['confidence_threshold']
            
            l_params = YoloParams(side, mask, coords, classes, anchors)
            objects += parse_yolo_region(out_blob,  resized_image_shape,
                                                original_image_shape, l_params,
                                                detection_threshold)
            parsing_time = time() - start_time

        # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
        objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
        for i in range(len(objects)):
            if objects[i]['confidence'] == 0:
                continue
            for j in range(i + 1, len(objects)):
                if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                        objects[j]['confidence'] = 0
        
        objects = [obj for obj in objects if obj['confidence'] >= detection_threshold]

        return objects

def decode_tiny_yolo_json(nnet_packet, **kwargs):
    convertList = []

    filtered_objects = decode_tiny_yolo(nnet_packet, **kwargs)
    for entry in filtered_objects:
        jsonConvertDict = {}
        jsonConvertDict["xmin"] = entry["xmin"]
        jsonConvertDict["ymin"] = entry["ymin"]
        jsonConvertDict["xmax"] = entry["xmax"]
        jsonConvertDict["ymax"] = entry["ymax"]
        if type(entry["confidence"]) is np.float16:
            jsonConvertDict["confidence"] = entry["confidence"].item()
        else:
            jsonConvertDict["confidence"] = entry["confidence"]
        jsonConvertDict["class_id"] = entry["class_id"]
        convertList.append(jsonConvertDict)

    return json.dumps(convertList)

BOX_COLOR = (0,255,0)
LABEL_BG_COLOR = (70, 120, 70) # greyish green background for text
TEXT_COLOR = (255, 255, 255)   # white text
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

def show_tiny_yolo(filtered_objects, frame, **kwargs):
    objects_found = []
    objectsDetected = []
    
    NN_metadata = kwargs['NN_json']
    labels = NN_metadata['mappings']['labels']
    config = kwargs['config']

    for detection in filtered_objects:
        
        # get all values from the filtered object list
        xmin = detection['xmin']
        ymin = detection['ymin']
        xmax = detection['xmax']
        ymax = detection['ymax']
        confidence = detection['confidence']
        class_id = detection['class_id']
        
        # Set up the text for display
        cv2.rectangle(frame,(xmin, ymin), (xmax, ymin+20), LABEL_BG_COLOR, -1)
        cv2.putText(frame, labels[class_id] + ': %.2f' % confidence, (xmin+5, ymin+15), TEXT_FONT, 0.5, TEXT_COLOR, 1)
        # Set up the bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), BOX_COLOR, 1)
        if config['ai']['calc_dist_to_bb']:
            depth_x = detection['depth_x']
            depth_y = detection['depth_y']
            depth_z = detection['depth_z']

            # print("hello-----------")
            # print("values for ", labels[class_id], " is ", depth_x, depth_y, depth_z )
            objects_found.append(labels[class_id])

            objectsDetected.append([labels[class_id], depth_x, depth_y, depth_z])
            navigation(objectsDetected)
            # textToSpeech(objects_found, depth_z)
            cv2.putText(frame, 'x Current:' '{:7.3f}'.format(depth_x) + ' m', (xmin, ymin+60),  TEXT_FONT, 0.5, TEXT_COLOR)
            cv2.putText(frame, 'y Current:' '{:7.3f}'.format(depth_y) + ' m', (xmin, ymin+80),  TEXT_FONT, 0.5, TEXT_COLOR)
            cv2.putText(frame, 'z Current:' '{:7.3f}'.format(depth_z) + ' m', (xmin, ymin+100), TEXT_FONT, 0.5, TEXT_COLOR)
    return frame

r = sr.Recognizer()
engine = pyttsx3.init()
volume = engine.getProperty('volume')
engine.setProperty('volume', volume + 1.50)



def navigation (details):
    print("details----", details)
    source_01 = details[1:]
    source_02 = [3,3,4]
    #------ detect bottle with x y z position
    #sound program 
    # source_01 = [depth_x]

    source_list = []
    t_source1_0 = 0
    t_source2_0 = 0

    # source_01 = [-1,-1,4]
    # source_02 = [3,3,4]

    
    initialtime_list = []
    #finaltime_list = []

    initial = time.time()

    def Play_Beep(source_pos,beep_filename):
        my_sound = oalOpen(beep_filename)           # source
        my_sound.set_position(source_pos)
        my_dest = oalGetListener()               # listener/destination
        my_dest.move_to([0,0,0])
        my_sound.play()
        while my_sound.get_state() == AL_PLAYING:
            # wait until the file is done playing
            time.sleep(0.2)
        del my_sound
        del my_dest
        
    # Manipulate list parameters:
    oalInit()
    while True:
        if source_01 not in source_list:
            source_list.append(source_01)
            initialtime_list.append(t_source1_0)
        if source_02 not in source_list:
            source_list.append(source_02)
            initialtime_list.append(t_source2_0)
        
        t_source1_1 = time.time()
        t_source2_1 = time.time()
        finaltime_list = [t_source1_1,t_source2_1]
        

        # print("source--list---->", source_list)
        #iterate through all sources
        index = 0
        # source_list = set(source_list)
        for source_object in source_list:
            # get distance value
            time_function = 1    # change to a function dependent on distance (exponentially)
            if finaltime_list[index] - initialtime_list[index] >= time_function:
                if index<10:
                    source_index = "_0" + str(index)
                else:
                    source_index = "_" + str(index)
        #             print("filename----", filename)
                filename = "Beep_frequencies" + os.sep + "Beep" + source_index + ".ogg"
                Play_Beep(source_object,filename)
            index+=1


            if keyboard.is_pressed('ENTER'):
                print("you pressed Enter, so exiting program..")
                sys.exit(0)
                oalQuit()
                break
        source_list = []


# for text to speech conversion
def textToSpeech(label_lists, depth_z):
    
    distance = depth_z
    # voices = engine.getProperty('voices')
    # for voice in voices:
    #     print("Voice: %s" % voice.name)
    #     print("-ID: %s" % voice.id)
    #     print("-GENDER: %s" % voice.gender)
    #     print("-LANGUAGES: %s" % voice.languages)
    #     print("AGE: %s" % voice.age)
    #     print("\n")
    
    # distance = 0
    # distance = math.sqrt((xmax-xmin)**2 +( ymax-ymin)**2) #euclidean distance
    # print("distance of object from camera is ", distance, "cm")
    # mp3fIO = BytesIO()
    # # print(arr_label, type(arr_label))
    # for i in range(len(arr_label)):
    #     print("word", arr_label[i])
    #     tts = gtts.gTTS(arr_label[i]) #lang='kn'/"bn"
    #     tts.save("objects detected.mp3")
    #     # tts.write_to_fp(mp3fIO) 
    #     playsound("objects detected.mp3")
    counter = 0
    while True:
        counter +=1
        print('counter', counter)
        with sr.Microphone() as source:
            
            guidanceVoice = "Speak Loud and clear"
            
            engine.say(guidanceVoice)
            engine.runAndWait()
        # read the audio data from the default microphone
            audio_data = r.record(source, duration=5)
            guidanceVoice = "Thank you"
            engine.say(guidanceVoice)
            engine.runAndWait()
            print("Recognizing... speech ")
            # convert speech to text
            text = r.recognize_google(audio_data)
            
            print(" and searching object from my voice ...... ", text.upper())
            arr_label = label_lists
            print(">>>>> objects detected by oak camera >>>> ", arr_label)
            voice_objects = text.split(" ")
            print("voice objects are ....", voice_objects)

            for i in range(len(voice_objects)):
                for j in range(len(arr_label)):
                    
                    if voice_objects[i] == arr_label[j]:
                        guidanceVoice = voice_objects[i] + "can be detected by oak d camera and is at ", distance, "metres away"
                        engine.say(guidanceVoice)
                        engine.runAndWait()
                    else:
                        pass
                # if (text in arr_label):
                #     guidanceVoice = text + "can be detected by oak d camera and is at ", distance, "metres away"
                #     engine.say(guidanceVoice)
                #     engine.runAndWait()
                # else:
                #     guidanceVoice = "Object cannot be found"
        #     engine.say(guidanceVoice)
        #     engine.runAndWait()
       


# >python depthai_demo.py -cnn yolo-v3