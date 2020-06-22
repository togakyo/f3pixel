#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from Nyolo import YOLO
#from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video

import json
import glob
import os

warnings.filterwarnings('ignore')

@classmethod
def getValue(cls, key, items):
    values = [x['Value'] for x in items if 'Key' in x and 'Value' in x and x['Key'] == key]
    return values[0] if values else None

def main(yolo, input):

    #拡張子ありのファイル名
    basename = os.path.basename(input)
    print(" START YOLOv4 + DeepSort input file is ", basename)

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = './model_data/mars-small128.pb'
    cencoder = gdet.create_box_encoder(model_filename, batch_size=1)
    pencoder = gdet.create_box_encoder(model_filename, batch_size=1)

    cmetric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    pmetric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    ctracker = Tracker(cmetric)
    ptracker = Tracker(pmetric)

    tracking = True
    writeVideo_flag = True

    #推論したいカテゴリを設定
    cl_list = ['Pedestrian', 'Car']

    video_capture = cv2.VideoCapture(input)

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    if writeVideo_flag:
        basename_without_ext = os.path.splitext(os.path.basename(input))[0]
        fname = basename_without_ext +'output_yolov4.mp4'
        output_path = './output/'+ fname
        video_FourCC = int(video_capture.get(cv2.CAP_PROP_FOURCC))
        video_fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        frame_index = -1

    Nm_fr = 0
    all_result = []

    while True:

        Car_result_ALL = []
        Pedestrian_result_ALL = []

        Nm_fr = Nm_fr + 1

        ret, frame = video_capture.read()  # frame shape 1920*1216*3

        if ret != True:
            break

        print("Frame no. = ", Nm_fr)
        t1 = time.time()
        image = Image.fromarray(frame[...,::-1])

        cboxes, cconfidence, cclasses = yolo.detect_image(image, cl_list[1])
        pboxes, pconfidence, pclasses = yolo.detect_image(image, cl_list[0])

        if tracking:
            cfeatures = cencoder(frame, cboxes)
            pfeatures = pencoder(frame, pboxes)

            cdetections = [Detection(cbbox, cconfidence, ceach_class, cfeature) for cbbox, cconfidence, ceach_class, cfeature in \
                          zip(cboxes, cconfidence, cclasses, cfeatures)]
            pdetections = [Detection(pbbox, pconfidence, peach_class, pfeature) for pbbox, pconfidence, peach_class, pfeature in \
                          zip(pboxes, pconfidence, pclasses, pfeatures)]
            #else:
            #    detections = [Detection_YOLO(bbox, confidence, each_class) for bbox, confidence, each_class in \
            #                 zip(boxes, confidence, classes)]

            # Run non-maxima suppression.
            cboxes = np.array([d.tlwh for d in cdetections])
            cscores = np.array([d.confidence for d in cdetections])
            cindices = preprocessing.non_max_suppression(cboxes, nms_max_overlap, cscores)
            cdetections = [cdetections[i] for i in cindices]

            pboxes = np.array([d.tlwh for d in pdetections])
            pscores = np.array([d.confidence for d in pdetections])
            pindices = preprocessing.non_max_suppression(pboxes, nms_max_overlap, pscores)
            pdetections = [pdetections[i] for i in pindices]

            if tracking:
                # Call the tracker
                ctracker.predict()
                ctracker.update(cdetections)

                ptracker.predict()
                ptracker.update(pdetections)

                for ctrack in ctracker.tracks:
                    if not ctrack.is_confirmed() or ctrack.time_since_update > 1:
                        continue
                    cbbox = ctrack.to_tlbr()
                    cv2.rectangle(frame, (int(cbbox[0]), int(cbbox[1])), (int(cbbox[2]), int(cbbox[3])), (0, 0, 255), 3)
                    cv2.putText(frame, "ID: " + str(ctrack.track_id), (int(cbbox[0]), int(cbbox[1])), 0, \
                                1.5e-3 * frame.shape[0], (0, 0, 255), 3)

                    #OUTPUT TRACKING
                    ID      = int(ctrack.track_id)
                    left    = int(cbbox[0])
                    top     = int(cbbox[1])
                    right   = int(cbbox[2])
                    bottom  = int(cbbox[3])

                    Car_result = {'id': ID, 'box2d': [left,top,right,bottom]}#予測結果
                    print("Car_result = ", Car_result)
                    Car_result_ALL.append(Car_result)

                for ptrack in ptracker.tracks:
                    if not ptrack.is_confirmed() or ptrack.time_since_update > 1:
                        continue
                    pbbox = ptrack.to_tlbr()
                    cv2.rectangle(frame, (int(pbbox[0]), int(pbbox[1])), (int(pbbox[2]), int(pbbox[3])), (255, 0, 0), 3)
                    cv2.putText(frame, "ID: " + str(ptrack.track_id), (int(pbbox[0]), int(pbbox[1])), 0, \
                                1.5e-3 * frame.shape[0], (255, 0, 0), 3)

                    #OUTPUT TRACKING
                    ID      = int(ptrack.track_id)
                    left    = int(pbbox[0])
                    top     = int(pbbox[1])
                    right   = int(pbbox[2])
                    bottom  = int(pbbox[3])

                    Pedestrian_result = {'id': ID, 'box2d': [left,top,right,bottom]}#予測結果
                    print("Pedestrian_result = ", Pedestrian_result)
                    Pedestrian_result_ALL.append(Pedestrian_result)

            #YOLOv4 output to frame for Car
            for cdet in cdetections:
                cbbox = cdet.to_tlbr()
                cscore = "%.2f" % round(cdet.confidence * 100, 2) + "%"
                cv2.rectangle(frame, (int(cbbox[0]), int(cbbox[1])), (int(cbbox[2]), int(cbbox[3])), (255, 255, 255), 2)
                if len(cclasses) > 0:
                    ceach_class = cdet.cls
                    cv2.putText(frame, str(ceach_class) + " " + cscore, (int(cbbox[0]), int(cbbox[3])), 0, \
                                1.5e-3 * frame.shape[0], (255, 255, 255), 2)

            #YOLOv4 output to frame for Pedestrian
            for pdet in pdetections:
                pbbox = pdet.to_tlbr()
                pscore = "%.2f" % round(pdet.confidence * 100, 2) + "%"
                cv2.rectangle(frame, (int(pbbox[0]), int(pbbox[1])), (int(pbbox[2]), int(pbbox[3])), (127, 127, 127), 2)
                if len(pclasses) > 0:
                    peach_class = pdet.cls
                    cv2.putText(frame, str(peach_class) + " " + pscore, (int(pbbox[0]), int(pbbox[3])), 0, \
                                1.5e-3 * frame.shape[0], (127, 127, 127), 2)

            # Each frame result
            all_result.append({'Car': Car_result_ALL, 'Pedestrian': Pedestrian_result_ALL})

            if writeVideo_flag:
                # save a frame
                out.write(frame)
                frame_index = frame_index + 1

            fps_imutils.update()

            fps = (fps + (1./(time.time()-t1))) / 2
            print("     FPS = %f"%(fps))

    if writeVideo_flag:
        out.release()

    video_capture.release()

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    return {basename: all_result}

if __name__ == '__main__':

    #出力結果
    Output_list = ''

    #読みこむデータのパスを記載
    data_path = './data/train_videos'

    #複数のファイルに対応済み
    videos = sorted(glob.glob(data_path+'/*.mp4'))

    for i in range(len(videos)):
        video_path = videos[i]

        Output = main(YOLO(), video_path)
        print("Output = ", Output)

        if i == 0:#最初はキーを指定して辞書作成
            Output_list = Output
        else:#2個目以降はキーを指定して辞書追加
            Output_list.update(Output)

    print("＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊")
    with open('./output/prediction.json', 'w') as f:
        json.dump(Output_list, f)
