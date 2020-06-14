#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video

import json
import glob

warnings.filterwarnings('ignore')

def main(yolo, input):

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Deep SORT
    model_filename = './model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    tracking = True
    writeVideo_flag = True

    #推論したいカテゴリを設定
    cl_list = ['Pedestrian', 'Car']

    predictions = []
    
    #使用しているベースソフトの制約で１クラスのTrackingしかできない
    Car_result_ALL = []
    Pedestrian_result_ALL = []
    all_result = []
    
    for i in range(len(cl_list)):
        video_capture = cv2.VideoCapture(input)

        fps = 0.0
        fps_imutils = imutils.video.FPS().start()
        if writeVideo_flag:
            fname = cl_list[i] + 'output_yolov4.mp4'
            output_path = './output/'+ fname
            video_FourCC = int(video_capture.get(cv2.CAP_PROP_FOURCC))
            video_fps = video_capture.get(cv2.CAP_PROP_FPS)
            video_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
            frame_index = -1

        while True:
            ret, frame = video_capture.read()  # frame shape 1920*1216*3
            if ret != True:
                break

            t1 = time.time()
            image = Image.fromarray(frame[...,::-1])
            
            cl_predstr = cl_list[i]
            boxes, confidence, classes = yolo.detect_image(image, cl_predstr)

            if tracking:
                features = encoder(frame, boxes)

                detections = [Detection(bbox, confidence, each_class, feature) for bbox, confidence, each_class, feature in \
                             zip(boxes, confidence, classes, features)]
            else:
                detections = [Detection_YOLO(bbox, confidence, each_class) for bbox, confidence, each_class in \
                             zip(boxes, confidence, classes)]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

            if tracking:
                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 10)
                    cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, \
                                1.5e-3 * frame.shape[0], (0, 0, 255), 3)
                    
                    ID     = int(track.track_id)   
                    top    = int(bbox[0]) 
                    left   = int(bbox[1]) 
                    bottom = int(bbox[2]) 
                    right  = int(bbox[3]) 

                    if cl_predstr == 'Car':
                        Car_result = {'id': ID, 'box2d': [left,top,right,bottom]}#予測結果
                        Car_result_ALL.append(Car_result)
                        
                    elif cl_predstr == 'Pedestrian':
                        Pedestrian_result = {'id': ID, 'box2d': [left,top,right,bottom]}#予測結果
                        Pedestrian_result_ALL.append(Pedestrian_result)

            for det in detections:
                bbox = det.to_tlbr()
                score = "%.2f" % round(det.confidence * 100, 2) + "%"
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 10)
                if len(classes) > 0:
                    each_class = det.cls
                    cv2.putText(frame, str(each_class) + " " + score, (int(bbox[0]), int(bbox[3])), 0, \
                                1.5e-3 * frame.shape[0], (255, 0, 0), 3)

            if writeVideo_flag:
                # save a frame
                out.write(frame)
                frame_index = frame_index + 1
            
            all_result = {'Car': Car_result_ALL, 'Pedestrian': Pedestrian_result_ALL}
            predictions.append({input: all_result})

            fps_imutils.update()

            fps = (fps + (1./(time.time()-t1))) / 2
            print(cl_predstr)
            print("     FPS = %f"%(fps))
            
        video_capture.release()
        
        fps_imutils.stop()
        print('imutils FPS: {}'.format(fps_imutils.fps()))

        if writeVideo_flag:
            out.release()
    
    return  


if __name__ == '__main__':

    #出力結果
    Output_list = ''

    #読みこむデータのパスを記載
    data_path = './data'
    
    #複数のファイルに対応済み
    videos = glob.glob(data_path+'/*.mp4')
    
    for i in range(len(videos)):
        video_path = videos[i]

        Output = main(YOLO(), video_path)
    
        if i == 0:#最初はキーを指定して辞書作成
            Output_list = Output
        else:#2個目以降はキーを指定して辞書追加
            Output_list.update(Output)
    
    print("＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊")
    with open('../output/prediction.json', 'w') as f:
        json.dump(Output_list, f)
