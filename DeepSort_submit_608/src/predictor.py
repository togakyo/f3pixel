from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from Nyolo import YOLO

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

class ScoringService(object):

    @classmethod
    def get_model(cls, model_path='../model'):
        #本当は引数でモデルパスを渡したい
        cls.yolo = YOLO()
        
        # Definition of the parameters
        max_cosine_distance = 0.3
        nn_budget = None
        cls.nms_max_overlap = 1.0

        # Deep SORT
        model_filename = '../model_data/mars-small128.pb'
        cls.cencoder = gdet.create_box_encoder(model_filename, batch_size=1)
        cls.pencoder = gdet.create_box_encoder(model_filename, batch_size=1)

        cmetric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        pmetric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        cls.ctracker = Tracker(cmetric)
        cls.ptracker = Tracker(pmetric)

        cls.tracking = True
        cls.writeVideo_flag = True

        #推論したいカテゴリを設定
        cls.cl_list = ['Pedestrian', 'Car']

        return True

    @classmethod
    def predict(cls, input):

        predictions = []
        cap = cv2.VideoCapture(input)
        fname = os.path.basename(input)

        Nm_fr = 0
        
        fps = 0.0
        fps_imutils = imutils.video.FPS().start()

        while True:
            Car_result_ALL = []
            Pedestrian_result_ALL = []
            
            Nm_fr = Nm_fr + 1

            ret, frame = cap.read()

            if not ret:
                break

            im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(im_rgb)

            print(frame.shape)
            print("PROCESSING FRAME = ", Nm_fr)
            
            t1 = time.time()
            
            cboxes, cconfidence, cclasses = cls.yolo.detect_image(image, cls.cl_list[1])
            pboxes, pconfidence, pclasses = cls.yolo.detect_image(image, cls.cl_list[0])
            
            if cls.tracking:
                cfeatures = cls.cencoder(frame, cboxes)
                pfeatures = cls.pencoder(frame, pboxes)

                cdetections = [Detection(cbbox, cconfidence, ceach_class, cfeature) for cbbox, cconfidence, ceach_class, cfeature in \
                                 zip(cboxes, cconfidence, cclasses, cfeatures)]
                pdetections = [Detection(pbbox, pconfidence, peach_class, pfeature) for pbbox, pconfidence, peach_class, pfeature in \
                                 zip(pboxes, pconfidence, pclasses, pfeatures)]

                # Run non-maxima suppression.
                cboxes = np.array([d.tlwh for d in cdetections])
                cscores = np.array([d.confidence for d in cdetections])
                cindices = preprocessing.non_max_suppression(cboxes, cls.nms_max_overlap, cscores)
                cdetections = [cdetections[i] for i in cindices]

                pboxes = np.array([d.tlwh for d in pdetections])
                pscores = np.array([d.confidence for d in pdetections])
                pindices = preprocessing.non_max_suppression(pboxes, cls.nms_max_overlap, pscores)
                pdetections = [pdetections[i] for i in pindices]

                if cls.tracking:
                    # Call the tracker
                    cls.ctracker.predict()
                    cls.ctracker.update(cdetections)

                    cls.ptracker.predict()
                    cls.ptracker.update(pdetections)

                    for ctrack in cls.ctracker.tracks:
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

                    for ptrack in cls.ptracker.tracks:
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

                # Each frame result
                predictions.append({'Car': Car_result_ALL, 'Pedestrian': Pedestrian_result_ALL})
            
            fps_imutils.update()
            fps = (fps + (1./(time.time()-t1))) / 2
            print("     FPS = %f"%(fps))
            
        cap.release()
        fps_imutils.stop()
        print('imutils FPS: {}'.format(fps_imutils.fps()))
        
        return {fname: predictions}