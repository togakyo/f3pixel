from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from cyolo import CYOLO
from pyolo import PYOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

import json
import glob
import os

warnings.filterwarnings('ignore')

class ScoringService(object):

    @classmethod
    def get_model(cls, model_path='../model'):
        #本当は引数でモデルパスを渡したい
        cls.cyolo = CYOLO()
        cls.pyolo = PYOLO()

        # Definition of the parameters for Car
        Cmax_cosine_distance = 0.3 # Param
        Cnn_budget = None # Param
        cls.Cnms_max_overlap = 1.0 # Param
        # Definition of the parameters for Pedestrian
        Pmax_cosine_distance = 0.3 # Param
        Pnn_budget = None # Param
        cls.Pnms_max_overlap = 1.0 # Param

        # Deep SORT
        model_filename = '../model/mars-small128.pb'
        cls.cencoder = gdet.create_box_encoder(model_filename, batch_size=1)
        cls.pencoder = gdet.create_box_encoder(model_filename, batch_size=1)

        cmetric = nn_matching.NearestNeighborDistanceMetric("euclidean", Cmax_cosine_distance, Cnn_budget) # Param
        pmetric = nn_matching.NearestNeighborDistanceMetric("cosine", Pmax_cosine_distance, Pnn_budget) # Param
        cls.ctracker = Tracker(cmetric)
        cls.ptracker = Tracker(pmetric)

        cls.tracking = True
        cls.writeVideo_flag = True

        #推論したいカテゴリを設定
        cls.cl_list = ['Car', 'Pedestrian']

        return True

    @classmethod
    def predict(cls, input):

        predictions = []
        cap = cv2.VideoCapture(input)
        fname = os.path.basename(input)

        if cls.writeVideo_flag:
            basename_without_ext = os.path.splitext(os.path.basename(input))[0]
            fmp4name = basename_without_ext +'output_DPSYv4.mp4'
            output_path = '../output/'+ fmp4name
            video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

        Nm_fr = 0
        #FPS 計算用
        FPS = 0
        tmpfrtime = 0

        while True:
            t1 = time.time()

            Car_result_ALL = []
            Pedestrian_result_ALL = []

            Nm_fr = Nm_fr + 1

            ret, frame = cap.read()

            if not ret:
                break
            
            #  Normal input 
            im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #  Equalize histgram of the value of each pixel
            #img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            # equalize the histogram of the Y channel
            #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            # convert the YUV image back to RGB format
            #img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

            image = Image.fromarray(im_rgb)
            #image = Image.fromarray(img_output)

            print(frame.shape)
            print("PROCESSING FRAME = ", Nm_fr)

            cboxes, cconfidence, cclasses = cls.cyolo.detect_image(image, cls.cl_list[0])
            pboxes, pconfidence, pclasses = cls.pyolo.detect_image(image, cls.cl_list[1])

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
                cindices = preprocessing.non_max_suppression(cboxes, cls.Cnms_max_overlap, cscores)
                cdetections = [cdetections[i] for i in cindices]

                pboxes = np.array([d.tlwh for d in pdetections])
                pscores = np.array([d.confidence for d in pdetections])
                pindices = preprocessing.non_max_suppression(pboxes, cls.Pnms_max_overlap, pscores)
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

                # save a frame
                if cls.writeVideo_flag:
                    out.write(frame)
            #End time
            t2 = time.time()
            frtime = t2 - t1
            print("Sec/Frame = ", frtime)
            tmpfrtime = tmpfrtime + frtime
            if tmpfrtime > 1:
                print("FPS = ", FPS)
                tmpfrtime = 0
                FPS = 0
            else:
                FPS = FPS + 1


        if cls.writeVideo_flag:
            out.release()
        cap.release()

        return {fname: predictions}
