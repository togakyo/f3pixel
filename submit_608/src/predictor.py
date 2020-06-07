import glob
import cv2
import pickle

import colorsys

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from model import yolo_eval, yolo_body, tiny_yolo_body
from utils import letterbox_image
from timeit import default_timer as timer

import os
import copy

class ScoringService(object):
    IDvalue_car = 0 # Car クラス変数を宣言 = 0
    IDvalue_ped = 0 # Pedestrian クラス変数を宣言 = 0

    old_top = 0
    old_left = 0
    old_bottom = 0
    old_right = 0
    ObjID_setimg = 0

    #前フレームで検出した結果を格納
    all_ObjectID_pos = []
    all_ObjectID_oldpos = []
    CurrentObjectID = []

    @classmethod
    def get_model(cls, model_path='../model'):
        modelpath = os.path.join(model_path, 'YOLOv3_608_cl2_ep013_val_loss51.h5')

        class_names = cls._get_class()
        anchors = cls._get_anchors()

        # Load model, or construct model and load weights.
        num_anchors = len(anchors)
        num_classes = len(class_names)
        is_tiny_version = num_anchors == 6  # default setting
        cls.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                         if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
        cls.yolo_model.load_weights(modelpath)  # make sure model, anchors and classes match
        return True

    @classmethod
    def predict(cls, input):

        cls.IDvalue_car = 0 # Reset Object ID
        cls.IDvalue_ped = 0 #

        cls.all_ObjectID_pos = []
        cls.all_ObjectID_oldpos = []

        predictions = []
        cap = cv2.VideoCapture(input)
        fname = os.path.basename(input)

        Nm_fr = 0

        while True:
            Nm_fr = Nm_fr + 1

            ret, frame = cap.read()
            if not ret:
                break

            im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(im_rgb)

            print(frame.shape)
            print("PROCESSING FRAME = ", Nm_fr)
            prediction = cls.detect_image(image, Nm_fr, cls.all_ObjectID_pos, cls.all_ObjectID_oldpos)

            predictions.append(prediction)

        cap.release()
        return {fname: predictions}

    @classmethod
    def detect_image(cls, image, frame_num, all_posinf, old_posinf):
        start = timer()

        model_image_size = (608, 608)
        class_names = cls._get_class()

        new_image_size = (image.width - (image.width % 32),
                                image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        image_shape = [image.size[1], image.size[0]]

        out_boxes, out_scores, out_classes = cls.compute_output(image_data, image_shape)

        Car_result_ALL = []
        Pedestrian_result_ALL = []
        all_result = []

        # フレーム単位の処理
        if frame_num > 1:
            old_posinf.clear()
            old_posinf = copy.copy(all_posinf)
            all_posinf.clear()

        #オブジェクト単位の処理
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            #JSON 形式の時はint32()未対応のため -> int()に変換する
            top = int(top)
            left = int(left)
            bottom = int(bottom)
            right = int(right)

            #2 検出したbox_sizeを計算する　設定した閾値1024pix**2
            sq_bdbox = (bottom - top)*(right - left)

            #3 検出したboxの中心点の座標を計算する
            center_bdboxX = int((bottom - top)/2) + top
            center_bdboxY = int((right - left)/2) + left

            if sq_bdbox >= 1024:#矩形サイズの閾値
                if predicted_class == 'Car':
                    ObjID_set = 0

                    if frame_num == 1:#1フレーム目は全て登録する
                        cls.IDvalue_car = cls.IDvalue_car + 1
                        #車を検出した時
                        ObjID_set = cls.IDvalue_car
                        Car_result = {'id': ObjID_set, 'box2d': [left,top,right,bottom]}#予測結果
                        #予測結果より次のFrameの物体位置を予測する情報を作成
                        tmp_car = {'frame':frame_num,'id':ObjID_set, 'left':left, 'top':top, 'right':right, 'bottom':bottom}
                        all_posinf.append(tmp_car)
                    else:
                        #current_pos check
                        cls.matches_cnt = 0

                        for kt in range(len(old_posinf)):
                            tmp_old_pos = old_posinf[kt]

                            tmp_ObjID = 0
                            tmp_left = 0
                            tmp_top = 0
                            tmp_right = 0
                            tmp_bottom = 0

                            for k, v in tmp_old_pos.items():
                                # k= Tanaka v= 80 // Tanaka: 80
                                if k == "id":
                                    print("Key = ", k)
                                    print("Value = ",v)
                                    tmp_ObjID = v
                                elif k == "left":
                                    print("Key = ", k)
                                    print("Value = ",v)
                                    tmp_left = v
                                elif k == "top":
                                    print("Key = ", k)
                                    print("Value = ",v)
                                    tmp_top = v
                                elif k == "right":
                                    print("Key = ", k)
                                    print("Value = ",v)
                                    tmp_right = v
                                elif k == "bottom":
                                    print("Key = ", k)
                                    print("Value = ",v)
                                    tmp_bottom = v
                            if (tmp_left <= center_bdboxX <= tmp_right) and (tmp_top <= center_bdboxY <= tmp_bottom):
                                ObjID_set = tmp_ObjID
                                cls.matches_cnt = cls.matches_cnt + 1
                                #該当する
                            #else:

                        #もしどのIDにも当てはまらない場合
                        if cls.matches_cnt == 0:
                            cls.IDvalue_car = cls.IDvalue_car + 1
                            ObjID_set = cls.IDvalue_car
                        #else:
                            #ObjID_set = tmp_ObjID

                        #更新したObjIDを登録する
                        tmp_car = {'frame':frame_num,'id':ObjID_set, 'left':left, 'top':top, 'right':right, 'bottom':bottom}
                        all_posinf.append(tmp_car)

                        #車を検出した時
                        Car_result = {'id': ObjID_set, 'box2d': [left,top,right,bottom]}#予測結果

                    #検出したオブジェクトを格納 検出しない場合は空欄が格納される
                    Car_result_ALL.append(Car_result)#車

                elif predicted_class == 'Pedestrian':
                    cls.IDvalue_ped = cls.IDvalue_ped + 1
                    #歩行者を検出した時
                    Pedestrian_result = {'id': int(cls.IDvalue_ped), 'box2d': [left,top,right,bottom]}#予測結果

                    #予測結果より次のFrameの物体位置を予測する情報を作成
                    tmp_ped = {'frame':frame_num,'id':int(cls.IDvalue_ped), 'left':left, 'top':top, 'right':right, 'bottom':bottom}
                    cls.all_ObjectID_pos.append(tmp_ped)

                    #検出したオブジェクトを格納 検出しない場合は空欄が格納される
                    Pedestrian_result_ALL.append(Pedestrian_result)#歩行者

        all_result = {'Car': Car_result_ALL, 'Pedestrian': Pedestrian_result_ALL}
        end = timer()
        print("1フレームの処理時間 = ", end - start)
        return all_result

    @classmethod
    def pw_outdouga(cls, input):
        cls.IDvalue_car = 0 # Reset Object ID
        cls.IDvalue_ped = 0 #

        cls.all_ObjectID_pos = []
        cls.all_ObjectID_oldpos = []

        cap = cv2.VideoCapture(input)
        fname = os.path.basename(input)

        output_path = "../output/" + "result_"+ fname
        video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        print("video_FourCC = " , video_FourCC)#1983148141
        print("video_fps = " , video_fps)  # 5.0
        print("video_size = " , video_size) # (1936, 1216)
        # output
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

        Nm_fr = 0

        while True:
            Nm_fr = Nm_fr + 1

            ret, frame = cap.read()
            if not ret:
                break

            im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(im_rgb)

            print(frame.shape)
            print("PROCESSING FRAME = ", Nm_fr)
            #image = cls.ret_frame(image, Nm_fr, cls.all_ObjectID_pos, cls.all_ObjectID_oldpos)
            image = cls.ret_frame(image, Nm_fr)
            # ここでフレーム毎＝画像イメージ毎に動画をバラしている

            result = np.asarray(image)
            im_rgbresult = result[:, :, [2, 1, 0]]

            if ret == True:
                out.write(im_rgbresult)

        cap.release()

    @classmethod
    def getValue(cls, key, items):
        values = [x['Value'] for x in items if 'Key' in x and 'Value' in x and x['Key'] == key]
        return values[0] if values else None

    @classmethod
    def ret_frame(cls, image, frame_num):
        hsv_tuples = [(x / 10, 1., 1.) for x in range(10)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        start = timer()

        model_image_size = (608, 608)
        class_names = cls._get_class()

        new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        image_shape = [image.size[1], image.size[0]]

        out_boxes, out_scores, out_classes = cls.compute_output(image_data, image_shape)

        font = ImageFont.truetype(font='../box_font/FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        # フレーム単位の処理
        if frame_num > 1:
            cls.all_ObjectID_oldpos = copy.copy(cls.all_ObjectID_pos)
            #cls.all_ObjectID_pos.clear()
            cls.all_ObjectID_pos = []

        # オブジェクト単位の処理
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            #JSON 形式の時はint32()未対応のため -> int()に変換する
            top, left, bottom, right = box
            top    = int(max(0, np.floor(top + 0.5).astype('int32')))
            left   = int(max(0, np.floor(left + 0.5).astype('int32')))
            bottom = int(min(image.size[1], np.floor(bottom + 0.5).astype('int32')))
            right  = int(min(image.size[0], np.floor(right + 0.5).astype('int32')))

            #2 検出したbox_sizeを計算する　設定した閾値1024pix**2
            sq_bdbox = (bottom - top)*(right - left)

            cls.ObjID_setimg = 0
            cls.matches_cnt = 0
            print("len(cls.all_ObjectID_oldpos) = ", len(cls.all_ObjectID_oldpos))

            if sq_bdbox >= 1024:#矩形サイズの閾値 1024
                #if predicted_class == 'Car'or predicted_class == 'Pedestrian':# Car or Pedes
                if predicted_class == 'Car':

                    print("cls.all_ObjectID_oldpos = ", cls.all_ObjectID_oldpos)

                    #3 検出したboxの中心点座標を計算する
                    center_bdboxX = int((bottom - top)/2) + top
                    center_bdboxY = int((right - left)/2) + left

                    if frame_num == 1:#1フレーム目は全て登録する
                        cls.IDvalue_car = cls.IDvalue_car + 1
                        cls.ObjID_setimg = cls.IDvalue_car
                    else:
                        for kt in range(len(cls.all_ObjectID_oldpos)):
                            tmp_old_pos = cls.all_ObjectID_oldpos[kt]
                            print("tmp_old_pos = ", tmp_old_pos)

                            cls.ObjID_setimg = cls.getValue('id', tmp_old_pos)
                            cls.old_left = cls.getValue('left', tmp_old_pos)
                            cls.old_top  = cls.getValue('top', tmp_old_pos)
                            cls.old_right = cls.getValue('right', tmp_old_pos)
                            cls.old_bottom = cls.getValue('bottom', tmp_old_pos)

                            print("ObjID_setimg = ", cls.ObjID_setimg)
                            print("old_left = ", cls.old_left)
                            print("old_top = ",cls.old_top)
                            print("old_right = ",cls.old_right)
                            print("old_bottom = ",cls.old_bottom)

                            band_value = 5
                            exp_old_left = int(cls.old_left - band_value)
                            exp_old_top = int(cls.old_top - band_value)
                            exp_old_right = int(cls.old_right + band_value)
                            exp_old_bottom = int(cls.old_bottom + band_value)

                            print("center_bdboxX = ", center_bdboxX)
                            print("center_bdboxY = ", center_bdboxY)

                            print("exp_old_left = ", exp_old_left)
                            print("exp_old_top = ", exp_old_top)
                            print("exp_old_right = ",exp_old_right)
                            print("exp_old_bottom = ",exp_old_bottom)

                            if(( center_bdboxX >= exp_old_left ) or ( center_bdboxX <= exp_old_right )):
                                if(( center_bdboxY >= exp_old_top) or ( center_bdboxY <= exp_old_bottom )):
                                    cls.matches_cnt = cls.matches_cnt + 1

                        #前回フレームより過去のオブジェクトを全てチェックした結果を出力
                        print("cls.matches_cnt = ", cls.matches_cnt)

                        #もしどのIDにも当てはまらない場合
                        if cls.matches_cnt == 0:
                            cls.old_top = 0
                            cls.old_left = 0
                            cls.old_bottom = 0
                            cls.old_right = 0

                            cls.IDvalue_car = cls.IDvalue_car + 1
                            cls.ObjID_setimg = cls.IDvalue_car

                    #更新したObjIDを登録する
                    tmp_car = [{'Key':'frame',  'Value':frame_num},
                               {'Key':'id',     'Value':cls.ObjID_setimg},
                               {'Key':'left',   'Value':left},
                               {'Key':'top',    'Value':top},
                               {'Key':'right',  'Value':right},
                               {'Key':'bottom', 'Value':bottom}]
                    cls.all_ObjectID_pos.append(tmp_car)

                    label = '{}_{:.2f}_{}'.format(predicted_class, score, str(cls.ObjID_setimg))#put the ID for each obj
                    draw = ImageDraw.Draw(image)
                    label_size = draw.textsize(label, font)
                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    if cls.old_top - label_size[1] >= 0:
                        text_origin2 = np.array([cls.old_left, cls.old_top - label_size[1]])
                    else:
                        text_origin2 = np.array([cls.old_left, cls.old_top + 1])

                    # My kingdom for a good redistributable image drawing library.
                    for i in range(thickness):
                        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[0])
                    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[0])
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)

                    for i in range(thickness):
                        draw.rectangle([cls.old_left + i, cls.old_top + i, cls.old_right - i, cls.old_bottom - i], outline=colors[1])
                    draw.rectangle([tuple(text_origin2), tuple(text_origin2 + label_size)], fill=colors[1])
                    draw.text(text_origin2, label, fill=(0, 0, 0), font=font)

                    del draw

        end = timer()
        print("1フレームの処理時間 = ", end - start)
        return image

    @classmethod√/§´≤√
    def _get_class(cls, model_path='../src'):
        classes_path = os.path.join(model_path, 'voc_2classes.txt')

        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    @classmethod
    def _get_anchors(cls, model_path='../src'):
        anchors_path = os.path.join(model_path, '2020_yolo_anchors9_FOC2_allimg.txt')

        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    @classmethod
    def compute_output(cls, image_data, image_shape):
        input_image_shape = tf.constant(image_shape)

        class_names = cls._get_class()
        anchors = cls._get_anchors()
        iou = 0.2    #Adjust param
        score = 0.8   #Adjust param

        boxes, scores, classes = yolo_eval(cls.yolo_model(image_data), anchors,
                                             len(class_names), input_image_shape,
                                             score_threshold=score, iou_threshold=iou)
        return boxes, scores, classes
