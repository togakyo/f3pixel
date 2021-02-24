# gmain-p8000-0000-a
#　機能名：＊＊＊＊＊


# import　ライブラリ
import sys
import argparse
import json
import glob

# external file, .h
from predictor import ScoringService
from PIL import Image
from timeit import default_timer as timer

# global variable 



# 関数名：
if __name__ == '__main__':
  """
Predict method

Args:
    input (str): path to the video file you want to make inference from

Returns:
     dict: Inference for the given input.
     format:
            - filename []:
            - category_1 []:
            - id: int
            - box2d: [left, top, right, bottom]
              ...
     Notes:
            - The categories for testing are "Car" and "Pedestrian".
              Do not include other categories in the prediction you will make.
            - If you do not want to make any prediction in some frames,
              just write "prediction = {}" in the prediction of the frame in the sequence(in line 65 or line 67).
  """
  
  
  #評価用の動画データをNo.0〜順番に読み込み
  test_data_path = '../data'
  videos = sorted(glob.glob(test_data_path+'/*.mp4'))


  # 検出したBounding Boxの座標を出力する
  # Output = [name_label, BoxA_X_min, BoxA_X_max, BoxA_Y_min, BoxA_Y_max]
  Output_list = ''
  for i in range(len(videos)):

    video_path = videos[i]
    ScoringService.get_model()
    Output = ScoringService.predict(video_path)
    print(Output)

    if i == 0:#最初はキーを指定して辞書作成
        Output_list = Output
    else:#2個目以降はキーを指定して辞書追加
        Output_list.update(Output)

    print("＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊")
  
    # Output douga mp4
    #ScoringService.get_model()
    #ScoringService.pw_outdouga(video_path)

  with open('../output/prediction.json', 'w') as f:
    json.dump(Output_list, f)
