import sys
from predictor import ScoringService
from PIL import Image
import argparse
import json
import glob

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
  # class YOLO defines the default value, so suppress any default here
  parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
  '''
  Command line options
  '''
  #parser.add_argument(
  #  '--model_path', type=str,
  #  help='path to model weight file, default ' + ScoringService.get_defaults("model_path")
  #)

  #parser.add_argument(
  #  '--anchors_path', type=str,
  #  help='path to anchor definitions, default ' + ScoringService.get_defaults("anchors_path")
  #)

  #parser.add_argument(
  #  '--classes_path', type=str,
  #  help='path to class definitions, default ' + ScoringService.get_defaults("classes_path")
  #)

  #parser.add_argument(
  #  '--gpu_num', type=int,
  #  help='Number of GPU to use, default ' + str(ScoringService.get_defaults("gpu_num"))
  #)

  #FLAGS = parser.parse_args()
  data_path = 'data'#読みこむデータのパスを記載

  #複数のファイルに対応済み
  videos = glob.glob(data_path+'/*.mp4')
  Output_list = ''
  
  for i in range(len(videos)):
    video_path = videos[i]
    ScoringService.get_model()
    #Output = ScoringService.predict(ScoringService(**vars(FLAGS)), video_path)
    Output = ScoringService.predict(video_path)
    print(Output)
    
    if i == 0:#最初はキーを指定して辞書作成
        Output_list = Output
    else:#2個目以降はキーを指定して辞書追加
        Output_list.update(Output)

    print("＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊")

  with open('../output/prediction.json', 'w') as f:
    json.dump(Output_list, f)

