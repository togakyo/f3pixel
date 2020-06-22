from __future__ import division, print_function, absolute_import

from predictor import ScoringService

import json
import glob
import os

if __name__ == '__main__':

  data_path = '../data'#読みこむデータのパスを記載

  #複数のファイルに対応済み
  videos = sorted(glob.glob(data_path+'/*.mp4'))
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

  with open('../output/prediction.json', 'w') as f:
    json.dump(Output_list, f)
