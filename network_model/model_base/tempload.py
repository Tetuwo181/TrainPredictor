from keras.applications import MobileNetV2
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras import optimizers
import keras.engine.training
from typing import Union
from typing import Tuple
from util_types import types_of_loco


def builder(
            temp_filepath: str
            ) -> keras.engine.training.Model:
    """
    配下においてある中間ファイルのモデルを読み込む
    インターフェースを無理や知統一するため引数は使わない
    :param temp_filepath: 中間ファイルのパス
    :return:
    """
    model = load_model(temp_filepath)
    # モデルの概要を表示
    model.summary()

    # モデルをコンパイル
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(), metrics=['accuracy'])

    return model
