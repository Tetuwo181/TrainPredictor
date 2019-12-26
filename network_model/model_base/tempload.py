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
            class_num: int,
            img_size: types_of_loco.input_img_size = 28,
            channels: int = 3,
            kernel_size: Union[int, Tuple[int, int]] = 3
            ) -> keras.engine.training.Model:
    """
    配下においてある中間ファイルのモデルを読み込む
    インターフェースを無理や知統一するため引数は使わない
    :param class_num: ダミー
    :param img_size: ダミー
    :param channels: ダミー
    :param kernel_size: ダミー
    :return:
    """
    model = load_model('temp.h5')
    # モデルの概要を表示
    model.summary()

    # モデルをコンパイル
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(), metrics=['accuracy'])

    return model
