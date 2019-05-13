# -*- coding: utf-8 -*-
import keras.engine.training
from typing import Callable
from typing import Union
from typing import Tuple
from util_types import types_of_loco
import importlib


def builder(
            class_num: int,
            img_size: types_of_loco.input_img_size = 28,
            channels: int = 3,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            model_name: str = "model1"
            ) -> keras.engine.training.Model:
    """
    モデルを作成する
    :param class_num : 出力するクラス数
    :param img_size : 画像のピクセル比　整数なら指定したサイズの正方形、タプルなら(raw, cal)
    :param channels:色の出力変数（白黒画像なら1）
    :param kernel_size: 2次元の畳み込みウィンドウの幅と高さ 整数なら縦横比同じに
    :param model_name: インポートするモデルの名前。models_base以下のディレクトリにモデル生成器を置く
    :return: discriminator部のモデル
    """
    model_module = importlib.import_module("network_model.model_base."+model_name)
    return model_module.builder(class_num, img_size, channels, kernel_size)


def init_input_image(size: types_of_loco.input_img_size):
    def builder_of_generator(class_num: int, channels: int =1, kernel_size: Union[int, Tuple[int, int]] = 3):
        """
        Ganのgenerator部を作成する
        :param channels:色の出力変数（白黒画像なら1）
        :param kernel_size: 2次元の畳み込みウィンドウの幅と高さ 整数なら縦横比同じに
        :return: discriminator部のモデル
        """
        return builder(class_num, size, channels, kernel_size)
    return builder_of_generator


def build_wrapper(img_size: types_of_loco.input_img_size = 28,
                  channels: int = 3,
                  kernel_size: Union[int, Tuple[int, int]] = 3,
                  model_name: str = "model1") -> Callable[[int], keras.engine.training.Model]:
    """
    モデル生成をする関数を返す
    交差検証をかける際のラッパーとして使う
    :param img_size:
    :param channels:
    :param kernel_size:
    :param model_name:
    :return:
    """
    return lambda class_num: builder(class_num, img_size, channels, kernel_size, model_name)

