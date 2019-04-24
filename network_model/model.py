import keras.engine.training
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import numpy as np
from typing import List
from typing import Tuple
from typing import Optional
import os
from datetime import datetime
import json


class Model(object):
    def __init__(self, model_base: keras.engine.training.Model,
                 class_set: List[str],
                 tf_log_path:str = os.path.join(os.getcwd(), "tflog")):
        """

        :param model_base: kerasで構築したモデル
        :param class_set: クラスの元となったリスト
        :param tf_log_path: tensorboardのログを残すためのパス
        """
        self.__model = model_base
        self.__class_set = class_set

    def fit(self,
            data: np.ndarray,
            label_set:  np.ndarray,
            epochs: int,
            validation_data:Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        モデルの適合度を算出する
        :param data: 学習に使うデータ
        :param label_set: 教師ラベル
        :param epochs: エポック数
        :param validation_data: テストに使用するデータ　実データとラベルのセットのタプル
        :return:
        """
        if validation_data is None:
            self.__model.fit(data, label_set, epochs=epochs)
        else:
            self.__model.fit(data, label_set, epochs=epochs, validation_data=validation_data)
        return self

    def predict(self, data: np.ndarray)->Tuple[np.array, np.array]:
        """
        モデルの適合度から該当するクラスを算出する
        :param data: 算出対象となるデータ
        :return: 判定したインデックスと具体的な名前
        """
        result_set = np.array([np.argmax(result) for result in self.__model.predict(data)])
        class_name_set = [self.__class_set[index] for index in result_set]
        return result_set, class_name_set

    def calc_succeed_rate(self,
                          data_set: np.ndarray,
                          label_set: np.ndarray,)->float:
        """
        指定したデータセットに対しての正答率を算出する
        :param data_set: テストデータ
        :param label_set: 正解のラベル
        :return:
        """
        predicted_index, predicted_name = self.predict(data_set)
        teacher_label_set = np.array([np.argmax(teacher_label) for teacher_label in label_set])
        # 教師データと予測されたデータの差が0でなければ誤判定
        diff = teacher_label_set - predicted_index
        return np.sum(diff == 0)/len(data_set)

    def test(self,
             train_data_set: np.ndarray,
             train_label_set: np.ndarray,
             test_data_set: np.ndarray,
             test_label_set: np.ndarray,
             epochs: int,
             result_dir_name:str = None,
             dir_path:str = None ,
             model_name: str = None):
        """
        指定したデータセットに対しての正答率を算出する
        :param train_data_set: 学習に使用したデータ
        :param train_label_set: 学習に使用した正解のラベル
        :param test_data_set: テストデータ
        :param test_label_set: テストのラベル
        :param epochs: エポック数
        :param result_dir_name: 記録するためのファイル名のベース
        :param dir_path: 記録するディレクトリ デフォルトではカレントディレクトリ直下にresultディレクトリを作成する
        :param model_name: モデル名　デフォルトではmodel
        :return:学習用データの正答率とテスト用データの正答率のタプル
        """
        self.fit(train_data_set, train_label_set, epochs, (test_data_set, test_label_set))
        train_rate = self.calc_succeed_rate(train_data_set, train_label_set)
        test_rate = self.calc_succeed_rate(test_data_set, test_label_set)
        # 教師データと予測されたデータの差が0でなければ誤判定
        try:
            return train_rate, test_rate
        finally:
            if (result_dir_name is not None) and (dir_path is not None):
                self.record(result_dir_name, dir_path, model_name, train_rate, test_rate)

    def record(self,
               result_dir_name: str,
               dir_path: str = os.path.join(os.getcwd(), "result"),
               model_name: str = "model",
               train_succeed_rate:Optional[float] = None,
               test_succeed_rate:Optional[float] = None):
        """
        モデルや設定などを記録する
        :param result_dir_name: 記録するためのファイル名のベース
        :param dir_path: 記録するディレクトリ デフォルトではカレントディレクトリ直下にresultディレクトリを作成する
        :param model_name: モデル名　デフォルトではmodel
        :param train_succeed_rate: モデルをテストした際の教師データでの正答率　指定しなければ書き出さない
        :param test_succeed_rate: モデルをテストした際のテストデータでの正答率　指定しなければ書き出さない
        :return:
        """
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        result_path = os.path.join(dir_path, result_dir_name+datetime.now().strftime("%Y%m%d%H%M%S"))
        os.mkdir(result_path)
        self.__model.save(os.path.join(result_path, model_name + ".h5"))
        write_set = {"class_set": self.__class_set}
        if train_succeed_rate is not None:
            write_set["train_succeed_rate"] = train_succeed_rate
        if test_succeed_rate is not None:
            write_set["test_succeed_rate"] = test_succeed_rate
        write_dic = {model_name: write_set}
        json_path = os.path.join(result_path, "model_conf.json")
        with open(json_path, 'w') as fw:
            json.dump(write_dic, fw)

