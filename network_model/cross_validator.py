from network_model import model as md
from sklearn.model_selection import StratifiedKFold
import keras.engine.training
from typing import Callable
from typing import List
import numpy as np


def build(model_builder:Callable[[], keras.engine.training.Model],
          class_set:List[str],
          result_dir_path:str,
          fold_num:int,
          epoch_num:int):
    """
    交差検証を行うための関数を生成する
    :param model_builder: モデル生成器
    :param class_set: クラスの元となったリスト
    :param result_dir_path: 結果を記録する際のパス
    :param fold_num: 交差検証を行う回数
    :param epoch_num: エポック数
    :return:
    """
    skf = StratifiedKFold(n_splits=fold_num)

    def test(data_set:np.array, label_set:np.ndarray, result_name:str = "result", model_name:str = "model"):
        """
        実際にバリデーションを行う
        :param data_set: データセット
        :param label_set: データのラベル
        :param result_name: 結果の出力先名
        :param model_name: モデルの名前
        :return:
        """
        label_index_set = [np.argmax(label) for label in label_set]
        for fold_itr, (train_index, test_index) in enumerate(skf.split(data_set, label_index_set)):
            train_data = np.array([data_set[index] for index in train_index])
            train_label = np.array([label_set[index] for index in train_index])
            test_data = np.array([data_set[index] for index in test_index])
            test_label = np.array([label_set[index] for index in test_index])
            model_name_iter = model_name + str(fold_itr)
            model = md.Model(model_builder(len(class_set)), class_set)
            model.test(train_data,
                       train_label,
                       test_data,
                       test_label,
                       epoch_num,
                       result_name,
                       result_dir_path,
                       model_name_iter)
        model = md.Model(model_builder(len(class_set)), class_set)
        model.fit(data_set, label_set, epoch_num).record(result_name, result_dir_path, model_name)
    return test


