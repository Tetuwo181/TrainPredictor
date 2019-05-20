import keras.engine.training
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from typing import List
from typing import Tuple
from typing import Optional
import os
from datetime import datetime
import json
import gc
import matplotlib.pyplot as plt
from DataIO import data_loader as dl


class Model(object):
    def __init__(self,
                 model_base: keras.engine.training.Model,
                 class_set: List[str],
                 tf_log_path:str = os.path.join(os.getcwd(), "tflog")):
        """

        :param model_base: kerasで構築したモデル
        :param class_set: クラスの元となったリスト
        :param tf_log_path: tensorboardのログを残すためのパス
        """
        self.__model = model_base
        self.__class_set = class_set
        self.__input_shape = model_base.input.shape.as_list()
        self.__history = None

    def fit(self,
            data: np.ndarray,
            label_set:  np.ndarray,
            epochs: int,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
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

    def fit_generator(self,
                      image_generator: ImageDataGenerator,
                      data: np.ndarray,
                      label_set:  np.ndarray,
                      epochs: int,
                      generator_batch_size: int = 32,
                      validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        モデルの適合度を算出する
        generatorを使ってデータを水増しして学習する場合に使用する
        :param image_generator: keras形式でのデータを水増しするジェネレータ
        :param data: 学習に使うデータ
        :param label_set: 教師ラベル
        :param epochs: エポック数
        :param generator_batch_size: ジェネレータのバッチサイズ
        :param validation_data: テストに使用するデータ　実データとラベルのセットのタプル
        :return:
        """
        print("fit generator")
        image_generator.fit(data)
        print("start learning")
        if validation_data is None:
            self.__history = self.__model.fit_generator(image_generator.flow(data,
                                                                             label_set,
                                                                             batch_size=generator_batch_size),
                                                        steps_per_epoch=len(data) / generator_batch_size,
                                                        epochs=epochs)
        else:
            self.__history = self.__model.fit_generator(image_generator.flow(data,
                                                                             label_set,
                                                                             batch_size=generator_batch_size),
                                                        steps_per_epoch=len(data)/generator_batch_size,
                                                        epochs=epochs,
                                                        validation_data=validation_data)
        return self

    def predict(self, data: np.ndarray) -> Tuple[np.array, np.array]:
        """
        モデルの適合度から該当するクラスを算出する
        :param data: 算出対象となるデータ
        :return: 判定したインデックスと形式名
        """
        result_set = np.array([np.argmax(result) for result in self.__model.predict(data)])
        class_name_set = np.array([self.__class_set[index] for index in result_set])
        return result_set, class_name_set

    def predict_top_n(self, data: np.ndarray, top_num: int = 5) -> List[Tuple[np.array, np.array, np.array]]:
        """
        適合度が高い順に車両形式を取得する
        :param data: 算出対象となるデータ
        :param top_num: 取得する上位の数値
        :return: 判定したインデックスと形式名と確率のタプルのリスト
        """
        predicted_set = self.__model.predict(data)
        return [self.get_predicted_upper(predicted_result, top_num) for predicted_result in predicted_set]

    def calc_succeed_rate(self,
                          data_set: np.ndarray,
                          label_set: np.ndarray,) -> float:
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
             normalize_type: dl.NormalizeType = dl.NormalizeType.Div255,
             image_generator: ImageDataGenerator = None,
             generator_batch_size: int = 32,
             result_dir_name: str = None,
             dir_path: str = None,
             model_name: str = None,
             will_del_from_ram: bool = False):
        """
        指定したデータセットに対しての正答率を算出する
        :param train_data_set: 学習に使用したデータ
        :param train_label_set: 学習に使用した正解のラベル
        :param test_data_set: テストデータ
        :param test_label_set: テストのラベル
        :param epochs: エポック数
        :param normalize_type: どのように正規化するか
        :param image_generator: keras形式でのデータを水増しするジェネレータ これを引数で渡さない場合はデータの水増しをしない
        :param generator_batch_size: ジェネレータのバッチサイズ
        :param result_dir_name: 記録するためのファイル名のベース
        :param dir_path: 記録するディレクトリ デフォルトではカレントディレクトリ直下にresultディレクトリを作成する
        :param model_name: モデル名　デフォルトではmodel
        :param will_del_from_ram: 記録後モデルを削除するかどうか
        :return:学習用データの正答率とテスト用データの正答率のタプル
        """
        if image_generator is None:
            self.fit(train_data_set, train_label_set, epochs, (test_data_set, test_label_set))
        else:
            self.fit_generator(image_generator,
                               train_data_set,
                               train_label_set,
                               epochs,
                               generator_batch_size,
                               (test_data_set, test_label_set))
        train_rate = self.calc_succeed_rate(train_data_set, train_label_set)
        test_rate = self.calc_succeed_rate(test_data_set, test_label_set)
        # 教師データと予測されたデータの差が0でなければ誤判定
        try:
            return train_rate, test_rate
        finally:
            if (result_dir_name is not None) and (dir_path is not None):
                self.record(result_dir_name,
                            dir_path,
                            model_name,
                            normalize_type,
                            train_rate,
                            test_rate,
                            will_del_from_ram)

    def record(self,
               result_dir_name: str,
               dir_path: str = os.path.join(os.getcwd(), "result"),
               model_name: str = "model",
               normalize_type: Optional[dl.NormalizeType] = None,
               train_succeed_rate: Optional[float] = None,
               test_succeed_rate: Optional[float] = None,
               will_del_from_ram: bool = False) -> None:
        """
        モデルや設定などを記録する
        :param result_dir_name: 記録するためのファイル名のベース
        :param dir_path: 記録するディレクトリ デフォルトではカレントディレクトリ直下にresultディレクトリを作成する
        :param model_name: モデル名　デフォルトではmodel
        :param train_succeed_rate: モデルをテストした際の教師データでの正答率　指定しなければ書き出さない
        :param test_succeed_rate: モデルをテストした際のテストデータでの正答率　指定しなければ書き出さない
        :param will_del_from_ram: 記録後モデルを削除するかどうか
        :return:
        """
        print("start record")
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        result_path = os.path.join(dir_path, result_dir_name+datetime.now().strftime("%Y%m%d%H%M%S"))
        os.mkdir(result_path)
        self.recorf_graph(result_path, model_name)
        self.__model.save(os.path.join(result_path, model_name + ".h5"))
        if will_del_from_ram:
            self.__model = None
            del self.__model
            gc.collect()
            print("model has deleted")
        write_set = {"class_set": self.__class_set}
        write_set["input_shape"] = self.__input_shape
        if normalize_type is not None:
            write_set["normalize_type"] = normalize_type
        if train_succeed_rate is not None:
            write_set["train_succeed_rate"] = train_succeed_rate
        if test_succeed_rate is not None:
            write_set["test_succeed_rate"] = test_succeed_rate
        write_dic = {model_name: write_set}
        json_path = os.path.join(result_path, "model_conf.json")
        with open(json_path, 'w',  encoding='utf8') as fw:
            json.dump(write_dic, fw, ensure_ascii=False)

    def get_predicted_upper(self, predicted_result: np.ndarray, top_num: int = 5) -> Tuple[np.array, np.array, np.array]:
        """
        予測した結果の数値からふさわしい形式を指定した上位n個だけ取り出す
        :param predicted_result: 予測結果
        :param top_num:
        :return:
        """
        top_index_set = np.argpartition(-predicted_result, top_num)[:top_num]
        top_value_set = predicted_result[top_index_set]
        top_series_set = np.array([self.__class_set[index] for index in top_index_set])
        return top_index_set, top_value_set, top_series_set

    def recorf_graph(self, result_dir: str, name: str):
        """
        正答率などのグラフを保存する
        :param result_dir: 保存先のディレクトリ
        :param name: 名前
        :return:
        """
        if self.__history is None:
            return self
        plt.plot(self.__history.history['loss'])
        plt.plot(self.__history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        loss_path = os.path.join(result_dir, name + "_loss.png")
        plt.savefig(loss_path)

        plt.figure()

        plt.plot(self.__history.history['acc'])
        plt.plot(self.__history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        acc_path = os.path.join(result_dir, name + "_acc.png")
        plt.savefig(acc_path)

        plt.figure()

        return self


