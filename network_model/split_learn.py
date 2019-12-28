import os
import shutil
import random
from typing import Callable
from typing import Tuple
import keras
from keras.preprocessing.image import ImageDataGenerator
from network_model import model as md
from DataIO.data_loader import count_data_num_in_dir
from DataIO.data_loader import NormalizeType


def image_dir_train_test_sprit(original_dir, base_dir, train_size=0.8, has_built:bool = True) -> str:
    '''
    画像データをトレインデータとテストデータにシャッフルして分割
    下記のURLで公開されていたコードの改変です
    https://qiita.com/komiya-m/items/c37c9bc308d5294d3260

    parameter
    ------------
    original_dir: str オリジナルデータフォルダのパス その下に各クラスのフォルダがある
    base_dir: str 分けたデータを格納するフォルダのパス　そこにフォルダが作られます
    train_size: float トレインデータの割合
    '''
    try:
        os.mkdir(base_dir)
    except FileExistsError:
        print(base_dir + "は作成済み")

    #クラス分のフォルダ名の取得
    dir_lists = os.listdir(original_dir)
    dir_lists = [f for f in dir_lists if os.path.isdir(os.path.join(original_dir, f))]
    original_dir_path = [os.path.join(original_dir, p) for p in dir_lists]

    if has_built:
        return dir_lists,\
               count_data_num_in_dir(os.path.join(base_dir, 'train')),\
               count_data_num_in_dir(os.path.join(base_dir, 'validation')),\
               count_data_num_in_dir(original_dir)

    num_class = len(dir_lists)

    # フォルダの作成(トレインとバリデーション)
    train_dir = os.path.join(base_dir, 'train')
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.mkdir(train_dir)

    validation_dir = os.path.join(base_dir, 'validation')
    if os.path.exists(validation_dir):
        shutil.rmtree(validation_dir)
    os.mkdir(validation_dir)

    #クラスフォルダの作成
    train_dir_path_lists = []
    val_dir_path_lists = []
    for directory_name in dir_lists:
        train_class_dir_path = os.path.join(train_dir, directory_name)
        os.mkdir(train_class_dir_path)
        train_dir_path_lists += [train_class_dir_path]

        val_class_dir_path = os.path.join(validation_dir, directory_name)
        os.mkdir(val_class_dir_path)
        val_dir_path_lists += [val_class_dir_path]


    #元データをシャッフルしたものを上で作ったフォルダにコピーします。
    #ファイル名を取得してシャッフル
    for i, path in enumerate(original_dir_path):
        files_class = os.listdir(path)
        random.shuffle(files_class)
        # 分割地点のインデックスを取得
        divide_num = int(len(files_class) * train_size)
        #トレインへファイルをコピー
        for file_name in files_class[:divide_num]:
            src = os.path.join(path, file_name)
            dst = os.path.join(train_dir_path_lists[i], file_name)
            shutil.copyfile(src, dst)
        #valへファイルをコピー
        for file_name in files_class[divide_num:]:
            src = os.path.join(path, file_name)
            dst = os.path.join(val_dir_path_lists[i], file_name)
            shutil.copyfile(src, dst)
        print(path + "コピー完了")

    print("分割終了")
    return dir_lists, \
           count_data_num_in_dir(os.path.join(base_dir, 'train')), \
           count_data_num_in_dir(os.path.join(base_dir, 'validation')), \
           count_data_num_in_dir(original_dir)


def train(model_builder: Callable[[int], keras.engine.training.Model],
          train_image_generator: ImageDataGenerator,
          test_image_generator: ImageDataGenerator,
          dataset_root_dir: str,
          result_dir_path: str ,
          normalize_type: NormalizeType = NormalizeType.Div255,
          temp_dir_path: str = "temp",
          batch_size = 32,
          epoch_num: int = 20,
          train_size: float = 0.8,
          image_size: Tuple[int, int] = (224, 224),
          result_name: str = "result",
          model_name: str = "model",
          has_built: bool = True,
          will_remove: bool = False,
          learn_only_original: bool = False,
          tmp_path:str = None
          ):
    """

    :param model_builder:
    :param image_generator:
    :param dataset_root_dir:
    :param result_dir_path:
    :param fold_num:
    :param epoch_num:
    :return:
    """
    try:
        class_list, train_data_num, test_data_num, original_data_num = image_dir_train_test_sprit(dataset_root_dir,
                                                                                                  temp_dir_path,
                                                                                                  train_size,
                                                                                                  has_built)
        model_val = md.ModelForManyData(model_builder(len(class_list)), class_list) if tmp_path is None else md.ModelForManyData(model_builder(tmp_path), class_list)
        train_dir = os.path.join(temp_dir_path, "train")
        validation_dir = os.path.join(temp_dir_path, "validation")
        train_generator = train_image_generator.flow_from_directory(train_dir,
                                                                    target_size=image_size,
                                                                    batch_size=batch_size,
                                                                    classes=class_list,
                                                                    class_mode="categorical")
        test_generator = test_image_generator.flow_from_directory(validation_dir,
                                                                  target_size=image_size,
                                                                  batch_size=batch_size,
                                                                  classes=class_list,
                                                                  class_mode="categorical")
        if learn_only_original == False:
        # テスト開始
          model_val.test(train_generator,
                         epoch_num,
                         test_generator,
                         normalize_type=normalize_type,
                         result_dir_name=result_name,
                         steps_per_epoch=train_data_num/batch_size,
                         validation_steps=test_data_num/batch_size,
                         dir_path=result_dir_path,
                         model_name=model_name+"val",
                         will_del_from_ram=True
                        )
        train_generator = train_image_generator.flow_from_directory(dataset_root_dir,
                                                                    target_size=image_size,
                                                                    batch_size=batch_size,
                                                                    classes=class_list,
                                                                    class_mode="categorical")
        model = md.ModelForManyData(model_builder(len(class_list)), class_list) if tmp_path is None else md.ModelForManyData(model_builder(tmp_path), class_list)
        model.fit_generator(train_generator, epoch_num,steps_per_epoch=original_data_num/batch_size) \
            .record(result_name,
                 result_dir_path,
                 model_name,
                 normalize_type=normalize_type
                 )
    finally:
        if os.path.exists(temp_dir_path) and will_remove:
            shutil.rmtree(temp_dir_path)

