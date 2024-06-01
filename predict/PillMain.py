import cv2
import numpy as np

from . import PillModel as PillModel
from . import ImageSide_circle as ImageSide_circle
from . import ImageSide_ellipse as ImageSide_ellipse
from . import ImageContourCount as ImageContourCount
from rembg import remove
from PIL import Image
import sys
import shutil
import configparser
import pandas as pd
import datetime


class PillMain():
    def __init__(self):
        pass

    def remove_background(self, image_path):
        input_image = Image.open(image_path)  # 이미지 로드
        output_image = remove(input_image)  # 배경 제거

        # PIL 이미지를 OpenCV 형식으로 변환
        open_cv_image = np.array(output_image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # 검정색 배경 생성
        background = np.zeros_like(open_cv_image)
        # 이미지 합성
        mask = open_cv_image[:, :, 3]  # 알파 채널 사용하여 마스크 생성
        for c in range(0, 3):
            background[:, :, c] = open_cv_image[:, :, c] * (mask / 255.0) + background[:, :, c] * (1.0 - mask / 255.0)

        return background

    def main(self, argv):
        if len(argv) != 4:
            print("Argument is wrong")
            print("Usage: python PillMain.py [IMAGE1 FULL PATH] [IMAGE2 FULL PATH] [TEXT FILE PATH]")
            sys.exit()

        image1_path = argv[1]
        image2_path = argv[2]
        text_file_path = argv[3]

        data_info = pd.read_csv(text_file_path, delimiter='\t')
        ori_shape = data_info['shape'][0]
        f_text = data_info['f_text'][0]
        b_text = data_info['b_text'][0]
        drug_list_ori = data_info['drug_code'][0].replace('[', '').replace(']', '').replace(' ', '').split(',')

        print(f_text, b_text, drug_list_ori)

        if drug_list_ori[0] == 'none':
            drug_list = drug_list_ori[0]
        else:
            drug_list = drug_list_ori

        nowdate = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        log_path = './data/pred_log/' + nowdate + '.log'
        f = open(log_path, 'a')

        shape_list = ['circle', 'ellipse', 'triangle', 'diamond', 'pentagon', 'hexagon', 'octagon', 'square', 'etc']
        if ori_shape not in shape_list:
            print("SHAPE : circle, ellipse, triangle, diamond, pentagon, hexagon, octagon, square, and etc")
            sys.exit()

        # 이미지 배경을 제거하고 검정색 배경에 넣기
        processed_image1 = self.remove_background(image1_path)
        processed_image2 = self.remove_background(image2_path)

        # 처리된 이미지를 임시 파일로 저장
        temp_image1_path = "./data/temp_processed_image1.jpeg"
        temp_image2_path = "./data/temp_processed_image2.jpeg"
        cv2.imwrite(temp_image1_path, processed_image1)
        cv2.imwrite(temp_image2_path, processed_image2)

        if ori_shape == 'circle':
            imageside = ImageSide_circle.ImageContour()
            proportion = 4.7
            _, image1_result, contourcnt1 = imageside.Process(temp_image1_path, proportion)
            _, image2_result, contourcnt2 = imageside.Process(temp_image2_path, proportion)
        elif ori_shape == 'ellipse':
            imageside = ImageSide_ellipse.ImageContour()
            proportion = 5.5
            _, image1_result, contourcnt1 = imageside.Process(temp_image1_path, proportion, proportion)
            _, image2_result, contourcnt2 = imageside.Process(temp_image2_path, proportion)

        else:
            imagecontourcount = ImageContourCount.ImageContour()
            proportion = 4.7
            contourcnt1 = imagecontourcount.Process(temp_image1_path, proportion)
            contourcnt2 = imagecontourcount.Process(temp_image2_path, proportion)

        # choice one input image
        choiceimage = PillModel.ChoiceImage()
        if ori_shape in ('circle', 'ellipse'):
            # if input text count is two, shape is 'BOTH'
            if f_text != 'none' and b_text != 'none':
                shape, image_path = choiceimage.ChoiceImage(ori_shape, image1_result, image2_result, contourcnt1,
                                                            contourcnt2, temp_image1_path, temp_image2_path, True)
            else:
                shape, image_path = choiceimage.ChoiceImage(ori_shape, image1_result, image2_result, contourcnt1,
                                                            contourcnt2, temp_image1_path, temp_image2_path)

        else:
            image_path = choiceimage.ChoiceImageContour(contourcnt1, contourcnt2, temp_image1_path, temp_image2_path)
            shape = ori_shape

        # f.write(shape + '\n')
        # f.write(image_path + '\n')
        print(image_path)
        # config file load for each shape
        config = configparser.ConfigParser()
        shape_path = './data/config_shape/'

        if shape == 'circle_BOTH':
            config_file = shape_path + 'config_circle_BOTH.ini'
        elif shape == 'circle_ONESIDE':
            config_file = shape_path + 'config_circle_ONESIDE.ini'
        elif shape == 'ellipse_BOTH':
            config_file = shape_path + 'config_ellipse_BOTH.ini'
        elif shape == 'ellipse_ONESIDE':
            config_file = shape_path + 'config_ellipse_ONESIDE.ini'
        elif shape == 'triangle':
            config_file = shape_path + 'config_triangle.ini'
        elif shape == 'diamond':
            config_file = shape_path + 'config_diamond.ini'
        elif shape == 'pentagon':
            config_file = shape_path + 'config_pentagon.ini'
        elif shape == 'hexagon':
            config_file = shape_path + 'config_hexagon.ini'
        elif shape == 'octagon':
            config_file = shape_path + 'config_octagon.ini'
        elif shape == 'square':
            config_file = shape_path + 'config_square.ini'
        elif shape == 'etc':
            config_file = shape_path + 'config_etc.ini'

        config.read(config_file, encoding='UTF-8')
        pillModel = PillModel.PillModel(config['pill_model_info'])

        # image processing
        pillModel.pill_image_process(image_path)

        # image open
        img = pillModel.testImage(config['pill_model_info']['make_folder_path'])

        # model loading
        pillModel.pill_shape_conf(shape)
        pillModel.pill_model_loading(config['pill_model_info'])

        # prediction
        output = pillModel.pill_prediction(img)
        indices_top, includ_count = pillModel.pill_sorting(output, drug_list)

        # if shape_oneside( or shape_both) model training drug code is not in drug list, try shape_both (or shape_oneside) model
        if (includ_count == 0) and (ori_shape in ('circle', 'ellipse')):
            if shape == ori_shape + '_ONESIDE':
                shape = ori_shape + '_BOTH'
                config_file = shape_path + 'config_' + shape + '.ini'

            else:
                shape = ori_shape + '_ONESIDE'
                config_file = shape_path + 'config_' + shape + '.ini'

            # f.write(shape + '\n')

            config.read(config_file, encoding='UTF-8')
            pillModel = PillModel.PillModel(config['pill_model_info'])
            pillModel.pill_shape_conf(shape)
            pillModel.pill_model_loading(config['pill_model_info'])

            output = pillModel.pill_prediction(img)
            indices_top, includ_count = pillModel.pill_sorting(output, drug_list)

        print(pillModel.pill_information(indices_top))
        f.write(pillModel.pill_information(indices_top))
        f.close()

        # remove filter image folder
        shutil.rmtree(config['pill_model_info']['make_folder_path'])


if __name__ == '__main__':
    main_class = PillMain()
    main_class.main(sys.argv)
