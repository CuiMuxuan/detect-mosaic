import cv2
import image_util
import mosaic
import numpy as np
import detect
import string_util
import os
import re
from tqdm import tqdm


def batch_mosaic_processing():
    image_files = image_util.read_images_in_folder(folder_path)
    # 读取模型文件
    haar = cv2.CascadeClassifier(haar_path)
    net = cv2.dnn.readNetFromCaffe(net_txt, net_path)
    exist = 0
    not_exist = 0
    for image_file in tqdm(image_files):
        if string_util.contains_chinese(image_file):
            continue
        file_name = string_util.get_last_part(image_file)
        image = cv2.imread(image_file)
        if method == "haar":
            faces = detect.do_detection_haar(image, haar)
        elif method == "net":
            faces, image = detect.do_detection_net(image, net, True)
        # 打马赛克
        img = cv2.imread(image_file)
        mosaic_img = mosaic.do_mosaic(img, faces, mosaic_neighbor)
        if faces.__len__() == 0:
            cv2.imwrite(save_path + '\\mosaic\\not_exist\\' + file_name, mosaic_img)
            not_exist += 1
        else:
            cv2.imwrite(save_path + '\\mosaic\\exist\\' + file_name, mosaic_img)
            exist += 1
    print("共检出存在人脸的照片" + str(exist) + "张")
    print("未检出存在人脸的照片" + str(not_exist) + "张")


def batch_detect_processing():
    image_files = image_util.read_images_in_folder(folder_path)
    # 读取模型文件
    haar = cv2.CascadeClassifier(haar_path)
    net = cv2.dnn.readNetFromCaffe(net_txt, net_path)
    exist = 0
    not_exist = 0
    for image_file in tqdm(image_files):
        if string_util.contains_chinese(image_file):
            continue
        file_name = string_util.get_last_part(image_file)
        image = cv2.imread(image_file)
        if method == "haar":
            faces = detect.do_detection_haar(image, haar)
        elif method == "net":
            faces, image = detect.do_detection_net(image, net, True)
        if faces.__len__() == 0:
            cv2.imwrite(save_path + '\\detect\\not_exist\\' + file_name, image)
            not_exist += 1
        else:
            cv2.imwrite(save_path + '\\detect\\exist\\' + file_name, image)
            exist += 1
    print("共检出存在人脸的照片" + str(exist) + "张")
    print("未检出存在人脸的照片" + str(not_exist) + "张")


def video_mosaic():
    # 读取模型文件
    haar = cv2.CascadeClassifier(haar_path)
    net = cv2.dnn.readNetFromCaffe(net_txt, net_path)
    file_name = string_util.get_last_part(video_path)
    cap = cv2.VideoCapture(video_path)
    fps=cap.get(cv2.CAP_PROP_FPS)
    fourcc=int(cap.get(cv2.CAP_PROP_FOURCC))
    frame_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out=cv2.VideoWriter(save_path + '\\mosaic\\exist\\' + file_name,fourcc,fps,frame_size)
    while True:
        flag, frame = cap.read()
        org = frame
        if not flag:
            break
        if method == "haar":
            faces = detect.do_detection_haar(frame, haar)
        elif method == "net":
            faces, image = detect.do_detection_net(frame, net, False)
        frame = mosaic.do_mosaic(image, faces, mosaic_neighbor)
        out.write(frame)
        cv2.imshow('result', frame)
        if ord('q') == cv2.waitKey(10):
            break
    cv2.destroyAllWindows()
    cap.release()
    out.release()


if __name__ == '__main__':
    with open('config', 'r', encoding='utf-8') as file:
        content = file.read()
        content = re.sub(r'#.*', '', content)
        exec(content)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + '\\mosaic')
        os.makedirs(save_path + '\\detect')
        os.makedirs(save_path + '\\mosaic\\not_exist')
        os.makedirs(save_path + '\\mosaic\\exist')
        os.makedirs(save_path + '\\detect\\not_exist')
        os.makedirs(save_path + '\\detect\\exist')
    if batch_mosaic:
        batch_mosaic_processing()
    if batch_detect:
        batch_detect_processing()
    if batch_video:
        video_mosaic()
    print("程序已终止")
