import cv2
import numpy as np

def show_detection(image, faces):
    """在每个检测到的人脸上绘制一个矩形进行标示"""
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), [0,0,255],2)
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

def do_detection_haar(img,haar):
    """
    参数1：image–待检测图片，一般为灰度图像加快检测速度；
    参数2：objects–被检测物体的矩形框向量组；
    参数3：scaleFactor–表示在前后两次相继的扫描中，搜索窗口的比例系数。（默认为1.1）即每次搜索窗口依次扩大10%，该值越大计算的越快，人脸检测也越差;
    参数4：minNeighbors–表示构成检测目标的相邻矩形的最小个数(默认为3个)。
    """
    faces = haar.detectMultiScale(img)
    return faces

def do_detection_net(image,net,frame):
    faces=[]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104., 117., 123.], False, False)
    # 将blob设置为输入以获得结果，对整个网络执行前向计算以计算输出
    net.setInput(blob)
    detections = net.forward()
    # 迭代检测并绘制结果，仅在相应置信度大于最小阈值时才将其可视化
    detected_faces = 0
    w, h = image.shape[1], image.shape[0]
    # 迭代所有检测结果
    for i in range(0, detections.shape[2]):
        # 获取当前检测结果的置信度
        confidence = detections[0, 0, i, 2]
        # 如果置信大于最小置信度，则将其可视化
        if confidence > 0.7:
            detected_faces += 1
            # 获取当前检测结果的坐标
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            #记录检测到的脸到faces
            faces.append([startX, startY, endX-startX, endY-startY])
            # 绘制检测结果和置信度
            if frame:
                text = "{:.3f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
                cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return faces,image