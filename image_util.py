import cv2
import os

def show_image(message,image):
    cv2.imshow(message,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_images_in_folder(folder_path):
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(root, file))
    return image_files
