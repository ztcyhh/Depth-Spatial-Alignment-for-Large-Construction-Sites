import cv2
import os

import numpy as np


def blur_detect(img):
    blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
    return blur_score

def is_blur_image(img, thres=1000):
    """
    :param img: 图片路径
    :param thres:
    :return: boolen
    """
    # img = cv2.imread(img)
    blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
    print(blur_score)
    if blur_score < thres:  # 清晰度计算太小，将会被判定为模糊图像
        return True
    else:
        return False


def extract_frames_from_one_video(video_path, output_dir, n):
    """
    :param video_path: 视频路径
    :param output_dir: 输出存储图片的位置
    :param n: 隔多少帧来截图
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0*n
    while success:
        if count % n == 0:
            cv2.imwrite(f"{output_dir}/frame{int(count/n):04}.jpg", image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1

def extract_frames_from_videos(video_folder_path, output_dir, n):
    """
    :param video_folder_path: 视频的文件夹的路径
    :param output_dir: 输出存储图片的位置
    :param n: 隔多少帧来截图
    """

    for video_index, video_name in enumerate(os.listdir(video_folder_path)):
        video_path = os.path.join(video_folder_path, video_name)
        vidcap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
        success, image = vidcap.read()

        print("读取视频可行度", cv2.VideoCapture(video_path).read())
        print("打开视频可行度", cv2.VideoCapture(video_path).isOpened())
        count = 0
        while success:
            if count % n == 0:
                cv2.imwrite(f"{output_dir}/v{video_index+1:02}_f{int(count / n):04}.jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1


def extract_frames_from_videos_with_blur_detection(video_folder_path, output_dir, n):
    """
    :param video_folder_path: 视频的文件夹的路径
    :param output_dir: 输出存储图片的位置
    :param n: 隔多少帧来截图
    """

    for video_index, video_name in enumerate(os.listdir(video_folder_path)):
        if video_name.endswith(".mp4") or video_name.endswith(".avi"):
            video_path = os.path.join(video_folder_path, video_name)
            vidcap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
            success, image = vidcap.read()
            count = 0
            sub_count = 0  # 可以优化，例如如果n帧里没有符合阈值的图像呢，那这n帧里就是选最清晰的那个？
            while success:
                IS_BLUR = is_blur_image(image)
                if not IS_BLUR:  # 如果是清晰的图像
                    idx_in_n = count % n  # n帧中的第几帧
                    cv2.imwrite(f"{output_dir}/v{video_index+1:02}_f{int(count // n):04}.jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])  # save frame as JPEG file
                    for i in range(n - idx_in_n):
                        success, image = vidcap.read()
                    count = count + n - idx_in_n
                else:
                    success, image = vidcap.read()
                    count = count + 1

def extract_frames_from_videos_most_clear(video_folder_path, output_dir, n):
    """
    :param video_folder_path: 视频的文件夹的路径
    :param output_dir: 输出存储图片的位置
    :param n: 隔多少帧来截图
    """
    for video_index, video_name in enumerate(os.listdir(video_folder_path)):
        video_path = os.path.join(video_folder_path, video_name)
        vidcap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
        success, image = vidcap.read()
        count = 0
        image_list = list()
        blur_score_list = list()
        sub_count = 0  # n帧中的第几帧

        while success:
            if sub_count < n:
                blur_score = blur_detect(image)
                image_list.append(image)
                blur_score_list.append(blur_score)
                sub_count += 1
                success, image = vidcap.read()
            else:
                most_clear_sub_index = np.argmax(np.asarray(blur_score_list))
                most_clear_image = image_list[most_clear_sub_index]
                cv2.imwrite(f"{output_dir}/v{video_index + 1:02}_f{count:04}.jpg", most_clear_image,
                            [cv2.IMWRITE_JPEG_QUALITY, 100])
                image_list = list()
                blur_score_list = list()
                sub_count = 0
                count += 1

video_path = 'D:\learning\data\3d_reconstruction\scale_model\test_bianjiao\vivo_x1\video\1.mp4'
video_folder_path = "/D/data/zt/project/colmap/hook/test2/video"
output_dir = '/D/data/zt/project/colmap/hook/test2/images'

# extract_frames_from_one_video(video_path, output_dir, 60)
extract_frames_from_videos(video_folder_path, output_dir, 60)
# extract_frames_from_videos_with_blur_detection(video_folder_path, output_dir, 10)
# extract_frames_from_videos_most_clear(video_folder_path, output_dir, 20)