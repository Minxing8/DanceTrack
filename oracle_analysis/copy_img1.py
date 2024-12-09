import os
import shutil

test_dir = '/home/minxing/datasets/NSVA_157_person/test'
source_dir = '/home/minxing/datasets/random300_videos_from_30games_frames'

for video_folder in os.listdir(test_dir):
    test_video_path = os.path.join(test_dir, video_folder)
    
    if os.path.isdir(test_video_path) and 'gt' in os.listdir(test_video_path) and 'img1' not in os.listdir(test_video_path):

        source_video_path = os.path.join(source_dir, video_folder)
        img1_source_path = os.path.join(source_video_path, 'img1')

        if os.path.isdir(source_video_path) and os.path.isdir(img1_source_path):

            img1_target_path = os.path.join(test_video_path, 'img1')
            shutil.copytree(img1_source_path, img1_target_path)
            print(f"Successfully copied img1 from {video_folder} to destination")
        else:
            print(f"img1 not fouond in {video_folder}, skipped. ")
    else:
        print(f"{video_folder} already has img1 or not sstisfied, skipped. ")

print("Copy completed. ")
