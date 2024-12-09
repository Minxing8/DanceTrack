import os
import shutil

source_folder = "/home/minxing/datasets/NSVA_157_person/test"
target_folder = "/home/minxing/datasets/NSVA_157_person/test_downsampled_20fps"

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

for video_name in sorted(os.listdir(source_folder)):
    video_path = os.path.join(source_folder, video_name)

    if not os.path.isdir(video_path):
        continue

    target_video_path = os.path.join(target_folder, video_name)
    os.makedirs(target_video_path, exist_ok=True)

    target_gt_path = os.path.join(target_video_path, "gt")
    os.makedirs(target_gt_path, exist_ok=True)

    source_gt_file = os.path.join(video_path, 'gt', 'gt.txt')
    target_gt_file = os.path.join(target_gt_path, 'gt.txt')

    with open(source_gt_file, 'r') as f_in, open(target_gt_file, 'w') as f_out:
        for line in f_in:
            linelist = line.split(',')
            frame_id = int(linelist[0])
            if frame_id % 3 == 0: 
                f_out.write(line)

    source_img1_path = os.path.join(video_path, 'img1')
    target_img1_path = os.path.join(target_video_path, 'img1')
    os.makedirs(target_img1_path, exist_ok=True)

    img_files = sorted(os.listdir(source_img1_path))
    for img_file in img_files:
        frame_id = int(os.path.splitext(img_file)[0])  
        if frame_id % 3 == 0:  
            source_img_file = os.path.join(source_img1_path, img_file)
            target_img_file = os.path.join(target_img1_path, img_file)
            shutil.copyfile(source_img_file, target_img_file)

    source_seqinfo_file = os.path.join(video_path, 'seqinfo.ini')
    target_seqinfo_file = os.path.join(target_video_path, 'seqinfo.ini')

    with open(source_seqinfo_file, 'r') as f_in, open(target_seqinfo_file, 'w') as f_out:
        for line in f_in:
            if line.startswith('frameRate'):
                f_out.write('frameRate=20\n')  
            else:
                f_out.write(line)

print("Downsample to 20fps process completed.")
