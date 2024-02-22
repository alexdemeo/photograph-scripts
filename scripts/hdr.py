import os
import shutil
from datetime import datetime
import cv2
import numpy as np
from PIL import ImageChops, Image

SRC_DIR = os.path.expanduser('~/Desktop/sample-japan-jpg/')
DEST_DIR = os.path.expanduser('~/Desktop/japan-hdr-jpg/')


def list_src_files():
    file_list = [SRC_DIR + file for file in os.listdir(SRC_DIR) if os.path.isfile(os.path.join(SRC_DIR, file)) and not file.endswith('.DS_Store')]
    file_list_with_dates = [(file, datetime.fromtimestamp(os.stat(file).st_birthtime)) for file in file_list]
    sorted_files = sorted(file_list_with_dates, key=lambda x: x[1])
    return [x[0] for x in sorted_files]


def compare_image_similarity(img1, img2):
    diff = ImageChops.difference(img1, img2)
    return np.mean(np.array(diff))


def bucket_images_by_date():
    src_files = list_src_files()
    prev_file_date = None
    curr_dir_name = 0
    for i in range(len(src_files)):
        src_file = src_files[i]
        src_file_name = src_file.split('/')[-1]
        src_file_date = datetime.fromtimestamp(os.stat(src_file).st_birthtime)

        if prev_file_date is None or (src_file_date - prev_file_date).seconds > 2:
            dest_file_dir = DEST_DIR + str(curr_dir_name) + '/'
            if not os.path.exists(dest_file_dir):
                os.mkdir(dest_file_dir)
                curr_dir_name += 1
                
        dest_file_path = dest_file_dir + src_file_name
        if not os.path.exists(dest_file_path):
            shutil.copy(src_file, dest_file_path)
        else:
            src_img = Image.open(src_file)
            dest_img = Image.open(dest_file_path)
            diff = compare_image_similarity(src_img, dest_img)
            print(f'diff({src_file}, {dest_file_path}) = {diff}')
            if diff > 0.01:
                suffix = src_file_name.split('.')[-1]
                dest_file_path = dest_file_dir + src_file_name.split('.')[0] + '_' + str(i) + '.' + suffix
                shutil.copy(src_file, dest_file_path)
        prev_file_date = src_file_date


def cleanup_ungrouped_images():
    result_dirs = [dir for dir in os.listdir(DEST_DIR) if os.path.isdir(os.path.join(DEST_DIR, dir))]
    for dir in result_dirs:
        files = [file for file in os.listdir(DEST_DIR + dir) if os.path.isfile(os.path.join(DEST_DIR + dir, file))]
        if len(files) == 1:
            src_file = DEST_DIR + dir + '/' + files[0]
            dest_file = DEST_DIR + files[0]
            print(f'moving {src_file} to {dest_file}')
            shutil.move(src_file, dest_file)
            os.rmdir(DEST_DIR + dir)


def stabilize_images(image_sequence: list[np.ndarray]):
    # Align images
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(image_sequence, image_sequence)
    # Assuming image_sequence is a list of numpy arrays (images)
    stabilized_sequence = [image_sequence[0]]  # Initialize with the first frame
    for i in range(1, len(image_sequence)):
        # Calculate optical flow between successive frames
        prev_gray = cv2.cvtColor(image_sequence[i - 1], cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(image_sequence[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute the translation from the flow
        translation = np.mean(flow, axis=(0, 1))
        # Apply the translation to the current frame
        transformation_matrix = np.array([[1, 0, -translation[0]], [0, 1, -translation[1]]], dtype=np.float32)
        height, width = image_sequence[i].shape[:2]
        stabilized_image = cv2.warpAffine(image_sequence[i], transformation_matrix, (width, height))
        stabilized_sequence.append(stabilized_image)
    return stabilized_sequence


def get_exposure_time_for_image(image_path: str) -> float:
    with Image.open(image_path) as img:
        exif_data = img._getexif()
        # Ensure EXIF data is present
        if not exif_data:
            raise ValueError("no EXIF data found")
        # Tag 0x829A corresponds to ExposureTime
        exposure_time = exif_data.get(0x829A)
        print(f'exposure time for {image_path} = {exposure_time}')
        if exposure_time:
            return exposure_time
        else:
            raise ValueError("no exposure time found in EXIF data")


def hdrify_images(img_files: list[str]) -> dict[str, np.ndarray]:
    img_list = [cv2.imread(file) for file in img_files]
    img_list = stabilize_images(img_list)

    exposure_times = np.array([get_exposure_time_for_image(file) for file in img_files], dtype=np.float32)
    
    merge_debevec = cv2.createMergeDebevec()
    hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
    merge_robertson = cv2.createMergeRobertson()
    hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())
    merge_mertens = cv2.createMergeMertens()
    hdr_mertens = merge_mertens.process(img_list)

    tonemap = cv2.createTonemap(gamma=2.2)
    res_debevec = tonemap.process(hdr_debevec.copy())
    res_robertson = tonemap.process(hdr_robertson.copy())
    res_mertens = hdr_mertens.copy()

    res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
    res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
    res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')

    return {"robertson": res_robertson_8bit, "debevec": res_debevec_8bit, "mertens": res_mertens_8bit}


def hdrify_remaining_folders():
    result_dirs = [dir for dir in os.listdir(DEST_DIR) if os.path.isdir(os.path.join(DEST_DIR, dir))]
    for dir in result_dirs:
        files = [DEST_DIR + dir + '/' + file for file in os.listdir(DEST_DIR + dir) if os.path.isfile(os.path.join(DEST_DIR + dir, file)) and not file.endswith('.DS_Store')]
        print(f'hdrify({dir}) from {len(files)} files')
        suffix = files[0].split('.')[-1]
        for hdr_method, hdr_img in hdrify_images(files).items():
            dest_file = DEST_DIR + dir + '/' + hdr_method + '.' + suffix
            cv2.imwrite(dest_file, hdr_img)


def main():
    bucket_images_by_date()
    cleanup_ungrouped_images()
    hdrify_remaining_folders()


if __name__ == '__main__':
    print('STARTING')
    main()
    print('DONE')
