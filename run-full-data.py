import os
import json
import re
import cv2
import time
import sys
import argparse
from tqdm import tqdm
from typing import Dict, Tuple, Union
from retinaface import RetinaFace

parent_fold = '/workspace/datasetvol/mvhuman_data/face_bboxes'
pattern = re.compile(r"^(\d{2}05)_img\.jpg$")
source_fold = '/workspace/datasetvol/mvhuman_data/mv_captures'
model = RetinaFace()

def infer_object(
        object: str,
        root_dir: str,
) -> Dict[str, Dict[str,Dict[str, Union[int, Tuple[int, ...]]]]]:
    return_data = {}
    object_root_path = os.path.join(root_dir, object, 'images_lr')

    cam_list = os.listdir(object_root_path)
    pattern = re.compile(r"^(\d{2}05)_img\.jpg$")
    for cam in tqdm(cam_list, desc=f"Processing cameras for object {object}"):
        return_data[cam] = {}
        image_list = os.listdir(os.path.join(object_root_path, cam))
        selected_images = [
            img for img in image_list
            if pattern.match(img) and int(pattern.match(img).group(1)) % 100 == 5 ]
        print(len(selected_images), "images selected for camera", cam)

        for image in selected_images:
            # --- timing starts here ---
            img = cv2.imread(os.path.join(object_root_path, cam, image))

            faces = model.predict(img)

            if len(faces) == 0:
                data = {}
            else:
                data = faces[0]
                data = {key: int(v) if not isinstance(v, tuple) else [int(item) for item in v]
                        for key, v in data.items()}
            return_data[cam][image] = data
    return return_data 

def read_file(file):
    with open(file, 'r') as f:
        return json.load(f)

def get_pod_index():
    """Get the pod index from environment variables."""
    # Try different ways to get pod index
    # hostname = os.environ.get('HOSTNAME', '')
    return int(os.environ.get('JOB_COMPLETION_INDEX', 0))

def face_recognition(root_dir, object_list, output_dir, pod_index):
    """Run face recognition on a list of objects and save results."""
    total_dict = {}
    for obj in object_list:
        print(f"Processing object {obj}")
        result = infer_object(obj, root_dir)
        total_dict[obj] = result

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'full_data_part_{pod_index}.json'), 'w') as f:
            json.dump(total_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

    print(f"Run face recogntion for images in {args.root_dir}, save results to {args.output_dir}")

    all_objects = os.listdir(args.root_dir)
    all_objects = [item for item in all_objects if not item.endswith(('json', 'txt'))]
    assert len(all_objects) > 0, "Empty root directory"

    # Remove object from 10001 to 100500
    all_objects.sort()  # ensure stable order
    all_objects = [obj for obj in all_objects if not (10001 <= int(obj) <= 100500)]

    assert len(all_objects) > 0, "No objects to process after filtering"
    len_objects = len(all_objects)

    # Get total pods and pod index
    total_pods = int(os.environ.get("JOB_PARALLELISM", 1))
    pod_index = get_pod_index()

    starting_index = len_objects // total_pods * pod_index
    ending_index = min(len_objects // total_pods * (pod_index + 1), len_objects)

    if ending_index == len_objects:
        object_list = all_objects[starting_index:]
    else:
        object_list = all_objects[starting_index:ending_index]
    
    face_recognition(args.root_dir, object_list, args.output_dir, pod_index)
