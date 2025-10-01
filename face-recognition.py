from retinaface import RetinaFace
from tqdm import tqdm
import time
import cv2
import argparse
import os
import json

model = RetinaFace('high')

def infer_object(
        object: str,
        root_dir: str,
        output_dir: str,
    ):
    os.makedirs(output_dir, exist_ok=True)

    object_root_path = os.path.join(root_dir, object, 'images_lr')
    object_output_path = os.path.join(output_dir, object)

    cam_list = os.listdir(object_root_path)
    for cam in tqdm(cam_list, desc="Run through cam list"):
        os.makedirs(os.path.join(object_output_path, cam), exist_ok=True)
        image_list = os.listdir(os.path.join(object_root_path, cam))
        for image in image_list:
            # --- timing starts here ---
            t0 = time.perf_counter()
            img = cv2.imread(os.path.join(object_root_path, cam, image))
            t1 = time.perf_counter()

            faces = model.predict(img)
            t2 = time.perf_counter()

            if len(faces) == 0:
                data = {}
            else:
                data = faces[0]
                data = {key: int(v) if not isinstance(v, tuple) else [int(item) for item in v]
                        for key, v in data.items()}

            with open(os.path.join(object_output_path, cam, image), 'w') as f:
                json.dump(data, f)
            t3 = time.perf_counter()

            # --- log times ---
            read_time = t1 - t0
            detect_time = t2 - t1
            dump_time = t3 - t2
            total_time = t3 - t0

            print(f"[{cam}/{image}] read: {read_time:.4f}s | detect: {detect_time:.4f}s | dump: {dump_time:.4f}s | total: {total_time:.4f}s")

def get_pod_index():
    """Get the pod index from environment variables."""
    # Try different ways to get pod index
    # hostname = os.environ.get('HOSTNAME', '')
    return int(os.environ.get('JOB_COMPLETION_INDEX', 0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--range', type=int, nargs=2, metavar=('START', 'END'), default=[0, -1], help="Range of objects to process (start end)")

    args = parser.parse_args()
    print(f"Run face recognition for images in {args.root_dir}, save results to {args.output_dir}")
    print(f"Object range: {args.range[0]}-{args.range[1]}")
    all_objects = os.listdir(args.root_dir)
    all_objects = [item for item in all_objects if not item.endswith(('json', 'txt'))]
    assert len(all_objects) > 0, "Empty root directory"

    if args.range != [0, -1]:
        min_object = int(min(all_objects))
        max_object = int(max(all_objects))

        assert min_object <= args.range[0], "Min value out of range"
        assert max_object >= args.range[1], "Max value out of range"

    # Generate images for a subset of subjects based on pod index
    pod_index = get_pod_index()
    total_pods = int(os.environ.get('JOB_PARALLELISM', 1))
    print(f"Total pods {total_pods}")

    objects_to_run = sorted([str(item) for item in range(*args.range) if str(item) in all_objects])
    len_objects = len(objects_to_run)

    starting_index = len_objects // total_pods * pod_index
    ending_index = min(len_objects // total_pods * (pod_index + 1), len_objects)

    if ending_index == len_objects:
        object_list = objects_to_run[starting_index:]
    else:
        object_list = objects_to_run[starting_index:ending_index]

    for object in tqdm(object_list, desc=f"Loop through object list"):
        print(f"Face recognition for object {object}")
        infer_object(object, args.root_dir, args.output_dir)