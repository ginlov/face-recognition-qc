from retinaface import RetinaFace
from tqdm import tqdm
import logging
import cv2
import argparse
import os
import json

model = RetinaFace('high')

def infer_object(
        object: str,
        root_dir: str,
        output_dir: str
    ):
    os.makedirs(output_dir, exist_ok=True)

    object_root_path = os.path.join(root_dir, object, 'image_lr')
    object_output_path = os.path.join(output_dir, object)

    cam_list = os.listdir(object_root_path)
    for cam in tqdm(cam_list, desc="Run through cam list"):
        os.makedirs(os.path.join(object_output_path, cam), exist_ok=True)
        image_list = os.listdir(os.path.join(object_root_path, cam))
        for image in image_list:
            img = cv2.imread(os.path.join(object_root_path, cam, image))

            # Detect bounding boxes
            faces = model.predict(img)
            if len(faces) == 0:
                data = {}
            else:
                data = faces[0]
            with open(os.path.join(object_output_path, cam, image), 'wb') as f:
                json.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--range', type=int, nargs=2, metavar=('START', 'END'), default=[0, -1], help="Range of objects to process (start end)")

    args = parser.parse_args()
    logging.info(f"Run face recognition for images in {args.root_dir}, save results to {args.output_dir}")
    logging.info(f"Object range: {args.range[0]}-{args.range[1]}")
    all_objects = os.listdir(args.root_dir)
    all_objects = [item for item in all_objects if not item.endswith(('json', 'txt'))]
    assert len(all_objects) > 0, "Empty root directory"

    if args.range != [0, -1]:
        min_object = int(min(all_objects))
        max_object = int(max(all_objects))

        assert min_object <= args.range[0], "Min value out of range"
        assert max_object >= args.range[1], "Max value out of range"

    objects_to_run = [str(item) for item in range(*args.range) if str(item) in all_objects]

    for object in tqdm(objects_to_run, desc=f"Loop through object list"):
        logging.info(f"Face recognition for object {object}")
        infer_object(object, args.root_dir, args.output_dir)