import os
import json
from tqdm import tqdm

parent_fold = '/workspace/datasetvol/mvhuman_data/face_bboxes'

def read_file(file):
    try:
        with open(file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        raise e

def process_objects(args):
    """Process a list of objects and return their data."""
    obj_list, process_idx = args  # unpack
    local_data = {}

    for obj in tqdm(
        obj_list,
        desc=f"Proc {process_idx+1}",
        position=process_idx,  # 👈 each process gets its own line
        leave=True,
        ncols=100
    ):
        obj_path = os.path.join(parent_fold, obj)
        local_data[obj] = {}
        for cam in os.listdir(obj_path):
            cam_path = os.path.join(obj_path, cam)
            local_data[obj][cam] = {}
            for file in os.listdir(cam_path):
                file_path = os.path.join(cam_path, file)
                local_data[obj][cam][file] = read_file(file_path)

    return local_data

def get_pod_index():
    """Get the pod index from environment variables."""
    # Try different ways to get pod index
    # hostname = os.environ.get('HOSTNAME', '')
    return int(os.environ.get('JOB_COMPLETION_INDEX', 0))

if __name__ == "__main__":
    all_objects = os.listdir(parent_fold)
    all_objects.sort()  # ensure stable order

    # Split into 10 roughly equal chunks
    num_processes = int(os.environ.get("JOB_PARALLELISM", 1))
    chunk_size = len(all_objects) // num_processes
    chunks = [all_objects[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes - 1)]
    chunks.append(all_objects[(num_processes - 1) * chunk_size:])

    # Get pod index
    pod_idnex = get_pod_index()
    object_list = chunks[pod_idnex]

    result = process_objects((object_list, pod_idnex))

    # Save each chunk’s results separately
    output_dir = "/workspace/datasetvol/mvhuman_data/unified_face_boxes"
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"data_part_{pod_idnex}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f)
    print(f"✅ Saved {out_path}")
