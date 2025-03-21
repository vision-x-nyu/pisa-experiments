
import subprocess
import argparse
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def launch_jobs(worker_args,num_workers,worker_fn):
    with tqdm(total=len(worker_args)) as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_fn, *args) for args in worker_args]
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)

def worker(i,args):
    video_id = f"{i:05d}"
    sample_save_dir = os.path.join(args.output_dir,video_id)
    if all(os.path.exists(os.path.join(sample_save_dir,f"rgba_{i:05}.jpg")) for i in range(32)):
        print(f"Already completed {video_id}.")
        return
    command = (
        "python sim_data/run.py "
        f"--video_id={video_id} "
        f"--output_dir={args.output_dir} "
        "--resolution=512x512 "
        "--num_threads=8 "
        "--save_mp4 "
        "--save_gif "
        f"--objects_split train "
        f"--backgrounds_split train "
        f"--seed {i+1} "
    )
    subprocess.run(command,shell=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n",
    type=int,
    default=1
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="sim_output"
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=1
)

args = parser.parse_args()
worker_args = [(i,args) for i in range(args.n)]
launch_jobs(worker_args,args.num_workers,worker)
    