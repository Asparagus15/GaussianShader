import os
import numpy as np
import math
import json
import glob
import argparse
import pickle
import shutil
from skimage.io import imread, imsave

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/remote-home/tujd/NeRF/gaussian-splatting-surfel/data/GlossySynthetic", help="path to the GlossyBlender dataset")
    parser.add_argument('--scene', type=str, default="cat", help="scene name")

    opt = parser.parse_args()

    root = os.path.join(opt.path, opt.scene)
    output_path = os.path.join(opt.path, opt.scene+"_blender")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    print(f'[INFO] Data from {root}')

    img_num = len(glob.glob(f'{root}/*.pkl'))
    img_ids = [str(k) for k in range(img_num)]
    cams = [read_pickle(f'{root}/{k}-camera.pkl') for k in range(img_num)]  # pose(3,4)  K(3,3)
    img_files = [f'{root}/{k}.png' for k in range(img_num)]
    depth_files = [f'{root}/{k}-depth.png' for k in range(img_num)]
    points_file = os.path.join(root, "eval_pts.ply")

    test_ids, train_ids = read_pickle(os.path.join(opt.path, 'synthetic_split_128.pkl'))

    # process 2 splits
    for split in ['train', 'test']:
        print(f'[INFO] Process transforms split = {split}')

        ids = test_ids if split == "test" else train_ids
        split_imgs = [img_files[int(i)] for i in ids]
        split_cams = [cams[int(i)] for i in ids]

        frames = []
        for image, cam in zip(split_imgs, split_cams):
            w2c = np.array(cam[0].tolist()+[[0,0,0,1]])
            c2w = np.linalg.inv(w2c)
            c2w[:3, 1:3] *= -1  # opencv -> blender/opengl
            frames.append({
                'file_path': os.path.join("rgb", os.path.basename(image)).replace(".png",""),
                'transform_matrix': c2w.tolist(),
            })

        fl_x = float(split_cams[0][1][0,0])
        fl_y = float(split_cams[0][1][1,1])

        transforms = {
            'w': 800,
            'h': 800,
            'fl_x': fl_x,
            'fl_y': fl_y,
            'cx': 400,
            'cy': 400,
            # 'aabb_scale': 2,
            'frames': frames,
        }

        # write json
        json_out_path = os.path.join(output_path, f'transforms_{split}.json')
        print(f'[INFO] write to {json_out_path}')
        with open(json_out_path, 'w') as f:
            json.dump(transforms, f, indent=2)
    
    # write imgs
    img_out_path = os.path.join(output_path, "rgb")
    if not os.path.exists(img_out_path):
        os.makedirs(img_out_path, exist_ok=True)
    print(f'[INFO] Process rgbs')
    print(f'[INFO] write to {img_out_path}')
    for img_id in img_ids:
        depth = imread(f'{root}/{img_id}-depth.png')
        depth = depth.astype(np.float32) / 65535 * 15
        mask = depth < 14.5
        mask = (mask[...,None] * 255).astype(np.uint8)

        image = imread(f'{root}/{img_id}.png')[..., :3]
        image = np.concatenate([image, mask], axis=-1)

        imsave(f'{img_out_path}/{img_id}.png', image)

    # copy ply
    points_out_path = os.path.join(output_path, "points.ply")
    shutil.copy2(points_file, points_out_path)

    print("[INFO] Finished.")




    
