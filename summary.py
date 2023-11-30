from glob import glob 
import numpy as np 
import os 
import json
import csv

# root = "output/original/TanksAndTemple"
# root = "output/ablation/TanksAndTemple_residual"
root = "output/ablation/GlossySynthetic_residual"
# root = "output/original/GlossySynthetic"

f_names = sorted(glob(os.path.join(root, "*")))
summary = {}
for f_name in f_names:

    results_path = os.path.join(f_name, "results.json")
    if not os.path.exists(results_path):
        continue
    with open(results_path) as json_file:
        contents = json.load(json_file)
        # method = sorted(contents.keys(), key=lambda method : int(method.split('_')[-1]))[-1]
        method = sorted(contents.keys(), key=lambda method : float(contents[method]['PSNR']))[-1]
        try:
            summary["Method"].append(os.path.basename(f_name))
            summary["PSNR"].append(contents[method]['PSNR'])
            summary["SSIM"].append(contents[method]['SSIM'])
            summary["LPIPS"].append(contents[method]['LPIPS'])
        except:
            summary["Method"] = [os.path.basename(f_name)]
            summary["PSNR"] = [contents[method]['PSNR']]
            summary["SSIM"] = [contents[method]['SSIM']]
            summary["LPIPS"] = [contents[method]['LPIPS']]
summary["Method"].append('Avg.')
summary["PSNR"].append(np.mean(summary["PSNR"]))
summary["SSIM"].append(np.mean(summary["SSIM"]))
summary["LPIPS"].append(np.mean(summary["LPIPS"]))

with open(os.path.join(root,'summary.csv'), 'w')as file_obj:
    writer_obj = csv.writer(file_obj)
    writer_obj.writerow(["Method"]+summary["Method"])
    writer_obj.writerow(["PSNR"]+summary["PSNR"])
    writer_obj.writerow(["SSIM"]+summary["SSIM"])
    writer_obj.writerow(["LPIPS"]+summary["LPIPS"])