import torch
import cv2
import numpy as np

def load_sample(img_file, lbl_file, options={}):
  joint_accuracy = (options.get("joint_acc", 3) - 1) // 2

  img = cv2.imread(img_file)[:,:,::-1].astype(np.float32) / 255.0 # H, W, 3
  lbl = np.zeros((img.shape[0], img.shape[1], 21), dtype=np.float32) # 20 joints + palm position

  lbl_info = []
  for j, line in enumerate(open(lbl_file, "r").readlines()):
    coords = [int(float(val)) for val in line.split()[1:]]

    if "new_size" in options:
      coords = [int(coords[i] * scale_factor[1-i]) for i in range(2)]

    lbl_info.append(coords)

  for coords in lbl_info:
    lbl[coords[1] - joint_accuracy : coords[1] + joint_accuracy + 1, 
        coords[0] - joint_accuracy : coords[0] + joint_accuracy + 1, j] = 1

  return img, lbl

def load_batch(img_files, lbl_files, joint_accuracy=3):
  np_img = np.zeros((len(img_files), 120, 160, 3))
  np_lbl = np.zeros((len(lbl_files), 120, 160, 21))

  for j in range(len(img_files)):
    np_img[j], np_lbl[j] = load_sample(img_files[j], lbl_files[j])

  np_img = np.transpose(np_img, (0, 3, 1, 2))
  np_lbl = np.transpose(np_lbl, (0, 3, 1, 2))

  return torch.Tensor(np_img), torch.Tensor(np_lbl)
