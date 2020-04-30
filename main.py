import torch
import glob
import random

import train
import handdataset
import keynet

BATCH_SIZE = 32

if __name__ == "__main__":
    img_files = glob.glob("./dataset/images/*.jpg")
    lbl_files = glob.glob("./dataset/labels/*.txt")

    data = list(zip(img_files, lbl_files))
    random.shuffle(data)

    num_train = int(len(data) * 0.91)
    train_data = torch.utils.data.DataLoader(data[:num_train], batch_size=BATCH_SIZE)
    test_data = torch.utils.data.DataLoader(data[num_train:], batch_size=BATCH_SIZE)

    print("Num Training Data:", num_train, "Num Batches:", len(train_data))
    print("Num Validation Data:", len(data) - num_train, "Num Batches:", len(test_data))

    device = torch.device("cpu:0")
    model = keynet.Model(3, 21).to(device)

    train.train(model, device, train_data, test_data, options={
        "CKPT_DIR": "./ckpts",
        "MAX_EPOCHS": 1,
        "BATCH_SIZE": BATCH_SIZE
    })





