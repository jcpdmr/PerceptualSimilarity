import os.path
import torchvision.transforms as transforms
from data.dataset.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
# from IPython import embed


class TwoAFCDataset(BaseDataset):
    def initialize(
        self,
        dataroots,
        model_net: str,
        load_size=64,
    ):
        print(
            f"TwoAFCDataset initialize -> img size: {load_size}, model: {model_net} (used for image transformation during loading)"
        )
        if not isinstance(dataroots, list):
            dataroots = [
                dataroots,
            ]
        self.roots = dataroots
        self.load_size = load_size

        # image directory
        self.dir_ref = [os.path.join(root, "ref") for root in self.roots]
        self.ref_paths = make_dataset(self.dir_ref)
        self.ref_paths = sorted(self.ref_paths)

        self.dir_p0 = [os.path.join(root, "p0") for root in self.roots]
        self.p0_paths = make_dataset(self.dir_p0)
        self.p0_paths = sorted(self.p0_paths)

        self.dir_p1 = [os.path.join(root, "p1") for root in self.roots]
        self.p1_paths = make_dataset(self.dir_p1)
        self.p1_paths = sorted(self.p1_paths)

        transform_list = []
        transform_list.append(transforms.Resize(load_size))
        transform_list += (
            [
                transforms.ToTensor(),  # ensure the image is in the range [0,1]
            ]
            if model_net == "yolov11m"
            else [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # ensure the image is in the range [-1,1]
            ]
        )

        self.transform = transforms.Compose(transform_list)

        # judgement directory
        self.dir_J = [os.path.join(root, "judge") for root in self.roots]
        self.judge_paths = make_dataset(self.dir_J, mode="np")
        self.judge_paths = sorted(self.judge_paths)

        # e0 directory
        self.dir_e0 = [os.path.join(root, "e0") for root in self.roots]
        self.e0_paths = make_dataset(self.dir_e0, mode="np")
        self.e0_paths = sorted(self.e0_paths)

        # e1 directory
        self.dir_e1 = [os.path.join(root, "e1") for root in self.roots]
        self.e1_paths = make_dataset(self.dir_e1, mode="np")
        self.e1_paths = sorted(self.e1_paths)

    def __getitem__(self, index):
        p0_path = self.p0_paths[index]
        p0_img_ = Image.open(p0_path).convert("RGB")
        p0_img = self.transform(p0_img_)

        p1_path = self.p1_paths[index]
        p1_img_ = Image.open(p1_path).convert("RGB")
        p1_img = self.transform(p1_img_)

        ref_path = self.ref_paths[index]
        ref_img_ = Image.open(ref_path).convert("RGB")
        ref_img = self.transform(ref_img_)
        # print(
        #     f"ref_img: {ref_img.shape}, p1_img: {p1_img.shape}, p0_img: {p0_img.shape}"
        # )
        # print(
        #     f"p0 Min: {p0_img.min()}, p0 Max: {p0_img.max()}    p1 Min: {p1_img.min()}, p1 Max: {p1_img.max()}"
        # )
        judge_path = self.judge_paths[index]
        # judge_img = (np.load(judge_path)*2.-1.).reshape((1,1,1,)) # [-1,1]
        judge_img = np.load(judge_path).reshape(
            (
                1,
                1,
                1,
            )
        )  # [0,1]
        judge_img = torch.FloatTensor(judge_img)

        e0_path = self.e0_paths[index]
        e0_img = np.load(e0_path).reshape(
            (
                1,
                1,
                1,
            )
        )  # [0,1]
        e0_img = torch.FloatTensor(e0_img)

        e1_path = self.e1_paths[index]
        e1_img = np.load(e1_path).reshape(
            (
                1,
                1,
                1,
            )
        )  # [0,1]
        e1_img = torch.FloatTensor(e1_img)

        return {
            "p0": p0_img,
            "p1": p1_img,
            "ref": ref_img,
            "judge": judge_img,
            "e0": e0_img,
            "e1": e1_img,
            "p0_path": p0_path,
            "p1_path": p1_path,
            "ref_path": ref_path,
            "judge_path": judge_path,
        }

    def __len__(self):
        return len(self.p0_paths)
