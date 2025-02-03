from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
from scipy.ndimage import zoom
from tqdm import tqdm
import lpips
import os


def smooth_l1_loss(true_diff, pred_diff, beta=1.0):
    diff = torch.abs(true_diff - pred_diff)
    return torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)


class CustomLossWithNet(nn.Module):
    def __init__(self, alpha=0.4, beta=0.5, gamma=0.1, chn_mid=32):
        super(CustomLossWithNet, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Rete G che prende due distanze e produce uno score
        self.net = nn.Sequential(
            nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, d0, d1, e0, e1, eps=0.1):
        # Usa la rete G per predire la probabilità che d1 sia migliore di d0
        score = self.net(
            torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1)
        )

        # 1. Ordering loss usando il punteggio predetto dalla rete
        h = (e0 > e1).float()
        ordering_loss = -h * torch.log(score) - (1 - h) * torch.log(1 - score)

        # 2. Normalized magnitude loss (come prima)
        true_diffs = e0 - e1
        pred_diffs = d0 - d1
        true_diffs_norm = (true_diffs - true_diffs.mean()) / (true_diffs.std() + 1e-8)
        pred_diffs_norm = (pred_diffs - pred_diffs.mean()) / (pred_diffs.std() + 1e-8)
        magnitude_loss = smooth_l1_loss(true_diffs_norm, pred_diffs_norm, beta=1.0)

        # 3. Regularization
        reg_loss = torch.mean(d0**2 + d1**2)

        return (
            self.alpha * ordering_loss
            + self.beta * magnitude_loss
            + self.gamma * reg_loss
        )


class RankingLoss(nn.Module):
    """
    A modified ranking loss that considers both order and magnitude of differences
    between perceptual distances and detection error scores.
    """

    def __init__(self, order_weight=0.4, magnitude_weight=0.6):
        super(RankingLoss, self).__init__()
        self.order_weight = order_weight
        self.magnitude_weight = magnitude_weight
        self.net = ModifiedDist2LogitLayer()

    def forward(self, scaled_distances, e0, e1):
        """
        Args:
            scaled_distances: Output from Dist2LogitLayer containing scaled perceptual distances
            e0, e1: Detection error scores (ground truth)

        Returns:
            Loss tensor without reduction, to be averaged during backward pass
        """
        # Extract d0 and d1 from scaled_distances
        d0 = scaled_distances[:, 0:1, :, :]  # First channel
        d1 = scaled_distances[:, 1:2, :, :]  # Second channel

        # Compute differences
        pred_diff = d0 - d1
        true_diff = e0 - e1

        # Sigmoid-based smooth approximation of sign agreement
        # pred and true with same sign -> sigmoid(-value) -> order_loss close to zero
        # pred and true with different sign -> sigmoid(value) -> order_loss close to one

        order_loss = torch.sigmoid(-pred_diff * true_diff * 100.0)

        # Compute magnitude loss using relative differences
        pred_abs_diff = torch.abs(pred_diff)
        true_abs_diff = torch.abs(true_diff)

        # Handle edge cases where true difference is very small
        eps = 1e-8
        scale_ratio = torch.min(
            pred_abs_diff / (true_abs_diff + eps), true_abs_diff / (pred_abs_diff + eps)
        )
        magnitude_loss = 1.0 - scale_ratio

        # Combine losses with weights without reduction
        return self.order_weight * order_loss + self.magnitude_weight * magnitude_loss


class ModifiedDist2LogitLayer(nn.Module):
    """
    Modified version of the Dist2LogitLayer that outputs scaled distances.
    The network takes raw LPIPS distances and learns to scale them to match
    the range and distribution of the detection error scores.
    """

    def __init__(self, chn_mid=32):
        super(ModifiedDist2LogitLayer, self).__init__()

        # Network to process both distances together
        self.model = nn.Sequential(
            # Initial processing
            nn.Conv2d(6, chn_mid, 1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            # Hidden layer
            nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            # Output layer - 2 channels for d0 and d1
            nn.Conv2d(chn_mid, 2, 1, stride=1, padding=0, bias=True),
        )

    def forward(self, d0, d1, eps=1e-8):
        """
        Takes two distance tensors and returns their scaled versions
        """
        # Concatenate distances along channel dimension
        distances = torch.cat(
            (d0, d1, d0 - d1, abs(d0 - d1), d0 / (d1 + eps), d1 / (d0 + eps)), dim=1
        )
        # Process and output scaled distances
        return self.model(distances)


class Trainer:
    def name(self):
        return self.model_name

    def initialize(
        self,
        model="lpips",
        net="alex",
        colorspace="Lab",
        pnet_rand=False,
        pnet_tune=False,
        model_path=None,
        use_gpu=True,
        printNet=False,
        spatial=False,
        is_train=False,
        lr=0.0001,
        beta1=0.5,
        version="0.1",
        gpu_ids=[0],
        a=0.4,
        b=0.5,
        c=0.1,
    ):
        """
        INPUTS
            model - ['lpips'] for linearly calibrated network
                    ['baseline'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        """
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.model_name = "%s [%s]" % (model, net)
        self.original_net = net

        if self.model == "lpips":  # pretrained net + linear layer
            self.net = lpips.LPIPS(
                pretrained=not is_train,
                net=net,
                version=version,
                lpips=True,
                spatial=spatial,
                pnet_rand=pnet_rand,
                pnet_tune=pnet_tune,
                use_dropout=True,
                model_path=model_path,
                eval_mode=False,
            )
        elif self.model == "baseline":  # pretrained network
            self.net = lpips.LPIPS(pnet_rand=pnet_rand, net=net, lpips=False)
        elif self.model in ["L2", "l2"]:
            self.net = lpips.L2(
                use_gpu=use_gpu, colorspace=colorspace
            )  # not really a network, only for testing
            self.model_name = "L2"
        elif self.model in ["DSSIM", "dssim", "SSIM", "ssim"]:
            self.net = lpips.DSSIM(use_gpu=use_gpu, colorspace=colorspace)
            self.model_name = "SSIM"
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        if self.is_train:  # training mode
            # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
            self.rankLoss = RankingLoss()
            self.parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(
                self.parameters, lr=lr, betas=(beta1, 0.999)
            )
        else:  # test mode
            if self.original_net != "yolov11m":
                self.net.eval()

        if use_gpu:
            self.net.to(gpu_ids[0])
            self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            if self.is_train:
                self.rankLoss = self.rankLoss.to(
                    device=gpu_ids[0]
                )  # just put this on GPU0

        if printNet:
            print("---------- Networks initialized -------------")
            # networks.print_network(self.net)
            print("-----------------------------------------------")

    def forward(self, in0, in1, retPerLayer=False):
        """Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        """

        return self.net.forward(
            in0, in1, retPerLayer=retPerLayer, net=self.original_net
        )

    # ***** TRAINING FUNCTIONS *****
    def optimize_parameters(self):
        self.forward_train()
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if hasattr(module, "weight") and module.kernel_size == (1, 1):
                module.weight.data = torch.clamp(module.weight.data, min=0)

    def set_input(self, data):
        self.input_ref = data["ref"]
        self.input_p0 = data["p0"]
        self.input_p1 = data["p1"]
        # self.input_judge = data["judge"]
        self.input_e0 = data["e0"]
        self.input_e1 = data["e1"]

        if self.use_gpu:
            self.input_ref = self.input_ref.to(device=self.gpu_ids[0])
            self.input_p0 = self.input_p0.to(device=self.gpu_ids[0])
            self.input_p1 = self.input_p1.to(device=self.gpu_ids[0])
            # self.input_judge = self.input_judge.to(device=self.gpu_ids[0])
            self.input_e0 = self.input_e0.to(device=self.gpu_ids[0])
            self.input_e1 = self.input_e1.to(device=self.gpu_ids[0])

        self.var_ref = Variable(self.input_ref, requires_grad=True)
        self.var_p0 = Variable(self.input_p0, requires_grad=True)
        self.var_p1 = Variable(self.input_p1, requires_grad=True)
        self.var_e0 = Variable(self.input_e0)
        self.var_e1 = Variable(self.input_e1)

        # # Save image for debug
        # import torchvision

        # for i, path in enumerate(data["p1_path"]):
        #     number = path.split("/")[-1].split(".")[0]
        #     torchvision.utils.save_image(self.var_p1[i], f"debug/p1/{number}.png")
        #     torchvision.utils.save_image(self.var_p0[i], f"debug/p0/{number}.png")

    def forward_train(self):  # run forward pass
        """
        Forward pass during training

        Args:
            data: Dictionary containing:
                - ref: Reference image
                - p0: First distorted image
                - p1: Second distorted image
                - error_score0: Detection error score for p0
                - error_score1: Detection error score for p1
        """
        # Get raw perceptual distances from LPIPS
        raw_d0 = self.net.forward(self.var_ref, self.var_p0, self.original_net)
        raw_d1 = self.net.forward(self.var_ref, self.var_p1, self.original_net)

        # Scale distances using Dist2LogitLayer
        scaled_distances = self.rankLoss.net.forward(raw_d0, raw_d1)

        # Compute loss using scaled distances
        self.loss_total = self.rankLoss(scaled_distances, self.var_e0, self.var_e1)

        # Extract scaled distances for monitoring
        d0 = scaled_distances[:, 0:1, :, :]
        d1 = scaled_distances[:, 1:2, :, :]
        self.acc_r = self.compute_accuracy(d0, d1, self.var_e0 < self.var_e1)

        return {
            "raw_d0": raw_d0.detach(),
            "raw_d1": raw_d1.detach(),
            "scaled_d0": d0.detach(),
            "scaled_d1": d1.detach(),
            "e0": self.var_e0.detach(),
            "e1": self.var_e1.detach(),
        }

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self, d0, d1, judge):
        """d0, d1 are Variables, judge is a Tensor"""
        d1_lt_d0 = (d1 < d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0 * judge_per + (1 - d1_lt_d0) * (1 - judge_per)

    def get_current_errors(self):
        retDict = OrderedDict(
            [("loss_total", self.loss_total.data.cpu().numpy()), ("acc_r", self.acc_r)]
        )

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def get_current_visuals(self):
        zoom_factor = 256 / self.var_ref.data.size()[2]

        ref_img = lpips.tensor2im(self.var_ref.data)
        p0_img = lpips.tensor2im(self.var_p0.data)
        p1_img = lpips.tensor2im(self.var_p1.data)

        ref_img_vis = zoom(ref_img, [zoom_factor, zoom_factor, 1], order=0)
        p0_img_vis = zoom(p0_img, [zoom_factor, zoom_factor, 1], order=0)
        p1_img_vis = zoom(p1_img, [zoom_factor, zoom_factor, 1], order=0)

        return OrderedDict(
            [("ref", ref_img_vis), ("p0", p0_img_vis), ("p1", p1_img_vis)]
        )

    def save(self, path, label):
        if self.use_gpu:
            self.save_network(self.net.module, path, "", label)
        else:
            self.save_network(self.net, path, "", label)
        self.save_network(self.rankLoss.net, path, "rank", label)

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        save_filename = "%s_net_%s.pth" % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = "%s_net_%s.pth" % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print("Loading network from %s" % save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self, nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group["lr"] = lr

        print("update lr [%s] decay: %f -> %f" % (type, self.old_lr, lr))
        self.old_lr = lr

    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, "done_flag"), flag)
        np.savetxt(
            os.path.join(self.save_dir, "done_flag"),
            [
                flag,
            ],
            fmt="%i",
        )


def score_2afc_dataset(data_loader, func, name=""):
    """Function computes Two Alternative Forced Choice (2AFC) score using
        distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a TwoAFCDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - 2AFC score in [0,1], fraction of time func agrees with human evaluators
        [1] - dictionary with following elements
            results - Dictionary with image names as keys, containing d0,d1,h values
            mean_score - The average score across all images
    """

    d0s = []
    d1s = []
    gts = []
    paths = []

    for data in tqdm(data_loader.load_data(), desc=name):
        d0s += func(data["ref"], data["p0"]).data.cpu().numpy().flatten().tolist()
        d1s += func(data["ref"], data["p1"]).data.cpu().numpy().flatten().tolist()
        gts += data["judge"].cpu().numpy().flatten().tolist()

        # Extract image name from path
        paths += [os.path.basename(p) for p in data["p0_path"]]

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    scores = (d0s < d1s) * (1.0 - gts) + (d1s < d0s) * gts + (d1s == d0s) * 0.5
    paths = np.array(paths)

    # Calculate final score
    mean_score = float(np.mean(scores))

    # Create per-image results dictionary with mean score
    results = {
        "mean_score": mean_score,  # Add mean score at the top level
        "results": {},  # Nested dictionary for individual results
    }

    for i, path in enumerate(paths):
        results["results"][path] = {
            "d0": float(d0s[i]),
            "d1": float(d1s[i]),
            "h (gt)": float(gts[i]),
            "score": float(scores[i]),
        }

    return (mean_score, results)


def score_jnd_dataset(data_loader, func, name=""):
    """Function computes JND score using distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a JNDDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return pytorch array of length N
    OUTPUTS
        [0] - JND score in [0,1], mAP score (area under precision-recall curve)
        [1] - dictionary with following elements
            ds - N array containing distances between two patches shown to human evaluator
            sames - N array containing fraction of people who thought the two patches were identical
    CONSTS
        N - number of test triplets in data_loader
    """

    ds = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        ds += func(data["p0"], data["p1"]).data.cpu().numpy().tolist()
        gts += data["same"].cpu().numpy().flatten().tolist()

    sames = np.array(gts)
    ds = np.array(ds)

    sorted_inds = np.argsort(ds)
    ds_sorted = ds[sorted_inds]
    sames_sorted = sames[sorted_inds]

    TPs = np.cumsum(sames_sorted)
    FPs = np.cumsum(1 - sames_sorted)
    FNs = np.sum(sames_sorted) - TPs

    precs = TPs / (TPs + FPs)
    recs = TPs / (TPs + FNs)
    score = lpips.voc_ap(recs, precs)

    return (score, dict(ds=ds, sames=sames))
