from src.inerf.lieutils import SO3
from src.pixelnerf.src.util.util import stratified_masked_sample
import numpy as np
import torch
from dotmap import DotMap

from src.transform import Transform
from src.inerf.lieutils import SE3Exp
from src.device_dict import DeviceDict
from src.pixelnerf.src.model import loss
from src.pixelnerf.src.util import gen_rays, batched_index_select_nd, stratified_bbox_sample, SO3_spherical


def fmt_loss_str(losses):
    return "loss " + (" ".join(k + ":" + str(losses[k]) for k in losses))


class iNeRF:
    def __init__(self, net, renderer, args, conf, z_near=0.2, z_far=4.0,
                 device='cpu', num_iters=100, n_candidates=2):
        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)

        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        self.lr_lambda = lambda epoch: 0.5 ** (float(epoch) / 100.0)

        self.net = net
        self.renderer = renderer
        self.lr = args.lr

        self.z_near = z_near
        self.z_far = z_far
        self.device = device

        self.num_iters = num_iters
        self.ray_batch_size = args.ray_batch_size
        self.n_candidates = n_candidates

    # def _stage1_single_cand(self, inputs, targets, init_angles, init_log_r):
    #     N_objects = inputs["images"].shape[0]
    #     _, H, W = targets["images"][0].shape

    #     angle_params = torch.clone(init_angles).unsqueeze(
    #         0).to(device=self.device)
    #     angle_params.requires_grad_(True)
    #     log_r = torch.clone(init_log_r).unsqueeze(0).to(device=self.device)
    #     log_r.requires_grad_(True)

    #     optim = torch.optim.Adam([angle_params, log_r], lr=self.lr)
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(
    #         optim, lr_lambda=self.lr_lambda)

    #     inp_pose = SO3_spherical(torch.zeros(
    #         (1, 3)), torch.zeros(1)).to(device=self.device)

    #     gt_0to1 = targets["images"] * 0.5 + 0.5  # (BS, 3, H, W)
    #     rgb_gt_all = (
    #         gt_0to1.permute(0, 2, 3, 1).contiguous().reshape(N_objects, -1, 3)
    #     )  # (NO, H*W, 3)

    #     for t in range(self.num_iters):
    #         curr_pose = SO3_spherical(angle_params, log_r)  # (NC, 4, 4)
    #         if t % 20 == 0:
    #             print(curr_pose)

    #         if "bbox" in targets:
    #             pix = stratified_bbox_sample(
    #                 targets["bbox"], self.ray_batch_size)
    #             pix_inds = pix[..., 0] * W + pix[..., 1]
    #         else:
    #             pix_inds = torch.randint(
    #                 0, H * W, self.ray_batch_size).expand(N_objects, -1).to(device=self.device)

    #         rgb_gt = batched_index_select_nd(rgb_gt_all, pix_inds)
    #         # (NO, ray_batch_size, 3)

    #         cam_rays = gen_rays(
    #             curr_pose.expand(N_objects, -1, -1),
    #             W, H, targets["focal"], self.z_near, self.z_far
    #         ).reshape(N_objects, -1, 8)
    #         # (NO, W*H, 8)
    #         cam_rays = batched_index_select_nd(cam_rays, pix_inds)
    #         # (NO, ray_batch_size, 8)

    #         self.net.encode(
    #             inputs["images"],
    #             inp_pose.expand(N_objects, -1, -1),
    #             inputs["focal"],
    #         )

    #         render_dict = DotMap(self.renderer(
    #             cam_rays, want_weights=True,))
    #         coarse = render_dict.coarse
    #         fine = render_dict.fine

    #         coarse_loss = self.rgb_coarse_crit(coarse.rgb, rgb_gt)
    #         fine_loss = self.rgb_fine_crit(fine.rgb, rgb_gt)
    #         loss = self.lambda_coarse * coarse_loss + self.lambda_fine * fine_loss
    #         loss.backward()

    #         loss_dict = dict(
    #             rc=coarse_loss.item(),
    #             rf=fine_loss.item(),
    #             t=loss.item(),
    #         )
    #         if t % 10 == 0:
    #             loss_str = fmt_loss_str(loss_dict)
    #             print("T", t, loss_str, " lr", optim.param_groups[0]["lr"])

    #         optim.step()
    #         optim.zero_grad()
    #         scheduler.step()

    #     pose = SO3_spherical(angle_params.detach(), log_r.detach())
    #     return loss_dict, pose

    # def _stage1(self, inputs, targets):
    #     losses = []
    #     poses = []
    #     for _ in range(self.n_candidates):
    #         angle_params = 2 * torch.randn(3)
    #         log_r = 0.2 * torch.randn(1)
    #         loss_dict, pose = self._stage1_single_cand(
    #             inputs, targets, angle_params, log_r)
    #         losses.append(loss_dict["t"])
    #         poses.append(pose)

    #     inp_pose = SO3_spherical(torch.zeros(
    #         (1, 3)), torch.zeros(1)).to(device=self.device)

    #     best_idx = np.argmin(np.array(losses))
    #     return inp_pose[0], poses[best_idx]

    def __call__(self, inputs, targets):
        """
        Find camera poses of target images relative to input images.
        :param inputs dict of
        images (B, 3, H, W)
        focal (B, 2)
        c (B, 2) (principal point)
        :return (B, 4, 4) pose of target image as a homogeneous matrix
        """
        self.net.eval()
        self.net.requires_grad(False)

        inputs = DeviceDict(**inputs).to(device=self.device)
        targets = DeviceDict(**targets).to(device=self.device)

        BS = inputs["images"].shape[0]
        num_sample_rays = self.ray_batch_size // BS
        input_T = Transform(np.eye(3), np.array([0, 0, 0]))
        input_poses = torch.tensor(
            input_T.homogeneous,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        delta_pose = 0.1 * torch.randn(1, 6, device=self.device)
        delta_pose.requires_grad_(True)
        optim = torch.optim.Adam([delta_pose], lr=self.lr)

        _, _, H, W = targets["images"].shape

        target_imgs = targets["images"].to(device=self.device)
        gt_0to1 = target_imgs * 0.5 + 0.5  # (BS, 3, H, W)
        rgb_gt_all = (
            gt_0to1.permute(0, 2, 3, 1).contiguous().reshape(BS, -1, 3)
        )  # (BS, H*W, 3)

        input_imgs = inputs["images"].to(device=self.device)
        input_focals = inputs["focal"].to(device=self.device)
        input_c = inputs["c"].to(device=self.device)
        input_bbox = inputs["bbox"].to(device=self.device)

        target_focals = targets["focal"].to(device=self.device)
        target_bbox = targets["bbox"].to(device=self.device)
        target_c = targets["c"].to(device=self.device)

        for t in range(self.num_iters):

            curr_poses = SE3Exp(delta_pose) @ input_poses

            cam_rays = gen_rays(
                curr_poses, W, H, target_focals, self.z_near, self.z_far, c=target_c)
            cam_rays = cam_rays.reshape(1, -1, 8).expand(BS, -1, -1)

            # pix = stratified_masked_sample(
            # targets["masks"], num_sample_rays
            # ).to(device=self.device)
            # pix_inds = torch.randint(
            # 0, H * W, (BS, num_sample_rays)).to(device=self.device)
            pix1 = stratified_bbox_sample(target_bbox, num_sample_rays // 2)
            pix1_inds = pix1[..., 0] * W + pix1[..., 1]

            pix2 = stratified_bbox_sample(input_bbox, num_sample_rays // 2)
            pix2_inds = pix2[..., 0] * W + pix2[..., 1]

            pix_inds = torch.cat((pix1_inds, pix2_inds), dim=-1)

            cam_rays = batched_index_select_nd(cam_rays, pix_inds)
            rgb_gt = batched_index_select_nd(rgb_gt_all, pix_inds)

            self.net.encode(
                input_imgs.unsqueeze(1),
                input_poses.unsqueeze(1),
                input_focals.unsqueeze(1),
                c=input_c.unsqueeze(1),
            )

            render_dict = DotMap(self.renderer(cam_rays, want_weights=True,))
            coarse = render_dict.coarse
            fine = render_dict.fine
            using_fine = len(fine) > 0

            loss_dict = {}

            coarse_loss = self.rgb_coarse_crit(coarse.rgb, rgb_gt)
            loss_dict["rc"] = coarse_loss.item() * self.lambda_coarse
            if using_fine:
                fine_loss = self.rgb_fine_crit(fine.rgb, rgb_gt)
                loss = coarse_loss * self.lambda_coarse + fine_loss * self.lambda_fine
                loss_dict["rf"] = fine_loss.item() * self.lambda_fine
            else:
                loss = coarse_loss

            loss.backward()
            loss_dict["t"] = loss.item()

            loss_str = fmt_loss_str(loss_dict)
            print("T", t, loss_str, " lr", optim.param_groups[0]["lr"])

            optim.step()
            optim.zero_grad()

        self.net.requires_grad(True)

        curr_poses = SE3Exp(delta_pose) @ input_poses

        return input_poses, curr_poses
