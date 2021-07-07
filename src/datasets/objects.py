import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.nn import functional as F

from src.transform import Transform
from src.pixelnerf.src.util import get_image_to_tensor_balanced

X_rot = Transform(np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
]), np.zeros(3))


def extract_object(rgb, mask, K, crop_margin=0.1, out_size=64):
    H, W, _ = rgb.shape
    channels = torch.split(rgb, 1, dim=-1)
    for c in channels:
        c[torch.logical_not(mask)] = 255
    masked_img = torch.stack([c.squeeze(-1) for c in channels], dim=-1)

    mask_y, mask_x = np.nonzero(mask.numpy())
    bbox_x1 = np.min(mask_x)
    bbox_x2 = np.max(mask_x)
    bbox_y1 = np.min(mask_y)
    bbox_y2 = np.max(mask_y)

    w = bbox_x2 - bbox_x1
    h = bbox_y2 - bbox_y1
    dim = int(max(w, h) * (0.5 + crop_margin))
    x_center = int((bbox_x1 + bbox_x2) * 0.5)
    y_center = int((bbox_y1 + bbox_y2) * 0.5)
    x1 = x_center - dim
    x2 = x_center + dim
    y1 = y_center - dim
    y2 = y_center + dim

    pad_size = max(0, -x1, -y1, x2-W, y2-H)

    padded_img = F.pad(masked_img, (0, 0, pad_size, pad_size,
                       pad_size, pad_size), value=255)
    cropped_img = padded_img[y1+pad_size:y2 +
                             pad_size, x1+pad_size:x2+pad_size, :]

    s = out_size / (2 * dim)
    fx = K[0, 0]
    fy = K[1, 1]
    focal = s * torch.tensor([fx, fy])
    cx = K[0, 2] - x1
    cy = K[1, 2] - y1
    c = s * torch.tensor([cx, cy])

    bbox = (s*torch.tensor([
        bbox_x1 - x1,
        bbox_y1 - y1,
        bbox_x2 - x1,
        bbox_y2 - y1,
    ])).int()

    img = Image.fromarray(cropped_img.numpy())
    img_resized = img.resize((out_size, out_size))
    return img_resized, bbox, focal, c


class ObjectsDataset:
    def __init__(self, scene_ds, chunks=5):
        frame_index = scene_ds.frame_index.copy()
        frame_index["frame_id"] = np.arange(len(frame_index))
        frame_index = frame_index.explode(
            "visib_objects", ignore_index=True).dropna()
        groups = frame_index.groupby(["scene_id", "visib_objects"]).groups
        obj_index = []

        random_state = np.random.RandomState(0)

        for (scene_id, object_id), group_ids in groups.items():
            group_ids = random_state.permutation(group_ids)
            for idx in range(0, len(group_ids), chunks):
                if len(group_ids) - idx >= chunks:
                    obj_index.append(dict(
                        scene_id=scene_id,
                        object_id=object_id,
                        view_ids=[frame_index.loc[id]["view_id"]
                                  for id in group_ids[idx:idx+chunks]],
                        scene_ds_ids=[frame_index.loc[id]["frame_id"]
                                      for id in group_ids[idx:idx+chunks]]
                    ))

        self.obj_index = pd.DataFrame(obj_index)
        self.scene_ds = scene_ds

        # pixelNeRF settings
        self.z_near = 0.2
        self.z_far = 4.0
        self.lindisp = False
        self.img_to_tensor = get_image_to_tensor_balanced()

    def __len__(self):
        return len(self.obj_index)

    def __getitem__(self, chunk_id):
        row = self.obj_index.iloc[chunk_id]
        object_id = row.object_id
        object_label = f'obj_{int(object_id):06d}'

        all_imgs = []
        all_poses = []
        all_bbox = []
        all_focal = []
        all_c = []
        for scene_ds_id in row.scene_ds_ids:
            rgb, mask, obs = self.scene_ds[scene_ds_id]
            obj = [obj for obj in obs["objects"]
                   if obj["label"] == object_label][0]
            rgb, bbox, focal, c = extract_object(
                rgb, mask == obj["id_in_segm"], obs["camera"]["K"])
            pose = obj["T"].inverse * X_rot
            all_imgs.append(self.img_to_tensor(rgb))
            all_poses.append(pose.homogeneous)
            all_bbox.append(bbox)
            all_focal.append(focal)
            all_c.append(c)

        data = dict(
            images=torch.stack(all_imgs, axis=0),
            poses=torch.tensor(all_poses, dtype=torch.float32),
            bbox=torch.stack(all_bbox, axis=0),
            focal=torch.stack(all_focal, axis=0).float(),
            c=torch.stack(all_c, axis=0).float(),
        )

        # rgb_source, mask_source, obs_source = self.scene_ds[source_scene_ds_id]
        # rgb_target, mask_target, obs_target = self.scene_ds[target_scene_ds_id]
        # source_obj =
        # target_obj = [obj for obj in obs_target["objects"]
        #               if obj["label"] == object_label][0]

        # rgb_source, focal_source, c_source = extract_object(
        #     rgb_source, mask_source == source_obj["id_in_segm"], obs_source["camera"]["K"])
        # rgb_target, focal_target, c_target = extract_object(
        #     rgb_target, mask_target == target_obj["id_in_segm"], obs_target["camera"]["K"])

        # source = dict(
        #     img=rgb_source,
        #     focal=focal_source,
        #     c=c_source,
        # )

        # target = dict(
        #     img=rgb_target,
        #     focal=focal_target,
        #     c=c_target,
        # )

        # T = source_obj["T"] * target_obj["T"].inverse()
        return data
