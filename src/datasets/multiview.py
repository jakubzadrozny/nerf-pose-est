import numpy as np
import pandas as pd
import torch


class ViewsDataset:
    def __init__(self, obj_ds):
        obj_index = obj_ds.obj_index.copy()
        obj_index["obj_idx"] = np.arange(len(obj_index))
        obj_index["batch_idx"] = obj_index.apply(
            lambda row: np.arange(len(row["view_ids"])),
            axis=1
        )

        obj_index = obj_index.explode(
            ["view_ids", "batch_idx", "scene_ds_ids"], ignore_index=True)
        groups = obj_index.groupby(["scene_id"]).groups
        groups_fine = obj_index.groupby(["scene_id", "view_ids"])

        view_index = []
        random_state = np.random.RandomState(0)

        for scene_id, group_ids in groups.items():
            views = obj_index.loc[group_ids, "view_ids"].drop_duplicates()
            views_list = list(random_state.permutation(np.array(views)))
            view_pairs = zip(views_list, views_list[1:])
            for view1, view2 in view_pairs:
                group1 = groups_fine.get_group((scene_id, view1))
                group2 = groups_fine.get_group((scene_id, view2))
                common = group1.merge(
                    group2, on="object_id", suffixes=("_1", "_2"))
                if len(common) > 1:
                    view_index.append(dict(
                        scene_id=scene_id,
                        view_1=view1,
                        view_2=view2,
                        scene_id_1=common["scene_ds_ids_1"].iloc[0],
                        scene_id_2=common["scene_ds_ids_2"].iloc[0],
                        objects=list(common["object_id"]),
                        obj_idx_1=list(common["obj_idx_1"]),
                        batch_idx_1=list(common["batch_idx_1"]),
                        obj_idx_2=list(common["obj_idx_2"]),
                        batch_idx_2=list(common["batch_idx_2"]),
                    ))

        self.view_index = pd.DataFrame(view_index)

    def __len__(self):
        return len(self.view_index)

    def __getitem__(self, idx):
        return self.view_index.iloc[idx].to_dict()


def zip_explode(row, rs):
    pi = rs.permutation(len(row["objects"]))
    pi_pairs = zip(pi, pi[1:])
    rows = []
    for idx1, idx2 in pi_pairs:
        new_row = dict(row)
        new_row.update(dict(
            objects=[row["objects"][idx1], row["objects"][idx2]],
            obj_idx_1=[row["obj_idx_1"][idx1], row["obj_idx_1"][idx2]],
            batch_idx_1=[row["batch_idx_1"][idx1], row["batch_idx_1"][idx2]],
            obj_idx_2=[row["obj_idx_2"][idx1], row["obj_idx_2"][idx2]],
            batch_idx_2=[row["batch_idx_2"][idx1], row["batch_idx_2"][idx2]],
        ))
        rows.append(new_row)
    return rows


class DoubleViewDataset:
    def __init__(self, obj_ds):
        self.obj_ds = obj_ds
        view_ds = ViewsDataset(obj_ds)
        index = view_ds.view_index.copy()
        rs = np.random.RandomState(0)
        rows = sum(index.apply(lambda row: zip_explode(row, rs), axis=1), [])
        self.index = pd.DataFrame(rows)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        data = self.index.iloc[idx]
        obj1_view1 = self.obj_ds[data["obj_idx_1"][0]]
        obj2_view1 = self.obj_ds[data["obj_idx_1"][1]]
        obj1_view2 = self.obj_ds[data["obj_idx_2"][0]]
        obj2_view2 = self.obj_ds[data["obj_idx_2"][1]]

        inputs = {
            k: torch.stack([
                obj1_view1[k][data["batch_idx_1"][0]],
                obj2_view1[k][data["batch_idx_1"][1]]
            ], dim=0)
            for k in ["images", "focal", "bbox"]
        }
        targets = {
            k: torch.stack([
                obj1_view2[k][data["batch_idx_2"][0]],
                obj2_view2[k][data["batch_idx_2"][1]]
            ], dim=0)
            for k in ["images", "focal", "bbox"]
        }
        return inputs, targets
