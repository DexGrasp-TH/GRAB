# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
# ``

import sys

sys.path.append(".")
sys.path.append("..")
import numpy as np
import torch
import os
import glob
import smplx
import argparse
from pathlib import Path

from tqdm import tqdm
from tools.objectmodel import ObjectModel
from tools.cfg_parser import Config
from tools.utils import makepath, makelogger
from tools.meshviewer import Mesh
from tools.utils import parse_npz
from tools.utils import params2torch
from tools.utils import prepare_params
from tools.utils import to_cpu
from tools.utils import append2dict
from tools.utils import np2torch
from tools.utils import contact_ids as CONTACT_IDS
from tools.utils import hand_contact_ids as HAND_CONTACT_IDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INTENTS = ["lift", "pass", "offhand", "use", "all"]


class GRABDataSet(object):
    def __init__(self, cfg, logger=None, **params):
        self.cfg = cfg
        self.grab_path = cfg.grab_path
        self.out_path = cfg.out_path
        makepath(self.out_path)

        if logger is None:
            log_dir = os.path.join(self.out_path, "grab_preprocessing.log")
            self.logger = makelogger(log_dir=log_dir, mode="a").info
        else:
            self.logger = logger
        self.logger("Starting data preprocessing !")

        assert cfg.intent in INTENTS

        self.intent = cfg.intent
        self.logger("intent:%s --> processing %s sequences!" % (self.intent, self.intent))

        if cfg.splits is None:
            raise ValueError("No data splits provided")
        else:
            assert isinstance(cfg.splits, dict)
            self.splits = cfg.splits

        self.all_seqs = glob.glob(self.grab_path + "/*/*.npz")

        ## to be filled
        self.selected_seqs = []
        self.obj_based_seqs = {}
        self.sbj_based_seqs = {}
        self.split_seqs = {}

        self.process_sequences()
        self.data_preprocessing(cfg)

    def data_preprocessing(self, cfg):
        self.subject_mesh = {}
        self.obj_info = {}
        self.sbj_info = {}

        contact_id_to_name = {val: key for key, val in CONTACT_IDS.items()}
        hand_contact_values = list(HAND_CONTACT_IDS.values())

        for split in self.split_seqs.keys():
            self.logger("Processing data for %s split." % (split))
            self.logger("Number of sequences: %d" % len(self.split_seqs[split]))

            # 统计计数初始化
            n_grasps = 0
            n_right_hand = 0
            n_left_hand = 0
            n_both_hand = 0
            total_frames_in_split = 0

            # 设定阈值
            MIN_GRASP_DURATION = 10

            for sequence in tqdm(self.split_seqs[split], desc=f"Processing {split} sequences"):
                # --- 1. 单序列数据读取与过滤 ---
                seq_data = parse_npz(sequence)
                obj_name = seq_data.obj_name
                sbj_id = seq_data.sbj_id
                n_comps = seq_data.n_comps
                gender = seq_data.gender
                frame_mask = self.filter_contact_frames(seq_data)

                T = frame_mask.sum()
                if T < 1:
                    continue
                total_frames_in_split += T

                # --- 2. 准备单序列参数 (不再存入大列表) ---
                current_seq_body = prepare_params(seq_data.body.params, frame_mask)
                current_seq_rhand = prepare_params(seq_data.rhand.params, frame_mask)
                current_seq_lhand = prepare_params(seq_data.lhand.params, frame_mask)
                current_seq_object = prepare_params(seq_data.object.params, frame_mask)

                # --- 3. 顶点计算 (如果配置需要) ---
                # 注意：这里我们只计算当前序列的顶点，存入临时字典以便切片
                current_body_data = {**current_seq_body}
                current_rhand_data = {**current_seq_rhand}
                current_lhand_data = {**current_seq_lhand}
                current_object_data = {**current_seq_object}

                if cfg.save_body_verts:
                    sbj_vtemp = self.load_sbj_verts(sbj_id, seq_data)
                    sbj_m = smplx.create(
                        model_path=cfg.model_path,
                        model_type="smplx",
                        gender=gender,
                        num_pca_comps=n_comps,
                        v_template=sbj_vtemp,
                        batch_size=T,
                    )
                    sbj_parms = params2torch(current_seq_body)
                    current_body_data["verts"] = to_cpu(sbj_m(**sbj_parms).vertices)

                if cfg.save_lhand_verts:
                    lh_mesh = os.path.join(self.grab_path, "..", seq_data.lhand.vtemp)
                    lh_vtemp = np.array(Mesh(filename=lh_mesh).vertices)
                    lh_m = smplx.create(
                        model_path=cfg.model_path,
                        model_type="mano",
                        is_rhand=False,
                        v_template=lh_vtemp,
                        num_pca_comps=n_comps,
                        flat_hand_mean=True,
                        batch_size=T,
                    )
                    lh_parms = params2torch(current_seq_lhand)
                    current_lhand_data["verts"] = to_cpu(lh_m(**lh_parms).vertices)

                if cfg.save_rhand_verts:
                    rh_mesh = os.path.join(self.grab_path, "..", seq_data.rhand.vtemp)
                    rh_vtemp = np.array(Mesh(filename=rh_mesh).vertices)
                    rh_m = smplx.create(
                        model_path=cfg.model_path,
                        model_type="mano",
                        is_rhand=True,
                        v_template=rh_vtemp,
                        num_pca_comps=n_comps,
                        flat_hand_mean=True,
                        batch_size=T,
                    )
                    rh_parms = params2torch(current_seq_rhand)
                    current_rhand_data["verts"] = to_cpu(rh_m(**rh_parms).vertices)

                # 物体顶点与接触
                obj_info = self.load_obj_verts(obj_name, seq_data, cfg.n_verts_sample)
                current_object_data["name"] = obj_name
                if cfg.save_object_scale:
                    verts = obj_info["verts_sample"]
                    v_extent = np.max(verts, axis=0) - np.min(verts, axis=0)
                    diagonal_scale = np.linalg.norm(v_extent)
                    current_object_data["scale"] = diagonal_scale
                if cfg.save_object_verts:
                    obj_m = ObjectModel(v_template=obj_info["verts_sample"], batch_size=T)
                    obj_parms = params2torch(current_seq_object)
                    current_object_data["verts"] = to_cpu(obj_m(**obj_parms).vertices)

                # 获取当前序列的接触数据
                current_contact_data = seq_data.contact.object[frame_mask][:, obj_info["verts_sample_id"]]
                if cfg.save_contact:
                    current_body_data["contact"] = seq_data.contact.body[frame_mask]
                    current_object_data["contact"] = current_contact_data

                # --- 4. 立即切分当前序列的抓取 (Segment Grasps) ---
                mask = np.isin(current_contact_data, hand_contact_values)
                filtered_contact_data = np.where(mask, current_contact_data, 0)
                n_contact_on_obj = np.sum(filtered_contact_data > 0, axis=1)

                is_contacting = n_contact_on_obj > 0
                diff = np.diff(is_contacting.astype(int), prepend=0, append=0)
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]

                for segment_idx, (start, end) in enumerate(zip(starts, ends)):
                    if (end - start) < MIN_GRASP_DURATION:
                        continue

                    segment_contacts = n_contact_on_obj[start:end]
                    grasp_frame_id = start + np.argmax(segment_contacts)

                    contacts = filtered_contact_data[grasp_frame_id]
                    right_hand = any(contact_id_to_name[c].startswith("R_") for c in contacts if c > 0)
                    left_hand = any(contact_id_to_name[c].startswith("L_") for c in contacts if c > 0)

                    grasp_type = (
                        "both_hand"
                        if (right_hand and left_hand)
                        else "right_hand"
                        if right_hand
                        else "left_hand"
                        if left_hand
                        else "none"
                    )

                    # 计数统计
                    n_grasps += 1
                    if grasp_type == "right_hand":
                        n_right_hand += 1
                    elif grasp_type == "left_hand":
                        n_left_hand += 1
                    elif grasp_type == "both_hand":
                        n_both_hand += 1

                    # 提取片段数据
                    grasp_info = {
                        "seq_source": sequence,
                        "segment_id": segment_idx,
                        "start_frame": start,
                        "end_frame": end,
                        "best_grasp_frame": grasp_frame_id,
                        "body": {k: v[grasp_frame_id] for k, v in current_body_data.items()},
                        "object": {
                            k: v if k in ["name", "scale"] else v[grasp_frame_id]
                            for k, v in current_object_data.items()
                        },
                        "left_hand": {k: v[grasp_frame_id] for k, v in current_lhand_data.items()},
                        "right_hand": {k: v[grasp_frame_id] for k, v in current_rhand_data.items()},
                        "grasp_type": grasp_type,
                    }

                    # 保存到硬盘
                    original_seq_path = sequence
                    rel_dir = Path(original_seq_path).parent.name
                    save_dir = os.path.join(self.out_path, rel_dir)
                    base_name = os.path.basename(original_seq_path).split(".")[0]
                    os.makedirs(save_dir, exist_ok=True)

                    save_filename = f"{base_name}_seg{segment_idx}.npy"
                    save_full_path = os.path.join(save_dir, save_filename)
                    np.save(save_full_path, grasp_info)

            # Split 总结打印
            self.logger("Processing for %s split finished" % split)
            self.logger("Total number of frames for %s split is:%d" % (split, total_frames_in_split))
            self.logger("Total number of sequences for %s split is:%d" % (split, len(self.split_seqs[split])))
            self.logger("Number of segmented grasps: %d" % n_grasps)
            self.logger("Number of right hand grasps: %d" % n_right_hand)
            self.logger("Number of left hand grasps: %d" % n_left_hand)
            self.logger("Number of both hand grasps: %d" % n_both_hand)

    def process_sequences(self):
        for sequence in self.all_seqs:
            subject_id = sequence.split("/")[-2]
            action_name = os.path.basename(sequence)
            object_name = action_name.split("_")[0]

            # filter data based on the motion intent
            if self.intent == "all":
                pass
            elif self.intent == "use" and any(intnt in action_name for intnt in INTENTS[:3]):
                continue
            elif self.intent not in action_name:
                continue

            # group motion sequences based on objects
            if object_name not in self.obj_based_seqs:
                self.obj_based_seqs[object_name] = [sequence]
            else:
                self.obj_based_seqs[object_name].append(sequence)

            # group motion sequences based on subjects
            if subject_id not in self.sbj_based_seqs:
                self.sbj_based_seqs[subject_id] = [sequence]
            else:
                self.sbj_based_seqs[subject_id].append(sequence)

            # split train, val, and test sequences
            self.selected_seqs.append(sequence)
            for key in self.splits:
                if key not in self.split_seqs.keys():
                    self.split_seqs[key] = []
                if object_name in self.splits[key]:
                    self.split_seqs[key].append(sequence)

    def filter_contact_frames(self, seq_data):
        if self.cfg.only_contact:
            frame_mask = (seq_data["contact"]["object"] > 0).any(axis=1)
        else:
            frame_mask = (seq_data["contact"]["object"] > -1).any(axis=1)
        return frame_mask

    def load_obj_verts(self, obj_name, seq_data, n_verts_sample=512):
        mesh_path = os.path.join(self.grab_path, "..", seq_data.object.object_mesh)
        if obj_name not in self.obj_info:
            np.random.seed(100)
            obj_mesh = Mesh(filename=mesh_path)
            verts_obj = np.array(obj_mesh.vertices)
            faces_obj = np.array(obj_mesh.faces)

            if verts_obj.shape[0] > n_verts_sample:
                verts_sample_id = np.random.choice(verts_obj.shape[0], n_verts_sample, replace=False)
            else:
                verts_sample_id = np.arange(verts_obj.shape[0])

            verts_sampled = verts_obj[verts_sample_id]
            self.obj_info[obj_name] = {
                "verts": verts_obj,
                "faces": faces_obj,
                "verts_sample_id": verts_sample_id,
                "verts_sample": verts_sampled,
                "obj_mesh_file": mesh_path,
            }

        return self.obj_info[obj_name]

    def load_sbj_verts(self, sbj_id, seq_data):
        mesh_path = os.path.join(self.grab_path, "..", seq_data.body.vtemp)
        if sbj_id in self.sbj_info:
            sbj_vtemp = self.sbj_info[sbj_id]
        else:
            sbj_vtemp = np.array(Mesh(filename=mesh_path).vertices)
            self.sbj_info[sbj_id] = sbj_vtemp
        return sbj_vtemp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRAB-vertices")

    parser.add_argument("--grab-path", default=None, type=str, help="The path to the downloaded grab data")
    parser.add_argument("--out-path", default=None, type=str, help="The path to the folder to save the processed data")
    parser.add_argument("--model-path", default=None, type=str, help="The path to the folder containing smplx models")
    parser.add_argument("--object-set", default="mug", type=str, help="The set of objects to process")

    args = parser.parse_args()

    grab_path = args.grab_path
    out_path = args.out_path
    model_path = args.model_path

    process_id = "GRAB_V01"  # choose an appropriate ID for the processed data
    model_path = "model/models_smplx_v1_1/models"
    grab_path = "/data/dataset/GRAB/extract/grab"
    out_path = "/data/dataset/GRAB/segment_grasps"

    if args.object_set == "all":
        grab_splits = {  # all objects
            "test": [
                "airplane",
                "alarmclock",
                "apple",
                "banana",
                "binoculars",
                "body",
                "bowl",
                "camera",
                "coffeemug",
                "cubelarge",
                "cubemedium",
                "cubemiddle",
                "cubesmall",
                "cup",
                "cylinderlarge",
                "cylindermedium",
                "cylindersmall",
                "doorknob",
                "duck",
                "elephant",
                "eyeglasses",
                "flashlight",
                "flute",
                "fryingpan",
                "gamecontroller",
                "hammer",
                "hand",
                "headphones",
                "knife",
                "lightbulb",
                "mouse",
                "mug",
                "phone",
                "piggybank",
                "pyramidlarge",
                "pyramidmedium",
                "pyramidsmall",
                "rubberduck",
                "scissors",
                "spherelarge",
                "spheremedium",
                "spheresmall",
                "stamp",
                "stanfordbunny",
                "stapler",
                "table",
                "teapot",
                "toothbrush",
                "toothpaste",
                "toruslarge",
                "torusmedium",
                "torussmall",
                "train",
                "watch",
                "waterbottle",
                "wineglass",
                "wristwatch",
            ],
        }
    else:
        grab_splits = {
            "test": [args.object_set],
        }

    cfg = {
        "intent": "all",  # from 'all', 'use' , 'pass', 'lift' , 'offhand'
        "only_contact": False,  # if True, returns only frames with contact
        "save_body_verts": False,  # if True, will compute and save the body vertices
        "save_lhand_verts": False,  # if True, will compute and save the body vertices
        "save_rhand_verts": False,  # if True, will compute and save the body vertices
        "save_object_scale": True,
        "save_object_verts": False,
        "save_contact": True,  # if True, will add the contact info to the saved data
        # splits
        "splits": grab_splits,
        # IO path
        "grab_path": grab_path,
        "out_path": out_path,
        # number of vertices samples for each object
        "n_verts_sample": 1024,
        # body and hand model path
        "model_path": model_path,
    }

    cwd = os.getcwd()
    default_cfg_path = os.path.join(cwd, "../configs/grab_preprocessing_cfg.yaml")
    cfg = Config(default_cfg_path=default_cfg_path, **cfg)
    cfg.write_cfg(write_path=cfg.out_path + "/grab_preprocessing_cfg.yaml")

    log_dir = os.path.join(cfg.out_path, "grab_processing.log")
    logger = makelogger(log_dir=log_dir, mode="a").info

    GRABDataSet(cfg, logger)
