import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle as pkl
from dataset.utils.misc import to_torch
import dataset.utils.rotation_conversions as geometry
from dataset.utils.smpl_skeleton import SMPLSkeleton
from .scaler import MinMaxScaler  # .scaler
from typing import Any

class Normalizer:
    def __init__(self, data):
        flat = data.reshape(-1, data.shape[-1])
        self.scaler = MinMaxScaler((-1, 1), clip=True)
        self.scaler.fit(flat)

    def normalize(self, x):
        batch, n_persons, seq, ch = x.shape
        x = x.reshape(-1, ch)
        return self.scaler.transform(x).reshape((batch, n_persons, seq, ch))

    def unnormalize(self, x):
        batch, n_persons, seq, ch = x.shape
        x = x.reshape(-1, ch)
        x = torch.clip(x, -1, 1)  # clip to force compatibility
        return self.scaler.inverse_transform(x).reshape((batch, n_persons, seq, ch))

class EdgeGdanceDataset(Dataset):
    def __init__(self,
                 datapath="/home/cvlab/xj/data/gdance/",   # "/fs03/ha81/jxu/Data/gdance"
                 split_file="train_split_sequence_names.txt",
                 target_seq_len=150,
                 sampling_stride=1,
                 split="train",
                 data_stride=2,
                 use_contact=True,
                 max_persons=5,
                 normalizer: Any = None,
    ):

        self.datapath = datapath
        self.split_file = split_file
        self.target_seq_len = target_seq_len
        self.sampling_stride = sampling_stride
        self.split = split
        self.data_stride = data_stride
        self.use_contact = use_contact
        self.max_persons = max_persons

        if use_contact:
            self.smpl = SMPLSkeleton()

        self.sequences = self.load_sequences()

        # Preprocess all data and store in self.data
        self.data = self.preprocess_all_data()

        # Initialize normalizer and apply normalization
        self.normalizer = Normalizer(self.data["pose"])
        self.data["pose"] = self.normalizer.normalize(self.data["pose"])

    def load_sequences(self):
        with open(os.path.join(self.datapath, self.split_file), 'r') as f:
            sequences = [line.strip() for line in f.readlines()]

        # Filter sequences based on target_seq_len
        sequences = [seq for seq in sequences if int(seq.split("_")[-1]) - int(seq.split("_")[-2]) >= self.target_seq_len]

        return sequences

    def preprocess_all_data(self):
        all_poses = []
        all_filenames = []
        all_wavs = []
        self.frame_indices = []

        for seq_name in self.sequences:
            # seq_name = 'K4MpjnkGFOw_05_0_242'
            motion_data = self._load_motion_data(seq_name)
            seq_len = motion_data['poses'].shape[1]
            frame_ix = self._sample_frames(seq_len)

            self.frame_indices.append(frame_ix)  # Store frame indices for consistent sampling

            if max(frame_ix) >= seq_len:
                frame_ix = np.clip(frame_ix, 0, seq_len - 1)

            processed_motion = self._process_motion(motion_data, frame_ix)

            # Padding for consistent shape
            eff_n_persons = processed_motion.shape[0]
            if eff_n_persons >= self.max_persons:
                sampled_persons = np.random.choice(eff_n_persons, size=self.max_persons, replace=False)
                processed_motion = processed_motion[sampled_persons]
            else:
                padded_motion = torch.zeros((self.max_persons, self.target_seq_len, processed_motion.shape[-1]))
                padded_motion[:eff_n_persons, :, :] = processed_motion
                processed_motion = padded_motion

            # Flatten num_persons
            '''
            for person_idx in range(self.max_persons):
                all_poses.append(processed_motion[person_idx])
                all_filenames.append(os.path.join(self.datapath, "motions_smpl", f"{seq_name}.pkl"))
                all_wavs.append(os.path.join(self.datapath, "jukebox_features", f"{seq_name}.npy"))
            '''
            all_poses.append(processed_motion)
            all_filenames.append(os.path.join(self.datapath, "motions_smpl", f"{seq_name}.pkl"))
            all_wavs.append(os.path.join(self.datapath, "jukebox_features", f"{seq_name}.npy"))

        all_poses = torch.stack(all_poses)  # Combine all poses into a single tensor

        return {
            "pose": all_poses,
            "filenames": all_filenames,
            "wavs": all_wavs,
        }

    def __len__(self):
        return len(self.data["pose"])

    def _load_motion_data(self, seq_name):
        motion_file = os.path.join(self.datapath, "motions_smpl", f"{seq_name}.pkl")
        with open(motion_file, "rb") as f:
            motion_data = pkl.load(f)

        smpl_poses = motion_data['smpl_poses']
        if smpl_poses.shape[-1] < 23 * 3:
            smpl_poses = np.concatenate([smpl_poses, np.zeros(list(smpl_poses.shape[:-1]) + [23 * 3 - smpl_poses.shape[-1]])], axis=-1)
        if 'smpl_orients' in motion_data:
            smpl_orients = motion_data['smpl_orients']
            smpl_poses = np.concatenate([smpl_orients, smpl_poses], axis=-1)

        smpl_trans = motion_data['root_trans']

        return {
            'poses': smpl_poses,
            'trans': smpl_trans
        }

    def _load_music_features(self, seq_name, frame_ix):
        music_file = os.path.join(self.datapath, "jukebox_features", f"{seq_name}.npy")
        music_data = np.load(music_file)
        # print(f"seq_name: {seq_name}, frame_ix: {frame_ix}")
        return music_data[frame_ix], music_file

    def _sample_frames(self, seq_len):
        # sampling = 'conseq'
        if seq_len < self.target_seq_len:
            padding = np.full(self.target_seq_len - seq_len, seq_len - 1, dtype=int)
            frame_ix = np.concatenate([np.arange(seq_len), padding])
        else:
            stride_max = (seq_len - 1) // (self.target_seq_len - 1)
            if self.data_stride == -1 or self.data_stride * (self.target_seq_len - 1) >= seq_len:  # if the specified stride (self.sampling_stride) exceed the max valid number.
                stride = stride_max
            else:
                stride = self.data_stride
            # frame_ix = np.arange(0, self.target_seq_len * stride, stride)[:self.target_seq_len]
            lastone = stride * (self.target_seq_len - 1)
            start_fr_max = seq_len - lastone - 1

            shift = random.randint(0, max(0, start_fr_max - 1))  # random start_frame
            # shift = 0
            frame_ix = shift + np.arange(0, lastone + 1, stride)

        return frame_ix

    def _process_motion(self, motion_data, frame_ix):
        poses = motion_data['poses'][:, frame_ix]
        poses = to_torch(poses).view(poses.shape[0], poses.shape[1], -1, 3)  # (n_persons, seq_len, J, 3)

        if self.use_contact:
            trans = to_torch(motion_data['trans'][:, frame_ix])
            positions = self.smpl.forward(poses, trans)
            feet = positions[:, :, (7, 8, 10, 11)]
            feetv = torch.zeros(feet.shape[:3])
            feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
            contacts = (feetv < 0.01).to(poses)  # cast to right dtype
        else:
            trans = to_torch(motion_data['trans'][:, frame_ix])
            contacts = None

        poses = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(poses))
        poses = poses.view(poses.shape[0], poses.shape[1], -1)

        motion_output = torch.cat([trans, poses], dim=-1)
        if contacts is not None:
            motion_output = torch.cat([motion_output, contacts.float()], dim=-1)

        return motion_output

    def __getitem__(self, idx):
        pose = self.data["pose"][idx]
        '''
        seq_name = self.sequences[idx // self.max_persons]  # Map back to the sequence name
        frame_ix = self.frame_indices[idx // self.max_persons]  # Use the stored frame indices
        '''
        seq_name = self.sequences[idx]
        frame_ix = self.frame_indices[idx]

        music, music_file = self._load_music_features(seq_name, frame_ix)
        motion_file = self.data["wavs"][idx]

        return pose, torch.from_numpy(music).float(), motion_file, music_file

if __name__ == "__main__":
    datapath = "/home/cvlab/xj/data/gdance/"
    split_file = "train_split_sequence_names.txt"
    dataset = EdgeGdanceDataset(datapath=datapath, split_file=split_file, split="test", max_persons=5)

    print("Dataset length:", len(dataset))
    sample_idx = 249
    sample_data = dataset[sample_idx]

    print(f"Sample {sample_idx} data:")
    print("Pose shape:", sample_data[0].shape)
    print("Music shape:", sample_data[1].shape)
    print("Motion file:", sample_data[2])
    print("Music file:", sample_data[3])
