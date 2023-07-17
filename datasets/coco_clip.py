"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import torchvision.transforms as TT

import torchvision.transforms.functional as F
import os
import random

from copy import deepcopy

from .coco import shuffled_aug, ConvertCocoPolysToMask
from .transforms import resize, hflip, crop, box_xyxy_to_cxcywh


class CocoClipDetection(TvCocoDetection):
    """
    The labels in each frame is one dict which contains all the bounding boxes.
    """
    def __init__(self,
                 img_folder,
                 ann_file,
                 transforms,
                 return_masks,
                 cache_mode=False,
                 local_rank=0,
                 local_size=1,
                 num_seq_frames=2,
                 num_global_frames=3,
                 shuffled_aug=None):
        super(CocoClipDetection, self).__init__(img_folder,
                                                ann_file,
                                                cache_mode=cache_mode,
                                                local_rank=local_rank,
                                                local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.num_seq_frames = num_seq_frames
        self.num_global_frames = num_global_frames
        self.shuffled_aug = shuffled_aug

    def __getitem__(self, idx):
        img, target = super(CocoClipDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        path = self.coco.loadImgs(image_id)[0]['file_name']
        path_name_id = int(path[-10:-4])
        path_filetyp = path[-4:]
        path_dirname = os.path.dirname(path)
        max_frm_id = len(os.listdir(os.path.join(self.root, path_dirname))) - 1
        ref_frms_id, ref_frms_iamges = [], []

        if self.num_seq_frames > 0:
            half_length = self.num_seq_frames // 2
            start_num = path_name_id - half_length
            if self.num_seq_frames % 2 != 0:
                end_num = path_name_id + half_length + 1
            else:
                end_num = path_name_id + half_length

            supp_frms_id = list(range(start_num, end_num + 1))
            supp_frms_id.remove(path_name_id)
            supp_frms_id = [min(max(i, 0), max_frm_id) for i in supp_frms_id]
            supp_frms = [
                self.get_image(f'{path_dirname}/{i:06d}{path_filetyp}')
                for i in supp_frms_id
            ]
            ref_frms_id += supp_frms_id
            ref_frms_iamges += supp_frms

        # get the global frames
        if self.num_global_frames > 0:
            select_range = [0, max_frm_id]
            select_candidate = list(range(*select_range))
            if path_name_id in select_candidate:
                select_candidate.remove(path_name_id)
            glb_file_id = random.sample(select_candidate, self.num_global_frames)
            global_frms = [
                self.get_image(f'{path_dirname}/{i:06d}{path_filetyp}')
                for i in glb_file_id
            ]
            ref_frms_id += glb_file_id
            ref_frms_iamges += global_frms

        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # to tensor
        if self.num_seq_frames > 0:
            img = [img] + supp_frms

        # suffled video augmentation
        if self.shuffled_aug is not None:
            global_frms, target_ = shuffled_aug(global_frms, target,
                                                self.shuffled_aug)
            global_frms, target_ = T.resize(global_frms, target, img[0].size,
                                            1333)

        if self.num_global_frames > 0:
            if isinstance(img, (list, tuple)):
                img = img + global_frms
            else:
                img = [img] + global_frms

        ref_frms_targets = self._get_ref_frame_targets(path_dirname, path_filetyp, ref_frms_id, ref_frms_iamges)
        target = [target] + ref_frms_targets

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

    def _get_ref_frame_targets(self, path_dirname, path_filetyp, ref_frms_id, ref_frms_images):
        ref_frms_images = deepcopy(ref_frms_images)
        file_names = [f'{path_dirname}/{i:06d}{path_filetyp}' for i in ref_frms_id]
        img_ids = []
        for file_name in file_names:
            img_ids += [img_id for img_id, img in self.coco.imgs.items() if img['file_name'] == file_name]
        ref_frame_targets = [self.coco.imgToAnns[img_id] for img_id in img_ids]
        tmp_frames, tmp_targets = [], []
        for temporal_img, temporal_target, img_id in zip(ref_frms_images, ref_frame_targets, img_ids):
            temporal_target = {'image_id': img_id, 'annotations': temporal_target}
            temporal_img, temporal_target = self.prepare(temporal_img, temporal_target)
            tmp_frames.append(temporal_img)
            tmp_targets.append(temporal_target)

        return tmp_targets


class TestCocoClipDetection(CocoClipDetection):
    def __init__(
            self,
            img_folder,
            ann_file,
            transforms,
            return_masks,
            cache_mode=False,
            local_rank=0,
            local_size=1,
            num_seq_frames=2,
            num_global_frames=3,
            shuffled_aug=None
    ):
        super().__init__(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
            num_seq_frames=num_seq_frames,
            num_global_frames=num_global_frames,
            shuffled_aug=shuffled_aug
        )
        start = num_seq_frames // 2
        self.ids = self.ids[start::num_seq_frames + 1]

    # def __getitem__(self, idx):
    #     img, target = super(CocoClipDetection, self).__getitem__(idx)
    #     image_id = self.ids[idx]
    #     path = self.coco.loadImgs(image_id)[0]['file_name']
    #     path_name_id = int(path[-10:-4])
    #     path_filetyp = path[-4:]
    #     path_dirname = os.path.dirname(path)
    #     max_frm_id = len(os.listdir(os.path.join(self.root, path_dirname))) - 1
    #     ref_frms_id, ref_frms_iamges = [], []
    #
    #     if self.num_seq_frames > 0:
    #         half_length = self.num_seq_frames // 2
    #         start_num = path_name_id - half_length
    #         if self.num_seq_frames % 2 != 0:
    #             end_num = path_name_id + half_length + 1
    #         else:
    #             end_num = path_name_id + half_length
    #
    #         supp_frms_id = list(range(start_num, end_num + 1))
    #         supp_frms_id.remove(path_name_id)
    #         supp_frms_id = [min(max(i, 0), max_frm_id) for i in supp_frms_id]
    #         supp_frms = [
    #             self.get_image(f'{path_dirname}/{i:06d}{path_filetyp}')
    #             for i in supp_frms_id
    #         ]
    #         ref_frms_id += supp_frms_id
    #         ref_frms_iamges += supp_frms
    #
    #     target = {'image_id': image_id, 'annotations': target}
    #     img, target = self.prepare(img, target)
    #
    #     # to tensor
    #     if self.num_seq_frames > 0:
    #         img = [img] + supp_frms
    #
    #     ref_frms_targets = self._get_ref_frame_targets(path_dirname, path_filetyp, ref_frms_id, ref_frms_iamges)
    #     target = [target] + ref_frms_targets
    #
    #     if self._transforms is not None:
    #         img, target = self._transforms(img, target)
    #
    #     return img, target


class MultiFramesMultiTargetsRandomResize(object):
    def __init__(
            self,
            sizes,
            max_size=None
    ):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target):
        assert isinstance(img, (list, tuple)) and isinstance(target, (list, tuple))
        size = random.choice(self.sizes)
        for i in range(len(img)):
            img[i], target[i] = resize(img[i], target[i], size, self.max_size)
        return img, target


class MultiFramesMultiTargetsRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        assert isinstance(img, (list, tuple)) and isinstance(target, (list, tuple))
        if random.random() < self.p:
            for i in range(len(img)):
                img[i], target[i] = hflip(img[i], target[i])
        return img, target


class MultiFramesMultiTargetsRandomSizeCrop(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, imgs, targets):
        assert isinstance(imgs, (list, tuple)) and isinstance(targets, (list, tuple))
        w = random.randint(self.min_size, min(imgs[0].width, self.max_size))
        h = random.randint(self.min_size, min(imgs[0].height, self.max_size))
        region = TT.RandomCrop.get_params(imgs[0], [h, w])
        for i in range(len(imgs)):
            imgs[i], targets[i] = crop(imgs[i], targets[i], region)
        return imgs, targets


class MultiFramesMultiTargetsNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        for i in range(len(image)):
            image[i] = F.normalize(image[i], mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        for l in range(len(image)):
            h, w = image[l].shape[-2:]
            if "boxes" in target[l]:
                boxes = target[l]["boxes"]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target[l]["boxes"] = boxes

        return image, target


def make_transforms_for_multi_frames_multi_targets(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        MultiFramesMultiTargetsNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576]
    if image_set == 'train':
        return T.Compose([
            MultiFramesMultiTargetsRandomHorizontalFlip(),
            T.RandomSelect(
                MultiFramesMultiTargetsRandomResize(scales, max_size=1333),
                T.Compose([
                    MultiFramesMultiTargetsRandomResize([400, 500, 600]),
                    MultiFramesMultiTargetsRandomSizeCrop(384, 600),
                    MultiFramesMultiTargetsRandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            MultiFramesMultiTargetsRandomResize([512], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset(image_set, args):
    root = Path(args.data_coco_lite_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    data_mode = args.data_mode
    PATHS = {
        "train":
            (root / "rawframes", root / f'imagenet_vid_train_{data_mode}.json'),
        "val": (root / "rawframes", root / f'imagenet_vid_val.json')
    }

    img_folder, ann_file = PATHS[image_set]
    if image_set == 'train':
        dataset = CocoClipDetection(img_folder,
                                    ann_file,
                                    transforms=make_transforms_for_multi_frames_multi_targets(image_set),
                                    return_masks=args.masks,
                                    cache_mode=args.cache_mode,
                                    local_rank=get_local_rank(),
                                    local_size=get_local_size(),
                                    num_global_frames=args.num_global_frames,
                                    num_seq_frames=args.num_support_frames,
                                    )
    else:
        dataset = TestCocoClipDetection(
            img_folder,
            ann_file,
            transforms=make_transforms_for_multi_frames_multi_targets(image_set),
            return_masks=args.masks,
            cache_mode=args.cache_mode,
            local_rank=get_local_rank(),
            local_size=get_local_size(),
            num_global_frames=args.num_global_frames,
            num_seq_frames=args.num_support_frames_testing,
        )
        # dataset = CocoClipDetection(
        #     img_folder,
        #     ann_file,
        #     transforms=make_transforms_for_multi_frames_multi_targets(image_set),
        #     return_masks=args.masks,
        #     cache_mode=args.cache_mode,
        #     local_rank=get_local_rank(),
        #     local_size=get_local_size(),
        #     num_global_frames=args.num_global_frames,
        #     num_seq_frames=args.num_support_frames,
        # )
    return dataset

