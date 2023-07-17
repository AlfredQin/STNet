import torch
import torch.nn.functional as F
from copy import deepcopy
from ...cva_net import DeformableDETR, NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid
from ...backbone import build_backbone
from .temporal_transformer import build_temporal_transformer
from ...cva_net import DETRsegm, SetCriterion, PostProcess, PostProcessPanoptic, PostProcessSegm
from ...matcher import build_matcher


class DETR(DeformableDETR):
    def __init__(
            self,
            backbone,
            transformer,
            num_classes,
            num_queries,
            num_feature_levels,
            aux_loss=True,
            with_box_refine=False,
            two_stage=False,
    ):
        super().__init__(
            backbone,
            transformer,
            num_classes,
            num_queries,
            num_feature_levels,
            aux_loss=aux_loss,
            with_box_refine=with_box_refine,
            two_stage=two_stage,
        )

        # self.query_embed = torch.nn.ModuleList([deepcopy(self.query_embed) for _ in range(self.transformer.num_frames)])

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        bs, f, c, h, w = samples.tensors.shape
        samples.tensors = samples.tensors.view(-1, c, h, w)
        samples.mask = samples.mask.repeat(f, 1, 1)
        features, pos = self.backbone(samples)
        if self.num_feature_levels == 1:
            features, pos = [features[-1]], [pos[-1]]

        srcs, masks = [], []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            src_proj = self.input_proj[l](src)
            srcs.append(src_proj)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src_proj = self.input_proj[l](features[-1].tensors)
                else:
                    src_proj = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src_proj.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src_proj, mask)).to(src_proj.dtype)
                srcs.append(src_proj)
                masks.append(mask)
                pos.append(pos_l)

        # query_embeds = [query_embed.weight for query_embed in self.query_embed]
        query_embeds = [self.query_embed.weight for _ in range(3)]
        (hs,
         query_pos,
         init_reference,
         inter_references,
         level_start_index,
         valid_ratios,
         spatial_shapes) = self.transformer(
            srcs, masks, pos, query_embeds
        )

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        pred = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
        }
        pred_logits_list = torch.chunk(pred['pred_logits'], self.transformer.num_frames, dim=0)
        pred_boxes_list = torch.chunk(pred['pred_boxes'], self.transformer.num_frames, dim=0)
        pred = [{'pred_logits': pred_logits, 'pred_boxes': pred_boxes} for pred_logits, pred_boxes in
                zip(pred_logits_list, pred_boxes_list)]

        return pred

def build(args):
    num_classes = 3
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_temporal_transformer(args)
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
    }
    if args.umap_loss:
        weight_dict['loss_new'] = 0.5 * args.cls_loss_coef

    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors

