import torch
import torch.nn as nn

from ..deformable_transformer import (DeformableTransformer,
    DeformableTransformerEncoderLayer, DeformableTransformerEncoder, DeformableTransformerDecoderLayer,
     _get_clones, inverse_sigmoid, )
from .temporal_ms_deform_attn import TemporalMSDeformAttnEncoder, TemporalMSDeformAttnDecoder


class TemporalTransformer(DeformableTransformer):
    def __init__(
            self,
            d_model=256,
            num_frames=6,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            num_feature_levels=4,
            enc_connect_all_embeddings=True,
            enc_temporal_window=2,
            enc_n_curr_points=4,
            enc_n_temporal_points=2,
            dec_n_curr_points=4,
            dec_n_temporal_points=2,
            dec_instance_aware_att=True,
            with_gradient=False
    ):
        super(TemporalTransformer, self).__init__(
            d_model=d_model,
            nhead=nhead,
            num_feature_levels=num_feature_levels,
        )
        if enc_connect_all_embeddings:
            enc_temporal_window = num_frames - 1

        encoder_layer = TemporalMSTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation,
            t_window=enc_temporal_window,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_curr_points=enc_n_curr_points,
            n_temporal_points=enc_n_temporal_points
        )
        self.encoder = TemporalMSTransformerEncoder(
            encoder_layer, num_encoder_layers, enc_temporal_window, enc_connect_all_embeddings
        )

        dec_temporal_window = num_frames - 1
        decoder_layer = TemporalMSTransformerDecoderLayer(
            d_model, dim_feedforward, dropout, activation, dec_temporal_window, dec_instance_aware_att,
            num_feature_levels, nhead, dec_n_curr_points, dec_n_temporal_points
        )
        self.decoder = TemporalMSTransformerDecoder(
            decoder_layer, num_decoder_layers, with_gradient
        )

        self._reset_parameters()

    def forward(
            self,
            srcs,
            masks,
            pos_embeds,
            query_embed=None
    ):
        (src_flatten,
         mask_flatten,
         lvl_pos_embed_flatten,
         spatial_shapes,
         level_start_index,
         valid_ratios) = self.prepare_data(srcs, masks, pos_embeds)

        memory = self.encoder(
            src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten
        )

        T_, _, channels = memory.shape
        query_embed, tgt = torch.split(query_embed, channels, dim=1)
        query_embed = query_embed.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points

        hs, inter_references = self.decoder(
            tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed
        )
        inter_references_out = inter_references

        # This is for instance segmentation
        # offset = 0
        # memories = []
        # for src in srcs:
        #     _, _, height, width = src.shape
        #     memory_slice = memory[:, offset:offset + height * width].permute(2, 0, 1).view(1, channels, T_, height, width)
        #     memories.append(memory_slice)
        #     offset += height * width
        #
        return hs, query_embed, init_reference_out, inter_references_out, level_start_index, valid_ratios, spatial_shapes

    def prepare_data(self, srcs, masks, pos_embeds):
        """
        Args:
            srcs: List[torch.Tensor(B, C, H, W)]. B is also the length of frames.
            masks:
            pos_embeds:

        Returns:

        """
        # prepare input for encoder
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long,
                                         device=src_flatten.device)

        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        return src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index, valid_ratios


class TemporalMSTransformerEncoderLayer(DeformableTransformerEncoderLayer):
    def __init__(
            self,
            d_model=256,
            d_ffn=1024,
            dropout=0.1,
            activation='relu',
            t_window=2,
            n_levels=4,
            n_heads=8,
            n_curr_points=4,
            n_temporal_points=2
    ):
        super(TemporalMSTransformerEncoderLayer, self).__init__(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=4
        )
        self.self_attn = TemporalMSDeformAttnEncoder(d_model, n_levels, t_window, n_heads, n_curr_points, n_temporal_points)


class TemporalMSTransformerEncoder(DeformableTransformerEncoder):
    def __init__(
            self,
            encoder_layer,
            num_layers,
            t_window,
            enc_connect_all_embeddings
    ):
        super(TemporalMSTransformerEncoder, self).__init__(encoder_layer=encoder_layer, num_layers=num_layers)
        self.t_window = t_window
        self.enc_connect_all_embeddings = enc_connect_all_embeddings

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        T_ = src.shape[0]
        temporal_offsets = []
        if self.enc_connect_all_embeddings:
            temporal_spatial_shapes = spatial_shapes.repeat(T_ - 1, 1)
            for curr_frame in range(0, T_):
                frame_offsets = torch.tensor([t for t in range(-curr_frame, T_ - curr_frame) if t != 0],
                                             device=src.device)
                temporal_offsets.append(frame_offsets)
        else:
            temporal_spatial_shapes = spatial_shapes.repeat(self.t_window, 1)
            temporal_frames = [t for t in range(-self.t_window // 2, (self.t_window // 2) + 1) if t != 0]
            for cur_frame in range(0, T_):
                frame_offsets = []
                for t_frame in temporal_frames:
                    if cur_frame + t_frame < 0 or cur_frame + t_frame > T_ - 1:
                        frame_offsets.append(-t_frame)
                    else:
                        frame_offsets.append(t_frame)
                temporal_offsets.append(torch.tensor(frame_offsets, device=src.device))

        kwargs = {'temporal_offsets': temporal_offsets}

        temporal_level_start_index = torch.cat((temporal_spatial_shapes.new_zeros((1, )),
                                                temporal_spatial_shapes.prod(1).cumsum(0)[:-1]))

        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, (spatial_shapes, temporal_spatial_shapes),
                           (level_start_index, temporal_level_start_index), **kwargs)

        return output


class TemporalMSTransformerDecoderLayer(DeformableTransformerDecoderLayer):
    def __init__(
            self,
            d_model=256,
            d_ffn=1024,
            dropout=0.1,
            activation="relu",
            t_window=2,
            dec_instance_aware_att=True,
            n_levels=4,
            n_heads=8,
            n_curr_points=4,
            n_temporal_points=2
    ):
        super(TemporalMSTransformerDecoderLayer, self).__init__(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            n_levels=n_levels,
            n_heads=8,
            n_points=None
        )
        self.cross_attn = TemporalMSDeformAttnDecoder(
            d_model=d_model,
            n_levels=n_levels,
            t_window=t_window,
            n_heads=n_heads,
            n_curr_points=n_curr_points,
            n_temporal_points=n_temporal_points,
            dec_instance_aware_att=dec_instance_aware_att,
        )


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, with_gradient=True):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.with_gradient = with_gradient
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.ref_point_embed = None

    def refine_reference_point(self, lid, output, reference_points, intermediate,
                               intermediate_reference_points):
        # hack implementation for iterative bounding box refinement
        if self.bbox_embed is not None:
            tmp = self.bbox_embed[lid](output)
            if reference_points.shape[-1] == 4:
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
            else:
                assert reference_points.shape[-1] == 2
                new_reference_points = tmp
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()

            if self.with_gradient:
                reference_points = new_reference_points
            else:
                reference_points = new_reference_points.detach()

        if self.ref_point_embed is not None:
            tmp = self.ref_point_embed[lid](output)
            new_reference_points = tmp + inverse_sigmoid(reference_points)
            reference_points = new_reference_points.sigmoid()

        intermediate.append(output)
        intermediate_reference_points.append(reference_points)

        return reference_points, intermediate, intermediate_reference_points

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index,
                src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        kwargs = {
            'input_padding_mask': src_padding_mask
        }

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:,
                                           None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes,
                           src_level_start_index, **kwargs)

            reference_points, intermediate, intermediate_reference_points = self.refine_reference_point(
                lid, output, reference_points, intermediate, intermediate_reference_points)

        return torch.stack(intermediate), torch.stack(intermediate_reference_points)


class TemporalMSTransformerDecoder(DeformableTransformerDecoder):
    def __init__(
            self,
            decoder_layer,
            num_layers,
            with_gradient=False,
            instance_aware_att=True,
    ):
        super(TemporalMSTransformerDecoder, self).__init__(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            with_gradient=with_gradient
        )
        self.instance_aware_att = instance_aware_att

    def forward(
            self,
            tgt,
            reference_points,
            src,
            src_spatial_shapes,
            src_level_start_index,
            src_valid_ratios,
            query_pos=None,
            src_padding_mask=None
    ):
        output = tgt
        intermediate = []
        intermediate_reference_points = []

        T_ = src.shape[0]
        temporal_offsets = []
        for curr_frame in range(0, T_):
            offsets = torch.tensor([t for t in range(-curr_frame, T_ - curr_frame) if t != 0], device=src.device)
            temporal_offsets.append(offsets)

        src_temporal_spatial_shapes = src_spatial_shapes.repeat(T_ - 1, 1)
        src_temporal_level_start_index = torch.cat((src_temporal_spatial_shapes.new_zeros((1, )),
                                                    src_temporal_spatial_shapes.prod(1).cumsum(0)[:-1]))
        kwargs = {'temporal_offsets': temporal_offsets}

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios[0, None], src_valid_ratios[0, None]], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[0, None]

            output = layer(output, query_pos, reference_points_input, src,
                           (src_spatial_shapes, src_temporal_spatial_shapes),
                           (src_level_start_index, src_temporal_level_start_index),
                           **kwargs)

            reference_points, intermediate, intermediate_reference_points = self.refine_reference_point(
                lid, output, reference_points, intermediate, intermediate_reference_points
            )

        return torch.stack(intermediate), torch.stack(intermediate_reference_points)


def build_temporal_transformer(args):
    return TemporalTransformer(
        d_model=args.hidden_dim,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_feature_levels=args.num_feature_levels,
        with_gradient=args.bbox_gradient_propa,

        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        nhead=args.nheads,
        enc_n_curr_points=args.enc_n_points,
        dec_n_curr_points=args.dec_n_points,

        num_frames=args.num_frames,
        enc_connect_all_embeddings=args.enc_connect_all_frames,
        enc_temporal_window=args.enc_temporal_window,
        enc_n_temporal_points=args.enc_n_points_temporal_frame,
        dec_instance_aware_att=args.instance_aware_attention,
        dec_n_temporal_points=args.dec_n_points_temporal_frame,

        activation="relu",

    )
