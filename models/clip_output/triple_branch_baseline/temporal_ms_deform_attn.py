import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

import math
import warnings
from models.ops.modules.ms_deform_attn import MSDeformAttnFunction, _is_power_of_2


class TemporalMSDeformAttnBase(nn.Module):
    def __init__(
            self,
            d_model=256,
            n_levels=4,
            t_window=2,
            n_heads=8,
            n_curr_points=4,
            n_temporal_points=2
    ):
        super(TemporalMSDeformAttnBase, self).__init__()
        """
        Multi-Scale Deformable Attention Module
        :param d_model          hidden dimension
        :param n_levels         number of feature levels
        :param n_heads          number of attention heads
        :param n_curr_points    number of sampling points per attention head per feature level from
                                each query corresponding frame
        :param n_temporal_points    number of sampling points per attention head per feature level
                                    from temporal frames
        """
        if d_model % n_heads != 0:
            raise ValueError(
                'd_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.t_window = t_window
        self.n_heads = n_heads
        self.n_curr_points = n_curr_points
        self.n_temporal_points = n_temporal_points

        # Used for sampling and attention in the current frame
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_curr_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_curr_points)

        # Used for sampling and attention in the prev or post frames
        self.temporal_sampling_offsets = nn.Linear(d_model, n_heads * n_levels * t_window * n_temporal_points * 2)
        self.temporal_attention_weights = nn.Linear(d_model, n_heads * n_levels * t_window * n_temporal_points)

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # sampling offset initialized weight to 0, so at initial iterations the bias is the only that matters at all
        constant_(self.temporal_sampling_offsets.weight.data, 0.)

        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)

        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])

        # curr_frame init
        curr_grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels,
                                                                      self.n_curr_points, 1)
        for i in range(self.n_curr_points):
            curr_grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(curr_grid_init.reshape(-1))

        # temporal init
        temporal_grid_init = grid_init.view(self.n_heads, 1, 1, 1, 2).repeat(1, self.n_levels,
                                                                             self.t_window,
                                                                             self.n_temporal_points,
                                                                             1)

        for i in range(self.n_temporal_points):
            temporal_grid_init[:, :, :, i, :] *= i + 1

        with torch.no_grad():
            self.temporal_sampling_offsets.bias = nn.Parameter(temporal_grid_init.reshape(-1))

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        constant_(self.temporal_attention_weights.weight.data, 0.)
        constant_(self.temporal_attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(
            self,
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask
    ):
        """
        Args:
            query:
            reference_points:
            input_flatten:
            input_spatial_shapes:
            input_level_start_index:
            input_padding_mask:

        Returns:

        """
        raise NotImplementedError

    # Computes current/temporal sampling offsets and attention weights,
    # which are treated different for the encoder and decoder later on
    def _compute_deformable_attention(self, query, input_flatten):
        T_q, Len_q, _ = query.shape
        T_, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        value = value.view(T_, Len_in, self.n_heads, self.d_model // self.n_heads)

        temporal_sampling_offsets = self.temporal_sampling_offsets(query).view(T_q,
                                                                               Len_q,
                                                                               self.n_heads,
                                                                               self.t_window,
                                                                               self.n_levels,
                                                                               self.n_temporal_points,
                                                                               2)
        temporal_sampling_offsets = temporal_sampling_offsets.flatten(3, 4)   # (T_, L, N_H, t*N_L, N_P, 2)
        temporal_attention_weights = self.temporal_attention_weights(query)
        temporal_attention_weights = temporal_attention_weights.view(T_q,
                                                                     Len_q,
                                                                     self.n_heads,
                                                                     self.t_window * self.n_levels * self.n_temporal_points
                                                                     )
        curr_frame_attention_weights = self.attention_weights(query).view(T_q,
                                                                          Len_q,
                                                                          self.n_heads,
                                                                          self.n_levels * self.n_curr_points
                                                                          )
        attention_weights_curr_temporal = torch.cat(
            [curr_frame_attention_weights, temporal_attention_weights], dim=3)
        attention_weights_curr_temporal = F.softmax(attention_weights_curr_temporal, -1)
        attention_weights_curr = attention_weights_curr_temporal[:, :, :, :self.n_levels * self.n_curr_points]
        attention_weights_temporal = attention_weights_curr_temporal[:, :, :, self.n_levels * self.n_curr_points:]
        attention_weights_curr = attention_weights_curr.view(T_q,
                                                             Len_q,
                                                             self.n_heads,
                                                             self.n_levels,
                                                             self.n_curr_points
                                                             ).contiguous()
        attention_weights_temporal = attention_weights_temporal.view(T_q,
                                                                     Len_q,
                                                                     self.n_heads,
                                                                     self.t_window * self.n_levels,
                                                                     self.n_temporal_points
                                                                     ).contiguous()

        curr_frame_sampling_offsets = self.sampling_offsets(query).view(T_q,
                                                                        Len_q,
                                                                        self.n_heads,
                                                                        self.n_levels,
                                                                        self.n_curr_points,
                                                                        2)

        return value, curr_frame_sampling_offsets, temporal_sampling_offsets, attention_weights_curr, attention_weights_temporal


class TemporalMSDeformAttnEncoder(TemporalMSDeformAttnBase):
    def forward(
            self,
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            temporal_offsets
    ):
        output = []
        input_current_spatial_shapes, input_temporal_spatial_shapes = input_spatial_shapes
        input_current_level_start_index, input_temporal_level_start_index = input_level_start_index
        T_, Len_q, _ = query.shape
        T_, Len_in, _ = input_flatten.shape
        assert reference_points.shape[-1] == 2

        (
            value,
            curr_frame_sampling_offsets,
            temporal_sampling_offsets,
            attention_weights_curr,
            attention_weights_temporal
        ) = super()._compute_deformable_attention(query, input_flatten)

        offset_normalizer = torch.stack(
            [input_current_spatial_shapes[..., 1], input_current_spatial_shapes[..., 0]], -1)
        temporal_offsets_normalizer = offset_normalizer.repeat(self.t_window, 1)

        for t in range(T_):
            current_frame_values = value[t][None]
            sampling_locations = reference_points[t][None, :, None, :, None] \
                                 + curr_frame_sampling_offsets[t][None] / offset_normalizer[None,
                                                                          None, None, :, None, :]

            output_curr = MSDeformAttnFunction.apply(
                current_frame_values, input_current_spatial_shapes, input_current_level_start_index,
                sampling_locations, attention_weights_curr[t][None], self.im2col_step)

            temporal_frames = temporal_offsets[t] + t
            temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]
            temporal_ref_points = reference_points[t, :, 0][None, :, None, None, None]

            temporal_sampling_locations = temporal_ref_points \
                                          + temporal_sampling_offsets[t][
                                              None] / temporal_offsets_normalizer[None, None, None,
                                                      :, None, :]

            output_temporal = MSDeformAttnFunction.apply(
                temporal_frames_values, input_temporal_spatial_shapes,
                input_temporal_level_start_index, temporal_sampling_locations,
                attention_weights_temporal[t][None], self.im2col_step)

            frame_output = output_curr + output_temporal
            output.append(frame_output)

        output = torch.cat(output, dim=0)
        output = self.output_proj(output)

        return output


class TemporalMSDeformAttnDecoder(TemporalMSDeformAttnBase):
    def __init__(
            self,
            d_model=256,
            n_levels=4,
            t_window=2,
            n_heads=8,
            n_curr_points=4,
            n_temporal_points=2,
            dec_instance_aware_att=True
    ):
        super(TemporalMSDeformAttnDecoder, self).__init__(
            d_model=d_model,
            n_levels=n_levels,
            t_window=t_window,
            n_heads=n_heads,
            n_curr_points=n_curr_points,
            n_temporal_points=n_temporal_points)
        self.dec_instance_aware_att = dec_instance_aware_att

    def forward(
            self,
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            temporal_offsets
    ):
        output = []
        input_current_spatial_shapes, input_temporal_spatial_shapes = input_spatial_shapes
        input_current_level_start_index, input_temporal_level_start_index = input_level_start_index

        T_q, Len_q, _ = query.shape
        T_, Len_in, _ = input_flatten.shape

        (
            value,
            curr_frame_sampling_offsets,
            temporal_sampling_offsets,
            attention_weights_curr,
            attention_weights_temporal
        ) = super()._compute_deformable_attention(query, input_flatten)

        # To add hook for att maps visualization
        current_sampling_locations_for_att_maps, temporal_sampling_locations_for_att_maps = [], []
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_current_spatial_shapes[..., 1], input_current_spatial_shapes[..., 0]], -1)
            temporal_offsets_normalizer = offset_normalizer.repeat(self.t_window, 1)

            for t in range(T_q):
                current_frame_values = value[t][None]
                sampling_locations = reference_points[t][None, :, None, :, None] \
                                     + curr_frame_sampling_offsets[t][None] / offset_normalizer[
                                                                              None, None, None, :,
                                                                              None, :]

                current_sampling_locations_for_att_maps.append(sampling_locations)
                output_curr = MSDeformAttnFunction.apply(
                    current_frame_values, input_current_spatial_shapes,
                    input_current_level_start_index,
                    sampling_locations, attention_weights_curr[t][None], self.im2col_step
                )

                temporal_frames = temporal_offsets[t] + t
                temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]

                if self.dec_instance_aware_att:
                    temporal_ref_points = reference_points[temporal_frames].transpose(0, 1).flatten(
                        1, 2)[None, :, None, :, None]
                else:
                    temporal_ref_points = reference_points[t].repeat(1, self.t_window, 1)[None, :,
                                          None, :, None]

                temporal_sampling_locations = temporal_ref_points \
                                              + temporal_sampling_offsets[t][
                                                  None] / temporal_offsets_normalizer[None, None,
                                                          None, :, None, :]

                temporal_sampling_locations_for_att_maps.append(temporal_sampling_locations)

                # In order to avoid a for loop that computes the attention for each temporal
                # frame, we STACK them all on the resolution level axis.
                output_temporal = MSDeformAttnFunction.apply(
                    temporal_frames_values, input_temporal_spatial_shapes,
                    input_temporal_level_start_index, temporal_sampling_locations,
                    attention_weights_temporal[t][None], self.im2col_step)

                frame_output = output_curr + output_temporal
                output.append(frame_output)

        elif reference_points.shape[-1] == 4:
            for t in range(T_q):
                current_frame_values = value[t][None]
                sampling_locations = reference_points[t][None, :, None, :, None, :2] \
                                     + (curr_frame_sampling_offsets[t][None] / self.n_curr_points) * \
                                     reference_points[t][None, :, None, :, None, 2:] * 0.5

                current_sampling_locations_for_att_maps.append(sampling_locations)
                output_curr = MSDeformAttnFunction.apply(
                    current_frame_values, input_current_spatial_shapes,
                    input_current_level_start_index, sampling_locations,
                    attention_weights_curr[t][None], self.im2col_step
                )

                temporal_frames = temporal_offsets[t] + t
                temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]

                if self.dec_instance_aware_att:
                    temporal_ref_points = reference_points[temporal_frames].transpose(0, 1).flatten(
                        1, 2)[None, :, None, :, None]
                else:
                    temporal_ref_points = reference_points[t].repeat(1, self.t_window, 1)[None, :,
                                          None, :, None]

                temporal_sampling_locations = temporal_ref_points[:, :, :, :, :, :2] \
                                              + (temporal_sampling_offsets[t][
                                                     None] / self.n_temporal_points) * temporal_ref_points[
                                                                                       :, :, :, :,
                                                                                       :, 2:] * 0.5

                temporal_sampling_locations_for_att_maps.append(temporal_sampling_locations)
                output_temporal = MSDeformAttnFunction.apply(
                    temporal_frames_values, input_temporal_spatial_shapes,
                    input_temporal_level_start_index, temporal_sampling_locations,
                    attention_weights_temporal[t][None], self.im2col_step
                )

                frame_output = output_curr + output_temporal
                output.append(frame_output)

        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(
                    reference_points.shape[-1]))

        output = torch.cat(output, dim=0)
        output = self.output_proj(output)

        return output
