from .cva_net import build as build_cva_net
from .temporal_ms_deform_detr import build as build_temporal_ms_deform_detr


def build_model(args):
    return build_temporal_ms_deform_detr(args)
