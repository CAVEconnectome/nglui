from neuroglancer import (
    LineAnnotation,
    PointAnnotation,
    EllipsoidAnnotation,
    AxisAlignedBoundingBoxAnnotation,
)


def convert_annotation(anno):
    if anno.type == "point":
        out_anno = convert_point_annotation(anno)
    elif anno.type == "line":
        out_anno = convert_line_annotation(anno)
    elif anno.type == "axis_aligned_bounding_box":
        out_anno = convert_bbox_annotation(anno)
    elif anno.type == "ellipsoid":
        out_anno = convert_sphere_annotation(anno)
    else:
        out_anno = None
    return out_anno


def convert_point_annotation(anno):
    out_anno = PointAnnotation(
        point=anno.point.tolist(),
        id=anno.id,
        description=anno.description,
    )
    if len(anno.segments) > 0:
        out_anno.segments = [[int(x) for x in anno.segments]]
    return out_anno


def convert_line_annotation(anno):
    out_anno = LineAnnotation(
        point_a=anno.point_a.tolist(),
        point_b=anno.point_b.tolist(),
        id=anno.id,
        description=anno.description,
    )
    if len(anno.segments) > 0:
        out_anno.segments = [[int(x) for x in anno.segments]]
    return out_anno


def convert_bbox_annotation(anno):
    # AxisAlignedBoundingBox happens to use the same parameters as LineAnnotation
    out_anno = AxisAlignedBoundingBoxAnnotation(
        point_a=anno.point_a.tolist(),
        point_b=anno.point_b.tolist(),
        id=anno.id,
        description=anno.description,
    )
    if len(anno.segments) > 0:
        out_anno.segments = [[int(x) for x in anno.segments]]
    return out_anno


def convert_sphere_annotation(anno):
    out_anno = EllipsoidAnnotation(
        center=anno.center.tolist(),
        radii=anno.radii.tolist(),
        id=anno.id,
        description=anno.description,
    )
    if len(anno.segments) > 0:
        out_anno.segments = [[int(x) for x in anno.segments]]
    return out_anno
