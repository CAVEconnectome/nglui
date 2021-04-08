from nglui.nglite import (
    LineAnnotation,
    PointAnnotation,
    EllipsoidAnnotation,
    AxisAlignedBoundingBoxAnnotation,
    CollectionAnnotation,
    random_token,
)
from .utils import omit_nones
from numpy import unique, concatenate


def line_annotation(
    a, b, id=None, description=None, linked_segmentation=None, tag_ids=None
):
    """Returns line annotation object.

    Attributes:
        a (list): x,y,z position of first point
        b (list): x,y,z position of second point
        description (str) : additional description of specific annotation.
    """
    if id is None:
        id = random_token.make_random_token()
    line = LineAnnotation(
        point_a=a,
        point_b=b,
        id=id,
        description=description,
        segments=omit_nones(linked_segmentation),
    )
    if tag_ids is not None:
        line._json_data["tagIds"] = omit_nones(tag_ids)
    return line


def point_annotation(
    point, id=None, description=None, linked_segmentation=None, tag_ids=None
):
    """Returns point annotation object

    Attributes:
        a (list): x,y,z position of point
        description (str) : additional description of specific annotation.
    """
    if id is None:
        id = random_token.make_random_token()
    point = PointAnnotation(
        point=[int(x) for x in point],
        id=id,
        description=description,
        segments=omit_nones(linked_segmentation),
    )
    if tag_ids is not None:
        point._json_data["tagIds"] = omit_nones(tag_ids)
    return point


def sphere_annotation(
    center,
    radius,
    z_multiplier,
    id=None,
    description=None,
    linked_segmentation=None,
    tag_ids=None,
):
    """
    Assumes the z-axis is anistropic
    """
    unit_v = [1, 1, z_multiplier]
    radii = [radius * x for x in unit_v]
    return ellipsoid_annotation(
        center,
        radii,
        id=id,
        description=description,
        linked_segmentation=linked_segmentation,
        tag_ids=tag_ids,
    )


def ellipsoid_annotation(
    center, radii, id=None, description=None, linked_segmentation=None, tag_ids=None
):
    """returns ellipsoid annotation object.

    Attributes:
        center (list): point position of centroid of ellipsoid
        radii (float): TODO
        description (str) : additional description of specific annotation.
    """
    if id is None:
        id = random_token.make_random_token()
    ellipsoid = EllipsoidAnnotation(
        center=center,
        radii=radii,
        id=id,
        description=description,
        segments=omit_nones(linked_segmentation),
    )
    if tag_ids is not None:
        ellipsoid._json_data["tag_ids"] = omit_nones(tag_ids)
    return ellipsoid


def bounding_box_annotation(
    point_a, point_b, id=None, description=None, linked_segmentation=None, tag_ids=None
):
    """returns axis aligned bounding box annotation object.

    Attributes:
        a (list): x,y,z position of first point
        b (list): x,y,z position of second point
        description (str) : additional description of specific annotation.
    """
    if id is None:
        id = random_token.make_random_token()
    bounding_box = AxisAlignedBoundingBoxAnnotation(
        point_a=point_a,
        point_b=point_b,
        id=id,
        description=description,
        segments=omit_nones(linked_segmentation),
    )
    if tag_ids is not None:
        bounding_box._json_data["tag_ids"] = omit_nones(tag_ids)
    return bounding_box


def group_annotations(
    annotations,
    source=None,
    id=None,
    return_all=True,
    gather_linked_segmentations=True,
    share_linked_segmentations=False,
):
    if len(annotations) == 0:
        return []
    if id is None:
        id = random_token.make_random_token()
    if source is None:
        if annotations[0].type == "point":
            source = annotations[0].point
        elif (
            annotations[0].type == "line"
            or annotations[0].type == "axis_aligned_bounding_box"
        ):
            source = annotations[0].point_a
        elif annotations[0].type == "ellipsoid":
            source = annotations[0].center
    entries = [anno.id for anno in annotations]

    seg_in_group = [anno.segments for anno in annotations]
    seg_in_group = unique(concatenate(seg_in_group))

    if gather_linked_segmentations:
        linked_segs = seg_in_group
    else:
        linked_segs = []

    collection_anno = CollectionAnnotation(
        source=source,
        entries=entries,
        id=id,
        segments=omit_nones(seg_in_group),
        children_visible=True,
    )

    for anno in annotations:
        anno.parent_id = collection_anno.id
        if share_linked_segmentations:
            anno.segments = seg_in_group
    if return_all:
        annotations.append(collection_anno)
        return annotations
    else:
        return collection_anno
