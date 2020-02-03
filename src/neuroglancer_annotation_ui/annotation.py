from neuroglancer_annotation_ui.nglite import LineAnnotation, \
                                              PointAnnotation, \
                                              EllipsoidAnnotation, \
                                              AxisAlignedBoundingBoxAnnotation, \
                                              random_token

def line_annotation(a, b, id=None, description=None, linked_segmentation=None, tag_ids=None):
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
        segments=linked_segmentation)
    if tag_ids is not None:
        line._json_data['tagIds'] = tag_ids
    return line


def point_annotation(point, id=None, description=None, linked_segmentation=None, tag_ids=None):
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
        segments=linked_segmentation)
    if tag_ids is not None:
        point._json_data['tagIds'] = tag_ids
    return point

def sphere_annotation(center, radius, z_multiplier, id=None, description=None, linked_segmentation=None, tag_ids=None):
    """
    Assumes the z-axis is anistropic
    """
    unit_v = [1, 1, z_multiplier]
    radii = [radius * x for x in unit_v]
    return ellipsoid_annotation(center, radii, id=id, description=description,
                                linked_segmentation=linked_segmentation, tag_ids=tag_ids)

def ellipsoid_annotation(center, radii, id=None, description=None, linked_segmentation=None, tag_ids=None):
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
        segments=linked_segmentation)
    if tag_ids is not None:
        ellipsoid._json_data['tag_ids'] = tag_ids
    return ellipsoid

def bounding_box_annotation(point_a, point_b, id=None, description=None, linked_segmentation=None, tag_ids=None):
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
        segments=linked_segmentation)
    if tag_ids is not None:
        bounding_box._json_data['tag_ids'] = tag_ids
    return bounding_box