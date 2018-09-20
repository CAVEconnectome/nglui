from neuroglancer import LineAnnotation, \
                         PointAnnotation, \
                         EllipsoidAnnotation, \
                         AxisAlignedBoundingBoxAnnotation, \
                         random_token

def line_annotation(a, b, id=None, description=None):
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
        description=description)
    return line


def point_annotation(point, id=None, description=None):
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
        description=description)
    return point


def ellipsoid_annotation(center, radii, id=None, description=None):
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
        description=description)
    return ellipsoid


def bounding_box_annotation(a, b, id=None, description=None):
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
        description=description)
    return bounding_box