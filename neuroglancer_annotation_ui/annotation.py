import neuroglancer
import secrets


def line_annotation(a, b, id=None, description=None):
    """Returns line annotation object.

    Attributes:
        a (list): x,y,z position of first point
        b (list): x,y,z position of second point
        description (str) : additional description of specific annotation.
    """
    if id is None:
        id = neuroglancer.random_token.make_random_token()
    line = neuroglancer.LineAnnotation(
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
        id = neuroglancer.random_token.make_random_token()
    point = neuroglancer.PointAnnotation(
        point=point,
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
        id = neuroglancer.random_token.make_random_token()
    ellipsoid = neuroglancer.EllipsoidAnnotation(
        center=center,
        radii=radii,
        id=id,
        description=description)
    return ellipsoid


def bounding_box_annotaiton(a, b, id=None, description=None):
    """returns axis aligned bounding box annotation object.

    Attributes:
        a (list): x,y,z position of first point
        b (list): x,y,z position of second point
        description (str) : additional description of specific annotation.
    """
    if id is None:
        id = neuroglancer.random_token.make_random_token()
    bounding_box = neuroglancer.AxisAlignedBoundingBoxAnnotation(
        point_a=point_a,
        point_b=point_b,
        id=id,
        description=description)
    return bounding_box


def generate_id(nbytes=32):
    """ Helper method to make random token hex byte string containing
    nbytes number of bytes.

    Attributes:
        nbytes (int):  length of byte string containing nbytes.
    """
    id = secrets.token_hex(nbytes)
    return id


if __name__ == '__main__':
    point_a = [150., 40., 65.]
    point_b = [147., 38., 65.]
    center = [38610.77, 29166.29, 512]
    radii = [1900, 1900, 190]
    des = 'This is a description string'
    line = line_annotation(point_a, point_a, generate_id(), des)
    point = point_annotation(point_a, generate_id(), des)
    ellipsoid = ellipsoid_annotation(center=center, radii=radii, id=generate_id())
    print(line, point, ellipsoid)
