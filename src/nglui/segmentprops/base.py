import attrs
import numpy as np
from typing import Optional, Union
from functools import partial
from itertools import chain

"""
Options and validation for neuroglancer segment properties based on
https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/segment_properties.md
"""


class SegmentPropertyBase(object):
    pass


ALLOWED_NUMBER_DATA_TYPES = [
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "float32",
]
NONE_ATTR_FILTERS = ["tag_descriptions", "description"]


def list_of_strings(x: list) -> list:
    return [str(v) for v in x]


def space_to_underscore(x: list) -> list:
    return [str(y).replace(" ", "_") for y in x]


def sort_tag_arrays(x: list) -> list:
    return [sorted(y) for y in x]


@attrs.define
class InlineProperties:
    ids = attrs.field(type=list[int], converter=list_of_strings, kw_only=True)
    properties = attrs.field(type=list[SegmentPropertyBase], default=[], kw_only=True)

    def __len__(self):
        return len(self.ids)

    def __attrs_post_init__(self):
        if len(self.properties) > 0:
            _validate_properties(self.ids, self.properties)


@attrs.define
class LabelProperty(SegmentPropertyBase):
    id = attrs.field(init=False, default="label", type=str)
    type = attrs.field(init=False, default="label", type=str)
    values = attrs.field(type=list, converter=list_of_strings, kw_only=True)
    description = attrs.field(type=Optional[str], default=None, kw_only=True)

    def __len__(self):
        return len(self.values)


@attrs.define
class DescriptionProperty(SegmentPropertyBase):
    id = attrs.field(init=False, default="description", type=str)
    type = attrs.field(init=False, default="description", type=str)
    values = attrs.field(type=list, converter=list_of_strings, kw_only=True)
    description = attrs.field(type=Optional[str], default=None, kw_only=True)

    def __len__(self):
        return len(self.values)


@attrs.define
class StringProperty(SegmentPropertyBase):
    id = attrs.field(type=str, kw_only=True)
    type = attrs.field(init=False, default="string", type=str)
    values = attrs.field(type=list, converter=list_of_strings, kw_only=True)
    description = attrs.field(type=Optional[str], default=None, kw_only=True)

    def __len__(self):
        return len(self.values)


@attrs.define
class NumberProperty(SegmentPropertyBase):
    id = attrs.field(type=str, kw_only=True)
    type = attrs.field(init=False, default="number", type=str)
    values = attrs.field(type=list, kw_only=True)
    description = attrs.field(type=Optional[str], default=None, kw_only=True)
    data_type = attrs.field(
        type=str,
        kw_only=True,
        validator=attrs.validators.in_(ALLOWED_NUMBER_DATA_TYPES),
    )

    def __attrs_post_init__(self):
        prop_dtype = np.dtype(self.data_type)
        self.values = (
            np.array(self.values).astype(prop_dtype, casting="same_kind").tolist()
        )

    def __len__(self):
        return len(self.values)


@attrs.define
class TagProperty(SegmentPropertyBase):
    id = attrs.field(type=str, kw_only=True)
    type = attrs.field(init=False, type=str, default="tags")
    tags = attrs.field(
        type=list[str],
        converter=space_to_underscore,
        validator=attrs.validators.not_(attrs.validators.matches_re(r"^\#")),
        kw_only=True,
    )
    tag_descriptions = attrs.field(
        type=list[str],
        default=None,
        kw_only=True,
    )
    values = attrs.field(
        type=list[list[int]],
        converter=sort_tag_arrays,
        kw_only=True,
    )

    def __len__(self):
        return len(self.values)


def prop_filter(attr, value):
    "Filters out None for optional attributes for use in'attrs.asdict' conversion"
    return value is not None or attr.name not in NONE_ATTR_FILTERS


prop_to_dict = partial(attrs.asdict, filter=prop_filter)


def _validate_properties(
    ids,
    properties,
):
    n_ids = len(ids)
    for prop in properties:
        if len(prop) != n_ids:
            msg = f"Property {prop} does not have the same number of entries as the id list."
            raise ValueError(msg)


def build_segment_properties(
    ids: list[int],
    properties: list,
):
    return {
        "@type": "neuroglancer_segment_properties",
        "inline": prop_to_dict(InlineProperties(ids=ids, properties=properties)),
    }


def _find_column_dtype(column):
    "Get data type string from a dataframe column"
    if column.dtype in ALLOWED_NUMBER_DATA_TYPES:
        return str(column.dtype)
    elif column.dtype == "int64":
        return "int32"
    elif column.dtype == "float64":
        return "float32"
    elif column.dtype == "object":
        try:
            column.astype("float32")
        except Exception as e:
            raise ValueError(
                f"Column {column} has an unsupported data type {column.dtype}"
            )
    else:
        raise ValueError(f"Column {column} has an unsupported data type {column.dtype}")


def _tag_descriptions(tags, tag_descriptions):
    if tag_descriptions is None:
        return None
    else:
        return [tag_descriptions.get(tag, tag) for tag in tags]


def _tag_property_from_columns(df, cols, tag_descriptions=None, name="tags"):
    tags = []
    for col in cols:
        unique_tags = df[col].unique()
        # df.unique works differently for categorical dtype columns and does not return an ndarray so we have to check
        if isinstance(unique_tags, np.ndarray):
            unique_tags = sorted(unique_tags.tolist())
        else:
            unique_tags = unique_tags.sort_values().tolist()
        if np.any(np.isin(tags, unique_tags)):
            raise ValueError("Tags across columns are not unique")
        tags.extend(unique_tags)
    tag_map = {tag: i for i, tag in enumerate(tags) if tag is not None}
    tag_values = []
    for _, row in df.iterrows():
        tag_values.append([tag_map[tag] for tag in row[cols] if tag is not None])
    return TagProperty(
        id=name,
        tags=tags,
        values=tag_values,
        tag_descriptions=_tag_descriptions(tags, tag_descriptions),
    )


def _tag_property_from_bool_cols(df, col_list, tag_descriptions=None, name="tags"):
    tags = col_list
    tag_map = {tag: i for i, tag in enumerate(tags)}
    tag_values = [[] for _ in range(len(df))]
    for tv in tag_values:
        for loc in np.flatnonzero(df[tv]):
            tag_values[loc].append(tag_map[tv])
    return TagProperty(
        id=name,
        tags=tags,
        values=tag_values,
        tag_descriptions=_tag_descriptions(tags, tag_descriptions),
    )


class SegmentProperties:
    def __init__(
        self,
        ids: list[int],
        label_property: Optional[LabelProperty] = None,
        description_property: Optional[DescriptionProperty] = None,
        tag_properties: Optional[TagProperty] = None,
        string_properties: Optional[Union[StringProperty, list[StringProperty]]] = None,
        number_properties: Optional[Union[NumberProperty, list[NumberProperty]]] = None,
    ):
        self.ids = ids
        self.label_property = label_property
        self.description_property = description_property
        self.tag_properties = tag_properties

        if isinstance(string_properties, StringProperty):
            string_properties = [string_properties]
        self.string_properties = string_properties

        if isinstance(number_properties, NumberProperty):
            number_properties = [number_properties]
        self.number_properties = number_properties

    def __len__(self):
        return len(self.ids)

    def _property_list(self):
        single_prop_list = [
            prop
            for prop in [
                self.label_property,
                self.description_property,
                self.tag_properties,
            ]
            if prop is not None
        ]
        multi_prop_list = list(
            chain.from_iterable(
                [
                    prop
                    for prop in [
                        self.string_properties,
                        self.number_properties,
                    ]
                    if prop is not None
                ]
            )
        )
        return single_prop_list + multi_prop_list

    def to_dict(self):
        "Converts the segment properties to a dictionary for use in neuroglancer"
        return build_segment_properties(
            self.ids,
            self._property_list(),
        )

    @classmethod
    def from_dataframe(
        cls,
        df,
        id_col: str = "pt_root_id",
        label_col: Optional[str] = None,
        description_col: Optional[str] = None,
        string_cols: Optional[list[str]] = None,
        number_cols: Optional[list[str]] = None,
        tag_value_cols: Optional[list[str]] = None,
        tag_bool_cols: Optional[list[list[str]]] = None,
        tag_descriptions: Optional[dict] = None,
    ):
        ids = df[id_col].tolist()
        properties = {}
        if label_col:
            properties["label_property"] = LabelProperty(values=df[label_col].tolist())
        if description_col:
            properties["description_property"] = DescriptionProperty(
                values=df[description_col].tolist()
            )
        if string_cols:
            properties["string_properties"] = [
                StringProperty(id=col, values=df[col].tolist()) for col in string_cols
            ]
        if number_cols:
            properties["number_properties"] = [
                NumberProperty(
                    id=col,
                    values=df[col].tolist(),
                    data_type=_find_column_dtype(df[col]),
                )
                for col in number_cols
            ]
        if tag_value_cols:
            if isinstance(tag_value_cols, str):
                tag_value_cols = [tag_value_cols]
            properties["tag_properties"] = _tag_property_from_columns(
                df,
                tag_value_cols,
                tag_descriptions,
            )
        elif tag_bool_cols:
            if "tag_properties" in properties:
                raise ValueError("Cannot set both tag_value_cols and tag_bool_cols")
            properties["tag_properties"] = _tag_property_from_bool_cols(
                df,
                tag_bool_cols,
                tag_descriptions,
            )
        return cls(ids, **properties)
