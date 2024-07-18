from functools import partial
from itertools import chain
from typing import Optional, Union, List

import attrs
import numpy as np
import pandas as pd

"""
Options and validation for neuroglancer segment properties based on the segment properties spec:
https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/segment_properties.md
"""


class SegmentPropertyBase:
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


def zero_null_strings(x: list) -> list:
    return [y if not is_null_value(y) else "" for y in x]


def is_null_value(value):
    if value is None:
        return True
    elif pd.isna(value):
        return True
    elif value == "":
        return True
    else:
        return False


def preprocess_string_column(x: list) -> list:
    return space_to_underscore(zero_null_strings(x))


@attrs.define
class InlineProperties:
    ids = attrs.field(type=List[int], converter=list_of_strings, kw_only=True)
    properties = attrs.field(type=List[SegmentPropertyBase], default=[], kw_only=True)

    def __len__(self):
        return len(self.ids)

    def __attrs_post_init__(self):
        if len(self.properties) > 0:
            _validate_properties(self.ids, self.properties)


@attrs.define
class LabelProperty(SegmentPropertyBase):
    id = attrs.field(default="label", type=str, kw_only=True)
    type = attrs.field(init=False, default="label", type=str)
    values = attrs.field(type=list, converter=list_of_strings, kw_only=True)
    description = attrs.field(type=Optional[str], default=None, kw_only=True)

    def __len__(self):
        return len(self.values)


@attrs.define
class DescriptionProperty(SegmentPropertyBase):
    id = attrs.field(default="description", type=str, kw_only=True)
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
    id = attrs.field(type=str, kw_only=True, default="tags")
    type = attrs.field(init=False, type=str, default="tags")
    tags = attrs.field(
        type=List[str],
        converter=space_to_underscore,
        validator=attrs.validators.not_(attrs.validators.matches_re(r"^\#")),
        kw_only=True,
    )
    tag_descriptions = attrs.field(
        type=List[str],
        default=None,
        kw_only=True,
    )
    values = attrs.field(
        type=List[List[int]],
        converter=sort_tag_arrays,
        kw_only=True,
    )

    def __len__(self):
        return len(self.values)


def prop_filter(attr, value):
    "Filters out None for optional attributes for use in 'attrs.asdict' conversion"
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
    ids: List[int],
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
        except Exception:
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


def _make_tag_property(df, value_columns, bool_columns, tag_descriptions, name="tags"):
    if value_columns is None:
        value_columns = []
    if bool_columns is None:
        bool_columns = []
    tags = []
    for col in value_columns:
        unique_tags = df[col].unique()
        # df.unique works differently for categorical dtype columns and does not return an ndarray so we have to check
        if isinstance(unique_tags, np.ndarray):
            unique_tags = [x for x in unique_tags.tolist() if not is_null_value(x)]
            unique_tags = sorted(unique_tags)
        else:
            unique_tags = unique_tags.sort_values().tolist()
            unique_tags = [x for x in unique_tags if not is_null_value(x)]
        if np.any(np.isin(tags, unique_tags)):
            raise ValueError("Tags across columns are not unique")
        tags.extend(unique_tags)
    tags.extend(bool_columns)
    tag_map = {tag: i for i, tag in enumerate(tags) if not is_null_value(tag)}
    tag_values = []
    for _, row in df.iterrows():
        tag_values.append(
            [tag_map[tag] for tag in row[value_columns] if not is_null_value(tag)]
        )
    for tv in bool_columns:
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
        ids: List[int],
        label_property: Optional[LabelProperty] = None,
        description_property: Optional[DescriptionProperty] = None,
        tag_properties: Optional[TagProperty] = None,
        string_properties: Optional[Union[StringProperty, List[StringProperty]]] = None,
        number_properties: Optional[Union[NumberProperty, List[NumberProperty]]] = None,
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

    def __repr__(self):
        return f"SegmentProperties ({len(self.ids)} segments, {len(self._property_list())} properties)"

    def __str__(self):
        return self.__repr__()

    def property_description(self):
        return [(prop.id, prop.type, len(prop)) for prop in self._property_list()]

    def to_dict(self):
        "Converts the segment properties to a dictionary for use in neuroglancer"
        return build_segment_properties(
            self.ids,
            self._property_list(),
        )

    def to_dataframe(self):
        "Converts the segment properties to a pandas dataframe"
        df_dict = {"ids": self.ids}
        for prop in self._property_list():
            if (
                isinstance(prop, LabelProperty)
                or isinstance(prop, DescriptionProperty)
                or isinstance(prop, StringProperty)
                or isinstance(prop, NumberProperty)
            ):
                df_dict[prop.id] = prop.values
            elif isinstance(prop, TagProperty):
                for ii, tag in enumerate(prop.tags):
                    df_dict[tag] = [ii in tags for tags in prop.values]
        return pd.DataFrame(df_dict)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        id_col: str = "pt_root_id",
        label_col: Optional[str] = None,
        description_col: Optional[str] = None,
        string_cols: Optional[Union[str, List[str]]] = None,
        number_cols: Optional[Union[str, List[str]]] = None,
        tag_value_cols: Optional[Union[str, List[str]]] = None,
        tag_bool_cols: Optional[List[str]] = None,
        tag_descriptions: Optional[dict] = None,
    ):
        """Generate a segment property object from a pandas dataframe based on column

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing propeties
        id_col : str, optional
            Name of the column with object ids, by default "pt_root_id"
        label_col : Optional[str], optional
            Name of column to use for producing labels, by default None
        description_col : Optional[str], optional
            Name of column to use for producing descriptions, by default None
        string_cols : Optional[Union[str, list[str]]], optional
            Column (or list of columns) to use for string properties, by default None.
            WARNING: Neuroglancer does not currently display these properties.
        number_cols : Optional[Union[str, list[str]]], optional
            Column (or list of columns) to use for numeric properties, by default None.
        tag_value_cols : Optional[list[str]], optional
            Column (or list of columns) to generate tags based on unique values.
            Each column produces one tag per row based on the value, by default None
        tag_bool_cols : Optional[list[str]], optional
            List of columns to generate tags based on boolean values where each column is a tag, and each id gets the tag if it has a True in its row.
            By default None.
        tag_descriptions : Optional[dict], optional
            Dictionary of tag values to long-form tag descriptions, by default None.
            Tags without a key/value are passed through directly.

        Returns
        -------
        SegmentProperties
            Segment properties object
        """
        ids = df[id_col].tolist()
        properties = {}
        if label_col:
            properties["label_property"] = LabelProperty(values=df[label_col].tolist())
        if description_col:
            properties["description_property"] = DescriptionProperty(
                values=df[description_col].tolist()
            )
        if string_cols:
            if isinstance(string_cols, str):
                string_cols = [string_cols]
            properties["string_properties"] = [
                StringProperty(id=col, values=df[col].tolist()) for col in string_cols
            ]
        if number_cols:
            if isinstance(number_cols, str):
                number_cols = [number_cols]
            properties["number_properties"] = [
                NumberProperty(
                    id=col,
                    values=df[col].tolist(),
                    data_type=_find_column_dtype(df[col]),
                )
                for col in number_cols
            ]
        if tag_value_cols or tag_bool_cols:
            if isinstance(tag_value_cols, str):
                tag_value_cols = [tag_value_cols]
            if isinstance(tag_bool_cols, str):
                tag_bool_cols = [tag_bool_cols]
            properties["tag_properties"] = _make_tag_property(
                df,
                tag_value_cols,
                tag_bool_cols,
                tag_descriptions,
            )
        return cls(ids, **properties)

    @classmethod
    def from_dict(
        cls,
        seg_prop_dict: dict,
    ):
        """Generate a segment property object from a segment property dictionary

        Parameters
        ----------
        seg_prop_dict : dict
            Segment property dictionary, as imported from the json.

        Returns
        -------
        SegmentProperties
            Segment properties object
        """
        ids = seg_prop_dict["inline"]["ids"]
        props = seg_prop_dict["inline"]["properties"]
        prop_classes = {}
        for prop in props:
            if prop["type"] == "label":
                prop_classes["label_property"] = LabelProperty(
                    values=prop["values"],
                    description=prop.get("description"),
                )
            elif prop["type"] == "description":
                prop_classes["description_property"] = DescriptionProperty(
                    values=prop["values"],
                    description=prop.get("description"),
                )
            elif prop["type"] == "string":
                if "string_properties" not in prop_classes:
                    prop_classes["string_properties"] = []
                prop_classes["string_properties"].append(
                    StringProperty(
                        id=prop["id"],
                        values=prop["values"],
                        description=prop.get("description"),
                    )
                )
            elif prop["type"] == "number":
                if "number_properties" not in prop_classes:
                    prop_classes["number_properties"] = []
                prop_classes["number_properties"].append(
                    NumberProperty(
                        id=prop["id"],
                        values=prop["values"],
                        data_type=prop["data_type"],
                        description=prop.get("description"),
                    )
                )
            elif prop["type"] == "tags":
                prop_classes["tag_properties"] = TagProperty(
                    tags=prop["tags"],
                    values=prop["values"],
                    tag_descriptions=prop.get("tag_descriptions"),
                )
        return cls(ids, **prop_classes)
