from functools import partial
from itertools import chain
from typing import List, Optional, Union

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
    if value is None or pd.isna(value) or value == "":
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
    id = attrs.field(default="label", type=str, converter=str, kw_only=True)
    type = attrs.field(init=False, default="label", type=str)
    values = attrs.field(type=list, converter=list_of_strings, kw_only=True)
    description = attrs.field(type=Optional[str], default=None, kw_only=True)

    def __len__(self):
        return len(self.values)


@attrs.define
class DescriptionProperty(SegmentPropertyBase):
    id = attrs.field(default="description", type=str, converter=str, kw_only=True)
    type = attrs.field(init=False, default="description", type=str)
    values = attrs.field(type=list, converter=list_of_strings, kw_only=True)
    description = attrs.field(type=Optional[str], default=None, kw_only=True)

    def __len__(self):
        return len(self.values)


@attrs.define
class StringProperty(SegmentPropertyBase):
    id = attrs.field(type=str, converter=str, kw_only=True)
    type = attrs.field(init=False, default="string", type=str)
    values = attrs.field(type=list, converter=list_of_strings, kw_only=True)
    description = attrs.field(type=Optional[str], default=None, kw_only=True)

    def __len__(self):
        return len(self.values)


@attrs.define
class NumberProperty(SegmentPropertyBase):
    id = attrs.field(type=str, converter=str, kw_only=True)
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

    def __repr__(self):
        return f"TagProperty(id='{self.id}', {len(self.tags)} tags, {len(self.values)} values)"


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


def _make_tag_map(
    df, value_columns, bool_columns, allow_disambiguation, prepend_col_name
):
    if prepend_col_name:
        for col in value_columns:
            df[col] = df[col].apply(lambda x: f"{col}:{x}")
    unique_tags = {}
    for col in value_columns:
        col_tags = df[col].unique()
        if isinstance(col_tags, np.ndarray):
            unique_tags[col] = sorted(
                [x for x in col_tags.tolist() if not is_null_value(x)]
            )
        else:
            unique_tags[col] = [
                x for x in col_tags.sort_values().tolist() if not is_null_value(x)
            ]
    for col in bool_columns:
        unique_tags[col] = [col]
    vals, counts = np.unique(
        np.concatenate([v for v in unique_tags.values()]), return_counts=True
    )
    duplicates = vals[counts > 1]
    swap_values = []
    if len(duplicates) > 0:
        if not allow_disambiguation:
            raise ValueError(f"Duplicate tags found: {duplicates}")
        for col in unique_tags:
            for dup in duplicates:
                if dup in unique_tags[col]:
                    if col in value_columns:
                        df.loc[:, col] = df[col].replace(dup, f"{col}:{dup}")
                        swap_values.append(
                            (col, unique_tags[col].index(dup), f"{col}:{dup}")
                        )
    for col, idx, swap in swap_values:
        unique_tags[col][idx] = swap
    tags = np.concatenate([v for v in unique_tags.values()]).tolist()
    return tags, {tag: i for i, tag in enumerate(tags)}, df


def _generate_tag_values(df, value_columns, bool_columns, tag_map):
    index_col = "new_column_index_temp_"
    tag_df = df.assign(**{index_col: np.arange(len(df))})
    concat_dfs = []
    if len(value_columns) > 0:
        tag_df_long = (
            tag_df.melt(
                id_vars=index_col,
                value_vars=value_columns,
            )
            .dropna(how="any")
            .sort_values(by=index_col)
        )
        tag_df_long["tag_value"] = (
            tag_df_long["value"].astype(object).apply(lambda x: tag_map[x]).values
        )
        concat_dfs.append(tag_df_long[[index_col, "tag_value"]])
    if len(bool_columns) > 0:
        for col in bool_columns:
            concat_dfs.append(
                pd.DataFrame(
                    {
                        index_col: np.flatnonzero(tag_df[col]),
                        "tag_value": tag_map[col],
                    }
                )
            )
    tag_values = (
        pd.concat(concat_dfs, ignore_index=True)
        .groupby(index_col)["tag_value"]
        .apply(lambda x: sorted(list(x)))
    )
    # IDs without a tag would be missing from the above
    index_missing = np.setdiff1d(
        np.arange(len(tag_df)),
        tag_values.index,
    )
    for idx in index_missing:
        tag_values[idx] = []
    return tag_values.sort_index().tolist()


def _make_tag_property(
    df,
    value_columns,
    bool_columns,
    tag_descriptions,
    name="tags",
    allow_disambiguation=True,
    prepend_col_name=False,
):
    if value_columns is None:
        value_columns = []
    if bool_columns is None:
        bool_columns = []
    tags, tag_map, tag_df = _make_tag_map(
        df[value_columns + bool_columns].replace({"": None}).copy(),
        value_columns,
        bool_columns,
        allow_disambiguation,
        prepend_col_name,
    )
    tag_values = _generate_tag_values(tag_df, value_columns, bool_columns, tag_map)
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
            if isinstance(
                prop,
                (DescriptionProperty, LabelProperty, NumberProperty, StringProperty),
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
        label_col: Optional[Union[str, List[str]]] = None,
        description_col: Optional[str] = None,
        string_cols: Optional[Union[str, List[str]]] = None,
        number_cols: Optional[Union[str, List[str]]] = None,
        tag_value_cols: Optional[Union[str, List[str]]] = None,
        tag_bool_cols: Optional[List[str]] = None,
        tag_descriptions: Optional[dict] = None,
        allow_disambiguation: bool = True,
        label_separator: str = "_",
        label_format_map: Optional[str] = None,
        prepend_col_name: bool = False,
        random_columns: Optional[int] = None,
        random_column_prefix: str = "random_sample",
    ):
        """Generate a segment property object from a pandas dataframe based on column

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing propeties
        id_col : str, optional
            Name of the column with object ids, by default "pt_root_id"
        label_col : Optional[str, list[str]], optional
            Name of column or columns to use for producing labels, by default None.
            If multiple columns are provided, they will be concatenated with the label_separator.
            Null values are skipped.
        description_col : Optional[str], optional
            Name of column to use for producing descriptions, by default None
        string_cols : Optional[Union[str, list[str]]], optional
            Column (or list of columns) to use for string properties, by default None.
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
        allow_disambiguation : bool, optional
            If True, will prepend the column name in the case of duplicate tags, by default True.
        label_separator : str, optional
            Separator to use when assembling multiple columns into a label, by default "_"
        label_format_map : Optional[str], optional
            Format string to use for label formatting, by default None.
            If provided, will override the label separator and use the format string to format the label
            via the "format" function, replacing the column names in `{..}` with the values.
            For example, "{cell_class}: {cell_type}_{region}" would pluck values from the columns
            "cell_class", "cell_type", and "region". Label columns will be ignored and the format string is not validated.
        prepend_col_name : bool, optional
            If True, will prepend the column name to tag values, by default False.
            This will effectively disambiguate all tags as well.

        Returns
        -------
        SegmentProperties
            Segment properties object
        """
        if random_columns:
            df = df.copy()
            random_column_names = []
            if random_columns > 1:
                random_column_names = [
                    f"{random_column_prefix}_{i}" for i in range(random_columns)
                ]
            else:
                random_column_names = [random_column_prefix]
            for col in random_column_names:
                if col in df.columns:
                    raise ValueError(f"Column {col} already exists in dataframe")
                df[col] = np.random.rand(len(df))
            if number_cols is None:
                number_cols = []
            elif isinstance(number_cols, str):
                number_cols = [number_cols]
            number_cols = number_cols + random_column_names

        ids = df[id_col].tolist()
        properties = {}
        if label_col or label_format_map:
            if isinstance(label_col, str):
                label_col = [label_col]
            if label_format_map:
                label_vals = df.apply(
                    lambda x: label_format_map.format(**x.to_dict()), axis=1
                ).to_list()
            else:
                label_vals = [
                    label_separator.join(filter(None, r))
                    for r in df[label_col]
                    .where(pd.notnull(df[label_col]), "")
                    .astype(str)
                    .values
                ]
            properties["label_property"] = LabelProperty(values=label_vals)
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
                allow_disambiguation=allow_disambiguation,
                prepend_col_name=prepend_col_name,
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
