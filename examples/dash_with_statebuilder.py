'''
NOTE:
This example uses dashdataframe to make a dash app for exploring data and genenerating
Neuroglancer links. This can be installed with `pip install dashdataframe` or through
github at https://github.com/AllenInstitute/DashDataFrame.
'''

import os
import dash
import pandas as pd
from dashdataframe import configure_app
from nglui import statebuilder

FLASK_PORT = 8880

df_filename = 'dash_example_data.h5'
df_key = 'data'
plot_columns = ['pre_syn', 'post_syn']


def make_statebuilder():
    image_layer = statebuilder.ImageLayerConfig(name='EM',
                                                source='precomputed://gs://neuroglancer/pinky100_v0/son_of_alignment_v15_rechunked')
    seg_layer = statebuilder.SegmentationLayerConfig(name='layer23',
                                                     source='precomputed://gs://microns_public_datasets/pinky100_v185/seg',
                                                     selected_ids_column='pt_root_id')
    points = statebuilder.PointMapper('pt_position')
    anno_layer = statebuilder.AnnotationLayerConfig(
        name='annos', mapping_rules=points)
    return statebuilder.StateBuilder([image_layer, seg_layer, anno_layer])


def build_dash_app():
    df = pd.read_hdf(df_filename, df_key)

    app = dash.Dash()

    sb = make_statebuilder()

    def render_state(select_indices, render_df):
        render_df = render_df.loc[select_indices]
        return sb.render_state(render_df,
                               return_as='url',
                               url_prefix='https://neuromancer-seung-import.appspot.com/')

    configure_app(app, df,
                  link_name='Neuroglancer Link',
                  link_func=render_state,
                  plot_columns=plot_columns)
    return app


if __name__ == '__main__':
    app = build_dash_app()
    app.run_server(port=FLASK_PORT)
