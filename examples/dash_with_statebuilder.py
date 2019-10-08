import os
import dash
import pandas as pd
from dashdataframe import configure_app
from neuroglancer_annotation_ui.statebuilder import StateBuilder

FLASK_PORT = 8880

df_filename = 'dash_example_data.h5'
df_key = 'data'
plot_columns = ['num_pre', 'num_post']

def build_dash_app():
    df = pd.read_hdf(df_filename, df_key)

    app = dash.Dash()

    sb = StateBuilder(dataset_name='pinky100',
                      selected_ids={'seg': ['pt_root_id']},
                      point_annotations={'soma': ['pt_position']})
    
    def render_state(select_indices, render_df):
        render_df = render_df.loc[select_indices]
        return sb.render_state(render_df,
                               return_as='url',
                               url_prefix='https://graphene-v0-dot-neuromancer-seung-import.appspot.com/')

    configure_app(app, df,
                  link_name='Neuroglancer Link',
                  link_func=render_state,
                  plot_columns = plot_columns) 
    return app


if __name__ == '__main__':
    app = build_dash_app()
    app.run_server(port=FLASK_PORT)

