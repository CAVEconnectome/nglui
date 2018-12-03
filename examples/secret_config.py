config = dict(sqlalchemy_database_uri="postgresql://analysis_user:connectallthethings@35.196.105.34/postgres",
              materialization_version=39,
              dataset_name='pinky100',
              correct_seg_src='precomputed://gs://neuroglancer/nkem/pinky100_v0/ws/lost_no-random/bbox1_0',
              cleft_src='precomputed://gs://neuroglancer/pinky100_v0/clefts/mip1_d2_1175k',
              )