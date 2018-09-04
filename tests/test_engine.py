from annotationengine import create_app
import cloudvolume
import numpy as np
import tempfile
from google.cloud import bigtable, exceptions
import subprocess
from annotationengine.anno_database import DoNothingCreds
from pychunkedgraph.backend import chunkedgraph
import grpc
from time import sleep
import os
import requests
from itertools import product
from signal import SIGTERM

# sudo kill `sudo lsof -t -i:8086`


def cg_settings():
    return 'test_dataset', 'cg_test'


def bigtable_emulator(cg_settings):
    cg_project, cg_table = cg_settings
    os.environ["BIGTABLE_EMULATOR_HOST"] = "localhost:8086"
    bigtables_emulator = subprocess.Popen(["gcloud",
                                           "beta",
                                           "emulators",
                                           "bigtable",
                                           "start",
                                           "--host-port",
                                           "localhost:8086",
                                           ],
                                          preexec_fn=os.setsid,
                                          stdout=subprocess.PIPE)

    startup_msg = "Waiting for BigTables Emulator to start up at {}..."
    print('bteh', startup_msg.format(os.environ["BIGTABLE_EMULATOR_HOST"]))
    c = bigtable.Client(project=cg_project,
                        credentials=DoNothingCreds(),
                        admin=True)
    retries = 5
    while retries > 0:
        try:
            c.list_instances()
        except exceptions._Rendezvous as e:
            # Good error - means emulator is up!
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                print("Ready!")
                break
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                sleep(1)
            retries -= 1
            print(".")
    if retries == 0:
        print("\nCouldn't start Bigtable Emulator."
              " Make sure it is setup correctly.")
        exit(1)

    return c


def cv(N=64, blockN=16):
    block_per_row = N / blockN
    tempdir = tempfile.mkdtemp()
    path = "file:/{}".format(tempdir)
    print(path)
    info = cloudvolume.CloudVolume.create_new_info(
            num_channels=1,
            layer_type='segmentation',
            data_type='uint64',
            encoding='raw',
            resolution=[4, 4, 40],
            voxel_offset=[0, 0, 0],
            chunk_size=[64, 64, 64],
            volume_size=[N, N, N],
            )
    print('info created')
    try:
        vol = cloudvolume.CloudVolume(path, info=info)
        vol.commit_info()
    except Exception as e:
        print(e)
    xx, yy, zz = np.meshgrid(np.arange(0, N),
                             np.arange(0, N),
                             np.arange(0, N))
    print('commited')
    seg = np.int64(xx / blockN)+ \
                block_per_row * np.int64(yy/blockN)+ \
                block_per_row * block_per_row * np.int64(zz/blockN)
    vol[:] = np.uint64(seg)
    return path


def create_chunk(cgraph, vertices=None, edges=None, timestamp=None):
    """
    Helper function to add vertices and edges to the chunkedgraph
    no safety checks!
    """
    if not vertices:
        vertices = []

    if not edges:
        edges = []

    vertices = np.unique(np.array(vertices, dtype=np.uint64))
    edges = [(np.uint64(v1), np.uint64(v2), np.float32(aff))
             for v1, v2, aff in edges]
    edge_ids = []
    cross_edge_ids = []
    edge_affs = []
    cross_edge_affs = []
    isolated_node_ids = [x for x in vertices
                         if (x not in [edges[i][0] for i in range(len(edges))])
                         and
                         (x not in [edges[i][1] for i in range(len(edges))])]

    for e in edges:
        if cgraph.test_if_nodes_are_in_same_chunk(e[0:2]):
            edge_ids.append([e[0], e[1]])
            edge_affs.append(e[2])
        else:
            cross_edge_ids.append([e[0], e[1]])
            cross_edge_affs.append(e[2])

    edge_ids = np.array(edge_ids, dtype=np.uint64).reshape(-1, 2)
    edge_affs = np.array(edge_affs, dtype=np.float32).reshape(-1, 1)
    cross_edge_ids = np.array(cross_edge_ids, dtype=np.uint64).reshape(-1, 2)
    cross_edge_affs = np.array(
        cross_edge_affs, dtype=np.float32).reshape(-1, 1)
    isolated_node_ids = np.array(isolated_node_ids, dtype=np.uint64)

    cgraph.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                      edge_affs, cross_edge_affs,
                                      isolated_node_ids)


def to_label(cgraph, l, x, y, z, segment_id):
    return cgraph.get_node_id(np.uint64(segment_id), layer=l, x=x, y=y, z=z)


def chunkgraph_tuple(bigtable_client,
                     cg_settings,
                     fan_out=2,
                     n_layers=3):
    cg_project, cg_table = cg_settings
    print('creating table {}'.format(cg_project))
    cgraph = chunkedgraph.ChunkedGraph(table_id=cg_table,
                                       project_id=cg_project,
                                       client=bigtable_client,
                                       credentials=DoNothingCreds(),
                                       instance_id="chunkedgraph",
                                       cv_path="",
                                       chunk_size=(32, 32, 32),
                                       is_new=True,
                                       fan_out=fan_out,
                                       n_layers=n_layers)
    return cgraph, cg_table


def test_cg(chunkgraph_tuple, N=64, blockN=16):
    """
    Create graph where a 4x4x4 grid of supervoxels is connected
    into a 2x2x2 grid of root_ids
    """

    cgraph, cg_table = chunkgraph_tuple

    for x, y, z in product(range(2), range(2), range(2)):
        verts = [to_label(cgraph, 1, x, y, z, k) for k in range(8)]
        create_chunk(cgraph,
                     vertices=verts,
                     edges=[(verts[k], verts[k + 1], .5) for k in range(7)])

    cgraph.add_layer(3, np.array(
        [[x, y, z] for x, y, z in product(range(2), range(2), range(2))]))

    return cgraph


def app(cv, test_dataset, cg_settings):
    cg_project, cg_table = cg_settings

    app = create_app(
        {
            'project_id': cg_project,
            'emulate': True,
            'TESTING': True,
            'DATASETS': [
                {
                    'name': test_dataset,
                    'CV_SEGMENTATION_PATH': cv
                }
            ],
            'BIGTABLE_CONFIG': {
                'emulate': True
            },
            'CHUNKGRAPH_TABLE_ID': cg_table
        }
    )
    return app


if __name__ == '__main__':
    import requests
    dataset = 'test_dataset'
    cg_setting = cg_settings()

    # setup bigtable and database schemas
    bigtable_test = bigtable_emulator(cg_setting)
    test_cv = cv()

    # create chunk graph and add vertices
    cgraph_tuple = chunkgraph_tuple(bigtable_test, cg_setting)
    cg = list(cgraph_tuple)
    chunks = create_chunk(cg[0])
    c_graph = test_cg(cgraph_tuple)
    test_app = app(test_cv, dataset, cg_setting)
    # test_app.run()
    # test client connection
    host = 'http://localhost:8086'
    s = requests.session()
    url = '{}/dataset/{}'.format(host, dataset)
    print(url)
    response = s.get(url)
    assert(response.status_code == 200)
