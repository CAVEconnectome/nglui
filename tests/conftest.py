import pytest
import cloudvolume
import numpy as np
import tempfile
import shutil
from google.cloud import bigtable, exceptions
import subprocess
from annotationengine.anno_database import DoNothingCreds
import grpc
from time import sleep
import os
from signal import SIGTERM
from pychunkedgraph.backend import chunkedgraph
from itertools import product
from annotationengine import create_app
from annotationengine.annotationclient import AnnotationClient
import requests_mock

INFOSERVICE_ENDPOINT = "http://infoservice"
TEST_DATASET_NAME = 'test'
tempdir = tempfile.mkdtemp()
TEST_PATH = "file:/{}".format(tempdir)


@pytest.fixture(scope='session')
def bigtable_settings():
    return 'anno_test', 'cg_test'


@pytest.fixture(scope='session', autouse=True)
def bigtable_client(request, bigtable_settings):
    # setup Emulator
    cg_project, cg_table = bigtable_settings
    bt_emul_host = "localhost:8086"
    os.environ["BIGTABLE_EMULATOR_HOST"] = bt_emul_host
    bigtables_emulator = subprocess.Popen(["gcloud",
                                           "beta",
                                           "emulators",
                                           "bigtable",
                                           "start",
                                           "--host-port",
                                           bt_emul_host],
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
                print(" Ready!")
                break
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                sleep(1)
            retries -= 1
            print(".")
    if retries == 0:
        print("\nCouldn't start Bigtable Emulator."
              " Make sure it is setup correctly.")
        exit(1)

    yield c

    # setup Emulator-Finalizer
    def fin():
        try:
            gid = os.getpgid(bigtables_emulator.pid)
            os.killpg(gid, SIGTERM)
        except ProcessLookupError:
            pass
        bigtables_emulator.wait()
        print('BigTable stopped')
    request.addfinalizer(fin)


@pytest.fixture(scope='session')
def cv(N=64, blockN=16):

    block_per_row = int(N / blockN)

    chunk_size = [32, 32, 32]
    num_chunks = [int(N/cs) for cs in chunk_size]
    info = cloudvolume.CloudVolume.create_new_info(
        num_channels=1,
        layer_type='segmentation',
        data_type='uint64',
        encoding='raw',
        resolution=[4, 4, 40],  # Voxel scaling, units are in nanometers
        voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size=chunk_size,  # units are voxels
        volume_size=[N, N, N],
    )
    vol = cloudvolume.CloudVolume(TEST_PATH, info=info)
    vol.commit_info()
    xx, yy, zz = np.meshgrid(*[np.arange(0, N) for cs in chunk_size])
    id_ind = (np.uint64(xx / blockN),
                np.uint64(yy / blockN),
                np.uint64(zz / blockN))
    id_shape = (block_per_row, block_per_row, block_per_row)
    seg = np.ravel_multi_index(id_ind, id_shape)
    vol[:] = np.uint64(seg)

    yield TEST_PATH
    shutil.rmtree(tempdir)

@pytest.fixture(scope='session')
def test_dataset():
    return TEST_DATASET_NAME


@pytest.fixture(scope='session')
def app(cv, test_dataset, bigtable_settings):
    cg_project, cg_table = bigtable_settings

    with requests_mock.Mocker() as m:
        dataset_url = os.path.join(INFOSERVICE_ENDPOINT, 'api/datasets')
        m.get(dataset_url, json=[test_dataset])
        dataset_info_url = os.path.join(INFOSERVICE_ENDPOINT,
                                        'api/dataset/{}'.format(test_dataset))
        dataset_d = {
            "annotation_engine_endpoint": "http://35.237.200.246",
            "flat_segmentation_source": cv,
            "id": 1,
            "image_source": cv,
            "name": test_dataset,
            "pychunkgraph_endpoint": "http://pcg/segmentation",
            "pychunkgraph_segmentation_source": cv
        }
        m.get(dataset_info_url, json=dataset_d)
        app = create_app(
            {
                'project_id': cg_project,
                'emulate': True,
                'TESTING': True,
                'INFOSERVICE_ENDPOINT':  INFOSERVICE_ENDPOINT,
                'BIGTABLE_CONFIG': {
                    'emulate': True
                },
                'CHUNKGRAPH_TABLE_ID': cg_table
            }
        )

    yield app


@pytest.fixture(scope='session')
def client(app):
    return app.test_client()

def mock_info_service(requests_mock):
    dataset_url = os.path.join(INFOSERVICE_ENDPOINT, 'api/datasets')
    requests_mock.get(dataset_url, json=[TEST_DATASET_NAME])
    dataset_info_url = os.path.join(INFOSERVICE_ENDPOINT,
                                    'api/dataset/{}'.format(TEST_DATASET_NAME))
    dataset_d = {
        "annotation_engine_endpoint": "http://35.237.200.246",
        "flat_segmentation_source": TEST_PATH,
        "id": 1,
        "image_source": TEST_PATH,
        "name": TEST_DATASET_NAME,
        "pychunkgraph_endpoint": "http://pcg/segmentation",
        "pychunkgraph_segmentation_source": TEST_PATH
    }
    requests_mock.get(dataset_info_url, json=dataset_d)


@pytest.fixture(scope='session')
def client(app):
    return app.test_client()

class TestAnnotationClient(AnnotationClient):
    def __init__(self, dataset_name, test_client):
        super(TestAnnotationClient, self).__init__('', dataset_name)
        self.session = test_client

    def get_datasets(self):
        """ Returns existing datasets

        :return: list
        """
        url = "{}/dataset".format(self.endpoint)
        response = self.session.get(url)
        assert(response.status_code == 200)
        return response.json

    def get_dataset(self, dataset_name=None):
        """ Returns information about the dataset

        :return: dict
        """
        if dataset_name is None:
            dataset_name = self.dataset_name
        url = "{}/dataset/{}".format(self.endpoint, dataset_name)
        response = self.session.get(url)
        assert(response.status_code == 200)
        return response.json

    def get_annotation(self, annotation_type, oid, dataset_name=None):
        """
        Returns information about one specific annotation
        :param dataset_name: str
        :param annotation_type: str
        :param oid: int
        :return dict
        """
        if dataset_name is None:
            dataset_name = self.dataset_name
        url = "{}/annotation/dataset/{}/{}/{}".format(self.endpoint,
                                                   dataset_name,
                                                   annotation_type,
                                                   oid)
        response = self.session.get(url)
        assert(response.status_code == 200)
        return response.json

    def post_annotation(self, annotation_type, data, dataset_name=None):
        """
        Post an annotation to the annotationEngine.
        :param dataset_name: str
        :param annotation_type: str
        :param data: dict
        :return dict
        """
        if dataset_name is None:
            dataset_name = self.dataset_name
        if isinstance(data,dict):
            data=[data]

        url = "{}/annotation/dataset/{}/{}".format(self.endpoint,
                                                   dataset_name,
                                                   annotation_type)
        response = self.session.post(url, json=data)
        assert(response.status_code == 200)
        return response.json

    def update_annotation(self, annotation_type, oid, data, dataset_name=None):
        if dataset_name is None:
            dataset_name = self.dataset_name
        url = "{}/annotation/dataset/{}/{}/{}".format(self.endpoint,
                                                   dataset_name,
                                                   annotation_type,
                                                   oid)
        response = self.session.put(url, json=data)
        assert(response.status_code == 200)
        return response.json

    def delete_annotation(self, annotation_type, oid, dataset_name=None):
        """
        Delete an existing annotation
        :param dataset_name: str
        :param annotation_type: str
        :param oid: int
        :return dict
        """
        if dataset_name is None:
            dataset_name = self.dataset_name
        url = "{}/annotation/dataset/{}/{}/{}".format(self.endpoint,
                                                   dataset_name,
                                                   annotation_type,
                                                   oid)
        response = self.session.delete(url)
        assert(response.status_code == 200)
        return response.json

@pytest.fixture(scope='session')
def annotation_client(test_dataset, client):
    return TestAnnotationClient(test_dataset, client)

@pytest.fixture(scope='session')
def img_layer():
    return 'precomputed://gs://neuroglancer-public-data/flyem_fib-25/image'

@pytest.fixture(scope='session')
def seg_layer():
    return 'precomputed://gs://neuroglancer-public-data/flyem_fib-25/ground_truth'

class S_Test():
    def __init__(self, pos):
        self.mouse_voxel_coordinates = pos

@pytest.fixture(scope='session')
def s1():
    return S_Test([1,1,1])

@pytest.fixture(scope='session')
def s2():
    return S_Test([2,2,2])

@pytest.fixture(scope='session')
def s3():
    return S_Test([3,3,3])