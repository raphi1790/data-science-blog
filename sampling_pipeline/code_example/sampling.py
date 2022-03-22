import argparse

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from tensorflow_transform.tf_metadata import  metadata_io
import tfx_bsl.public.tfxio as tfxio

ORDERED_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label'
]
CATEGORICAL_FEATURE_KEYS = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]
NUMERIC_FEATURE_KEYS = [
    'age',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
]
OPTIONAL_NUMERIC_FEATURE_KEYS = [
    # 'education-num',
]

FEATURES = NUMERIC_FEATURE_KEYS + OPTIONAL_NUMERIC_FEATURE_KEYS + CATEGORICAL_FEATURE_KEYS
LABELS = ['label']


def read_from_gcs(pipeline, tfrecord_tfxio):
    """
    Read all tfrecord-records stored at the input_path-location
    :param pipeline: beam-pipeline object
    :param tfrecord_tfxio: mapper for TFRecord into provided schema
    :return: PCollection, ready for further processing
    """

    raw_data = (
            pipeline
            | 'Read All - Read Data from Cloud-Storage' >> tfrecord_tfxio.BeamSource(batch_size=1)
            | 'Extract Dict from pa.RecordBatch' >> beam.Map(lambda x: x.to_pydict())
    )
    return raw_data


def read_sample_from_gcs(pipeline, tfrecord_tfxio, sample_size):
    """
    Read a sample of tfrecord-records stored at the input_path-location
    :param sample_size: size of the random sample; if <1 then the value is treated as a percentage
    :param pipeline: beam-pipeline object
    :param tfrecord_tfxio: mapper for TFRecord into provided schema
    :return: PCollection, ready for further processing
    """
    if sample_size < 1:
        _, sampled_data = (
                pipeline
                | 'Read Sample - Read Data from Cloud-Storage' >> tfrecord_tfxio.BeamSource(batch_size=1)
                | 'Sampling' >> beam.Partition(
                    lambda elem, _: int(random.uniform(0, 1) < sample_size), 2)
        raw_data = (
                sampled_data
                | 'Extract Dict from pa.RecordBatch ' >> beam.Map(lambda x: x.to_pydict())
        )
    else:
        raw_data = (
                pipeline
                | 'Read Sample - Read Data from Cloud-Storage' >> tfrecord_tfxio.BeamSource(batch_size=1)
                | 'Sample N elements' >> beam.combiners.Sample.FixedSizeGlobally(sample_size)
                | 'Split sample-collection into single records ' >> beam.FlatMap(lambda x: x)
                | 'Extract Dict from pa.RecordBatch ' >> beam.Map(lambda x: x.to_pydict())
        )
    return raw_data


class DictToCSVFn(beam.DoFn):
    """
    This class prepares the values from the PCollection for writing into a csv-file
    """

    def __init__(self, features, labels):
        self._features = features
        self._labels = labels
        import numpy as np
        self._np = np

    def process(self, element):
        """
        Extracts the values within an element.
        Due to the obj.to_pydict()-method the values are nested within lists.
        """
        output_features = {}
        for feature in self._features:
            values = element[feature][0]
            output_features[feature] = self._fix_value_type(values[0])

        for label in self._labels:
            label_value = element[label][0][0]
            output_features[label] = str(label_value)
        return [output_features]

    def _fix_value_type(self, value):
        """
        This method transforms the input-values into strings.
        """
        if type(value) == int or type(value) == float or isinstance(value, self._np.float32):
            return str(value)
        else:
            return str(value.decode("utf-8"))


def transform_and_write(raw_data, output_path, features, labels):
    # Create header of .csv-file
    csv_header = ""
    for column in features + labels[:-1]:
        csv_header += column + ","
    csv_header += labels[-1]

    main_data = (raw_data
                 | 'Create Dictionary' >> beam.ParDo(DictToCSVFn(features, labels))
                 # | 'Print Dict' >> beam.Map(print_example)
                 | 'Read Dictionary-Values' >> beam.Map(lambda x: list(x.values()))
                 | 'Transform Rows into .csv-format' >> beam.Map(lambda row: ','.join([column for column in row]))
                 | 'Write' >> beam.io.WriteToText(output_path, file_name_suffix='.csv', header=csv_header)
                 )


if __name__ == '__main__':
    """Read tfrecord-files and therefore maps the schema stored in the PreprocessData-file \
    If parameter sample-size  is set, it samples the records randomly\
    Outputs a .csv-file at the ouptut-path \
    It's possible to run the pipeline either using the DirectRunner or DataflowRunner. \
    Therefore just provide the arguments --runner, --zone, --region, --max_num_workers and temp_location """

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--work-dir',
        required=True,
        dest='work_dir',
        help='Directory for staging and working files. '
             'This can be a Google Cloud Storage path.')
    arg_parser.add_argument(
        '--output-path',
        dest='output_path',
        required=True,
        help='Output-Path, either local or GS-path; including filename without suffix')
    arg_parser.add_argument(
        '--mode',
        dest='mode',
        required=True,
        default='train',
        help='Decides whether we take the train- or the eval-directory. Possible Values: "train", "eval"')
    arg_parser.add_argument(
        '--sample-size',
        dest='sample_size',
        required=False,
        default=0,
        help='Size of the random-sample'
    )
    arg_parser.add_argument(
        '--input-credentials',
        dest='input_credentials',
        required=False,
        default=None,
        help='Path to service-key.json'
    )

    # Parse arguments from the command line
    known_args, pipeline_args = arg_parser.parse_known_args()
    pipeline_options = PipelineOptions(pipeline_args)

    # folder of schema.pbtxt; exported from tensorflow-transform
    metadata_path = "transformed_data/transformed_metadata"
    # Read schema with tensorflow_transform
    schema = metadata_io.read_metadata(metadata_path).schema

    # Based on the mode we collect the .tfrecord-files from train- or test-folder
    if known_args.mode == 'train':
        # takes the files containing in train-folder
        file_pattern = "transformed_data/train_transformed/train_part*"
    else:
        # takes the files containing in test-folder
        file_pattern = "transformed_data/train_transformed/test_part*"
    print("file_pattern", file_pattern)

    # Create a TFX-IO-object capable of reading .tfrecord-files
    # This reader will be used within the pipeline
    tfrecord_tfxio = tfxio.TFExampleRecord(file_pattern=file_pattern, schema=schema)
    print("known_args.output_path", known_args.output_path)
    # Define the beam-pipeline
    with beam.Pipeline(options=pipeline_options) as pipeline:
        if int(known_args.sample_size) > 0:
            # The records are randomly sampled using Apache Beam
            print("sampling with {} records".format(int(known_args.sample_size)))
            raw_data = read_sample_from_gcs(pipeline, tfrecord_tfxio, int(known_args.sample_size))
        else:
            print("full export")
            raw_data = read_from_gcs(pipeline, tfrecord_tfxio)

        transform_and_write(raw_data, known_args.output_path, FEATURES, LABELS)
