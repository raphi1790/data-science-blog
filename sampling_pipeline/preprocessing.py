import math
import os
import pprint

import tensorflow as tf

print('TF: {}'.format(tf.__version__))

import apache_beam as beam

print('Beam: {}'.format(beam.__version__))

import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

print('Transform: {}'.format(tft.__version__))

from tfx_bsl.public import tfxio
from tfx_bsl.coders.example_coder import RecordBatchToExamples

import tempfile

train = 'raw_data/adult.data'
test = 'raw_data/adult.test'

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
ORDERED_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label'
]
LABEL_KEY = 'label'

RAW_DATA_FEATURE_SPEC = dict(
    [(name, tf.io.FixedLenFeature([], tf.string))
     for name in CATEGORICAL_FEATURE_KEYS] +
    [(name, tf.io.FixedLenFeature([], tf.float32))
     for name in NUMERIC_FEATURE_KEYS] +
    [(name, tf.io.VarLenFeature(tf.float32))
     for name in OPTIONAL_NUMERIC_FEATURE_KEYS] +
    [(LABEL_KEY, tf.io.FixedLenFeature([], tf.string))]
)

SCHEMA = tft.tf_metadata.dataset_metadata.DatasetMetadata(
    tft.tf_metadata.schema_utils.schema_from_feature_spec(RAW_DATA_FEATURE_SPEC)).schema

NUM_OOV_BUCKETS = 1

# Names of temp files
TRANSFORMED_TRAIN_DATA_FILEBASE = 'train_transformed/train_part'
TRANSFORMED_TEST_DATA_FILEBASE = 'test_transformed/test_part'


def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    # Since we are modifying some features and leaving others unchanged, we
    # start by setting `outputs` to a copy of `inputs.
    outputs = inputs.copy()

    # Scale numeric columns to have range [0, 1].
    for key in NUMERIC_FEATURE_KEYS:
        outputs[key] = tft.scale_to_0_1(inputs[key])

    for key in OPTIONAL_NUMERIC_FEATURE_KEYS:
        # This is a SparseTensor because it is optional. Here we fill in a default
        # value when it is missing.
        sparse = tf.sparse.SparseTensor(inputs[key].indices, inputs[key].values,
                                        [inputs[key].dense_shape[0], 1])
        dense = tf.sparse.to_dense(sp_input=sparse, default_value=0.)
        # Reshaping from a batch of vectors of size 1 to a batch to scalars.
        dense = tf.squeeze(dense, axis=1)
        outputs[key] = tft.scale_to_0_1(dense)

    # For all categorical columns except the label column, we generate a
    # vocabulary but do not modify the feature.  This vocabulary is instead
    # used in the trainer, by means of a feature column, to convert the feature
    # from a string to an integer id.
    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[key] = tft.compute_and_apply_vocabulary(
            tf.strings.strip(inputs[key]),
            num_oov_buckets=NUM_OOV_BUCKETS,
            vocab_filename=key)

    # for key in CATEGORICAL_FEATURE_KEYS:
    #     tft.vocabulary(inputs[key], vocab_filename=key)

    # For the label column we provide the mapping from string to index.
    table_keys = ['>50K', '<=50K']
    with tf.init_scope():
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys=table_keys,
            values=tf.cast(tf.range(len(table_keys)), tf.int64),
            key_dtype=tf.string,
            value_dtype=tf.int64)
        table = tf.lookup.StaticHashTable(initializer, default_value=-1)
    # Remove trailing periods for test data when the data is read with tf.data.
    label_str = tf.strings.regex_replace(inputs[LABEL_KEY], r'\.', '')
    label_str = tf.strings.strip(label_str)
    data_labels = table.lookup(label_str)
    transformed_label = tf.one_hot(
        indices=data_labels, depth=len(table_keys), on_value=1.0, off_value=0.0)
    outputs[LABEL_KEY] = tf.reshape(transformed_label, [-1, len(table_keys)])

    return outputs


def transform_data(train_data_file, test_data_file, working_dir):
    """Transform the data and write out as a TFRecord of Example protos.

    Read in the data using the CSV reader, and transform it using a
    preprocessing pipeline that scales numeric data and converts categorical data
    from strings to int64 values indices, by creating a vocabulary for each
    category.

    Args:
    train_data_file: File containing training data
    test_data_file: File containing test data
    working_dir: Directory to write transformed data and metadata to
    """

    # The "with" block will create a pipeline, and run that pipeline at the exit
    # of the block.
    with beam.Pipeline() as pipeline:
        with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
            # Create a TFXIO to read the census data with the schema. To do this we
            # need to list all columns in order since the schema doesn't specify the
            # order of columns in the csv.
            # We first read CSV files and use BeamRecordCsvTFXIO whose .BeamSource()
            # accepts a PCollection[bytes] because we need to patch the records first
            # (see "FixCommasTrainData" below). Otherwise, tfxio.CsvTFXIO can be used
            # to both read the CSV files and parse them to TFT inputs:
            # csv_tfxio = tfxio.CsvTFXIO(...)
            # raw_data = (pipeline | 'ToRecordBatches' >> csv_tfxio.BeamSource())
            csv_tfxio = tfxio.BeamRecordCsvTFXIO(
                physical_format='text',
                column_names=ORDERED_CSV_COLUMNS,
                schema=SCHEMA)

            # Read in raw data and convert using CSV TFXIO.  Note that we apply
            # some Beam transformations here, which will not be encoded in the TF
            # graph since we don't do the from within tf.Transform's methods
            # (AnalyzeDataset, TransformDataset etc.).  These transformations are just
            # to get data into a format that the CSV TFXIO can read, in particular
            # removing spaces after commas.
            raw_data = (
                    pipeline
                    | 'ReadTrainData' >> beam.io.ReadFromText(
                train_data_file, coder=beam.coders.BytesCoder())
                    | 'FixCommasTrainData' >> beam.Map(
                lambda line: line.replace(b', ', b','))
                    | 'DecodeTrainData' >> csv_tfxio.BeamSource())

            # Combine data and schema into a dataset tuple.  Note that we already used
            # the schema to read the CSV data, but we also need it to interpret
            # raw_data.
            raw_dataset = (raw_data, csv_tfxio.TensorAdapterConfig())

            # The TFXIO output format is chosen for improved performance.
            transformed_dataset, transform_fn = (
                    raw_dataset | tft_beam.AnalyzeAndTransformDataset(
                preprocessing_fn, output_record_batches=True))

            # Transformed metadata is not necessary for encoding.
            transformed_data, _ = transformed_dataset

            # Extract transformed RecordBatches, encode and write them to the given
            # directory.
            _ = (
                    transformed_data
                    | 'EncodeTrainData' >>
                    beam.FlatMapTuple(lambda batch, _: RecordBatchToExamples(batch))
                    | 'WriteTrainData' >> beam.io.WriteToTFRecord(
                os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE)))

            # Now apply transform function to test data.  In this case we remove the
            # trailing period at the end of each line, and also ignore the header line
            # that is present in the test data file.
            raw_test_data = (
                    pipeline
                    | 'ReadTestData' >> beam.io.ReadFromText(
                test_data_file, skip_header_lines=1,
                coder=beam.coders.BytesCoder())
                    | 'FixCommasTestData' >> beam.Map(
                lambda line: line.replace(b', ', b','))
                    | 'RemoveTrailingPeriodsTestData' >> beam.Map(lambda line: line[:-1])
                    | 'DecodeTestData' >> csv_tfxio.BeamSource())

            raw_test_dataset = (raw_test_data, csv_tfxio.TensorAdapterConfig())

            # The TFXIO output format is chosen for improved performance.
            transformed_test_dataset = (
                    (raw_test_dataset, transform_fn)
                    | tft_beam.TransformDataset(output_record_batches=True))

            # Transformed metadata is not necessary for encoding.
            transformed_test_data, _ = transformed_test_dataset

            # Extract transformed RecordBatches, encode and write them to the given
            # directory.
            _ = (
                    transformed_test_data
                    | 'EncodeTestData' >>
                    beam.FlatMapTuple(lambda batch, _: RecordBatchToExamples(batch))
                    | 'WriteTestData' >> beam.io.WriteToTFRecord(
                os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE)))

            # Will write a SavedModel and metadata to working_dir, which can then
            # be read by the tft.TFTransformOutput class.
            _ = (
                    transform_fn
                    | 'WriteTransformFn' >> tft_beam.WriteTransformFn(working_dir))


if __name__ == "__main__":
    working_dir = os.path.join("transformed_data")
    transform_data(train, test, working_dir)
