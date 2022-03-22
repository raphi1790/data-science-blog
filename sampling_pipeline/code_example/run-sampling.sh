WORK_DIR=.

MODE='train' # 'test' or 'train', depending on the directory containing the TFRecords
SAMPLE_SIZE=3 # size of the random sample; 0 --> all records
CURRENT_DATE=`date +%Y%m%d_%H%M%S`
OUTPUT_PATH=${WORK_DIR}/sampled_data/${MODE}_${CURRENT_DATE}

RUNNER=DirectRunner # DataflowRunner OR DirectRunner
PROJECT_ID=adtrac-data-and-analytics
REGION=europe-west1
ZONE=europe-west6-a
MAX_NUM_WORKERS=5
INPUT_CREDENTIALS=/mnt/c/Users/rapha/Documents/Raphael/keys/dataflow_keys.json

# Wrapper function to print the command being run
function run {
  # shellcheck disable=SC2145
  echo "$ $@"
  "$@"
}

echo '>> Sampling'
run python sampling.py --project $PROJECT_ID \
  --runner=${RUNNER}    \
  --temp_location=${WORK_DIR}/beam-temp  \
  --region=${REGION} \
  --worker_zone=${ZONE} \
  --setup_file ./setup.py  \
  --work-dir=${WORK_DIR} \
  --max_num_workers=${MAX_NUM_WORKERS}   \
  --input-credentials=${INPUT_CREDENTIALS} \
  --mode=${MODE} \
  --output-path=${OUTPUT_PATH} \
  --sample-size=${SAMPLE_SIZE}
echo ''