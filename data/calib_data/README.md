## 校准与评测数据
- 已预生成 5 份校准/评测 JSON（位于 `data/calib_data`）：
  - `data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json`
  - `data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json`
  - `data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json`
  - `data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json`
  - `data/calib_data/Qwen2.5-VL-7B-Instruct_OCRBench_20250923102641_struct.json`
- 对应图片压缩包：`http://10.10.1.53:8081/artifactory/model_zoo2/houmo/vllm_datasets/LMUData.tar.gz`，下载后解压至 `data/calib_data`。