<div align="center">
<h1> Object-centric Video Question Answering with Visual Grounding and Referring </h1>

[🏡 Project Page](https://qirui-chen.github.io/RGA3-release) |  [📄 Paper]() | [📦 VideoInfer Dataset](https://www.dropbox.com/scl/fo/9mcd1yrf8ca8b5heziqz4/AKfHt8pYjPvi0_kQUk8hx9o?rlkey=e7p4d0v3e2zuih7rbsuynrmd0&st=nqd8bhym&dl=0) | [🤗 RGA3 Checkpoints](https://huggingface.co/SurplusDeficit/UniGR-7B)

</div>

## News
* **[2025-07]** We have released the paper, codes, datasets and checkpoints.


## Environment

First create conda environment according to your CUDA version.
```
conda create -n rga3 python=3.10.16 -y
conda activate rga3
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia

pip install --upgrade pip  # enable PEP 660 support 
pip install -r requirements.txt

pip install ninja
pip install flash-attn --no-build-isolation
```

Then you need to install the [SAM2](https://github.com/facebookresearch/sam2?tab=readme-ov-file#installation) package. In our implementation, the version of core packages are as follows:
```bash 
torch==2.5.1+cu124
flash_attn==2.7.4post1
```

Then, install the [CoTracker3](https://github.com/facebookresearch/co-tracker?tab=readme-ov-file#install-a-development-version) package. Afterwards, install the following packages.

```bash
apt update && apt install openjdk-11-jdk -y && apt install zip
```

**Trouble Shooting**:
Since we adopt an early version of Qwen2.5-VL (4.49.0.dev0 for HuggingFace), some bfloat16 problems should be manually addressed, according to this [issue](https://github.com/QwenLM/Qwen2.5-VL/issues/706).


## Demo

After downloading checkpoints & installing environments, you can open an interface to inference via Gradio.

```bash
python app.py --version /PATH/TO/UniGR-7B
```

![demo](assets/demo.gif)


## Prepare Datasets

You can check the used training datasets and the corresponding sampling rate in `run_torchrun.sh` and `utils/dataset.py`.

- For image segmentation datasets, please refer to [LISA](https://github.com/dvlab-research/LISA/tree/main?tab=readme-ov-file#training-data-preparation).
- For video segmentation datasets, please refer to [VideoLISA](https://github.com/showlab/VideoLISA/blob/main/README.md#prepare-data-for-training) & [ReVOS](https://github.com/cilinyan/ReVOS-api).
- For region-level image question-answering datasets, please refer to [ViP-LLaVA](https://github.com/WisconsinAIVision/ViP-LLaVA?tab=readme-ov-file#visual-instruction-tuning) & [Osprey](https://github.com/CircleRadon/Osprey?tab=readme-ov-file#dataset-).
- For region-level video question-answering datasets, you can download from [VideoInfer](https://www.dropbox.com/scl/fo/9mcd1yrf8ca8b5heziqz4/AKfHt8pYjPvi0_kQUk8hx9o?rlkey=e7p4d0v3e2zuih7rbsuynrmd0&st=nqd8bhym&dl=0) & [VideoRefer-Bench](https://github.com/DAMO-NLP-SG/VideoRefer?tab=readme-ov-file#%EF%B8%8F-videorefer-bench).
- For general question-answering datasets, you can download from [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) & [LLaVA-Video](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K).

You should replace the absolute path in the code with the actual saved path on your machine.


### VideoInfer Structure

The train/test spliting of [VideoInfer](https://www.dropbox.com/scl/fo/9mcd1yrf8ca8b5heziqz4/AKfHt8pYjPvi0_kQUk8hx9o?rlkey=e7p4d0v3e2zuih7rbsuynrmd0&st=nqd8bhym&dl=0) follows ReVOS to avoid data leakage between segmentation and question-answering.

```bash
VideoInfer-Release
├── frames                        # all images of the train set and test set
├── visual_prompts                # fixed visual prompts for the test set
├── mask_dict.json                # mask dict (train set & test set)
├── train.json                    # QA pairs & masks for generating visual prompts (train set)
└── test.json                     # QA pairs & fixed visual prompts (test set)
```


## Training

Our original training is conducted on 8xH800 (80G) of 2 nodes for about 1 day.

```bash
bash run_torchrun.sh
```

After training, you should merge LoRA weights:

```bash
bash merge.sh
```


## Evaluation

You can check the details of each benchmark in the `evaluation` folder. Before executing the inference and evaluation commands, you may change the codes with the actual dataset paths.

### Video Segmentation

For example, when evaluating on MeViS, you should
```bash
cd RGA3-release

# Step 1
bash evaluation/mevis_val_u/run_inference_mevis.sh

# Step 2
bash evaluation/mevis_val_u/run_eval_mevis.sh
```

**Trouble Shooting**:
The inference script we adopted from VideoLISA may skip some samples, so you may need to execute Step 1 more than once before executing Step 2.

### VideoRefer-Bench<sup>Q</sup>

To evaluate RGA3 on VideoRefer-Bench<sup>Q</sup>, execute following command and the calculated accuracy will be printed.

```bash
bash evaluation/videorefer_bench/run_inference_videorefer.sh
```


### VideoInfer

To evaluate RGA3 on the VideoInfer test split, you should execute the following commands:

```bash
bash evaluation/videoinfer/run_inference_parallel.sh
```
This step will conduct inference and offline metric calculation, such as BLEU-4, saving predicted answers and ground truth answers. Afterwards, to obtain GPT4 accuracy/score, you can refer to `eval_gpt.ipynb`, where we implement the evaluation through OpenAI batch inference. However, you can re-implement it while keeping the original prompt and model version according to your API provider.

We also provide the evaluation scripts of several baseline methods in the `baselines` folder.


## Citation

If you find this paper or repo helpful, you can use the following format to cite:
```bibtex
@inproceedings{wang2025object,
  title={Object-centric Video Question Answering with Visual Grounding and Referring},
  author={Wang, Haochen and Chen, Qirui and Yan, Cilin and Cai, Jiayin and Jiang, Xiaolong and Hu, Yao and Xie, Weidi and Gavves, Stratis},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```


## 🫡 Acknowledgements

- Our codes are based on [LISA](https://github.com/dvlab-research/LISA/) & [VideoLISA](https://github.com/showlab/VideoLISA/). The copyright for adding language embedding in SAM2 belongs to [Sa2VA](https://github.com/magic-research/Sa2VA). The implementation of generating and processing visual prompts is based on [ViP-LLaVA](https://github.com/WisconsinAIVision/ViP-LLaVA).

- We also thank the open-source projects like [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [CoTracker3](https://github.com/facebookresearch/co-tracker) and [SAM2](https://github.com/facebookresearch/sam2).
