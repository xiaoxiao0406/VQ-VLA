
# VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2507.01016-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2507.01016)
[![Project](https://img.shields.io/badge/Project-Page-orange.svg?style=for-the-badge)](https://xiaoxiao0406.github.io/vqvla.github.io/)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/VQ-VLA)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

**ICCV 2025**

[Yating Wang](https://scholar.google.com/citations?hl=zh-CN&user=5SuBWh0AAAAJ), [Haoyi Zhu](https://www.haoyizhu.site/), [Mingyu Liu](https://mingyulau.github.io/), [Jiange Yang](https://yangjiangeyjg.github.io/),  [Hao-Shu Fang](https://fang-haoshu.github.io/), [Tong He](http://tonghe90.github.io/)
<!-- <hr style="border: 2px solid gray;"></hr> -->
</div>

![teaser](assets/vqvla_pipeline.png)

**VQ-VLA** is an innovative vector quantization based action tokenizer built upon the largest-scale action trajectory dataset to date, leveraging over 100 times more data than previous approaches. It demonstrates that action tokenizers can be effectively scaled by leveraging large-scale simulated action data. We prove that our action tokenizers improve the performance, inference speed, and long-horizon capabilities of
VLA models.

## :clipboard: Contents
- [Installation](#hammer-installation)
- [Fine-Tuning VQ-VLA via LoRA](#fire-fine-tuning-vq-vla-via-lora)
- [VQ-VLA Evaluation (LIBERO)](#rocket-vq-vla-evaluation-libero)
- [Acknowledgements](#sparkles-acknowledgements)
- [License](#books-license)
- [Citation](#pencil-citation)

## :hammer: Installation
1. Setting up VQ-VLA Training Environment
```bash
# create conda environment
conda create -n vqvla python=3.10 -y
conda activate vqvla

# install PyTorch (adjust for your CUDA version)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# clone project and install the vqvla repo
git clone https://github.com/xiaoxiao0406/VQ-VLA.git
cd vqvla
pip install -e .

# install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```
2. Setting up VQ-VLA Evaluation Environment (LIBERO)
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .

cd vqvla
pip install -r experiments/robot/libero/libero_requirements.txt
```

## :fire: Fine-Tuning VQ-VLA via LoRA
### Step 0: Download Dataset (Optional)
Download the Libero-90 RLDS dataset using the following command:
```
huggingface-cli download --resume-download --repo-type dataset VQ-VLA/libero_90_rlds --local-dir <YOUR_DATA_DIRECTORY>
```
Replace <YOUR_DATA_DIRECTORY> with your desired data storage path
**Note:** If you want to train with your own dataset, you can convert your data to RLDS format by following the code at: [rlds_dataset_builder](https://github.com/moojink/rlds_dataset_builder)
### Step 1: Training VQ
```
bash scripts/train_action_vqvae.sh <TRAIN_DATASET_NAME> <WANDB_NAME> <YOUR_DATA_DIRECTORY>

# For example：
bash scripts/train_action_vqvae.sh libero_90_no_noops train_vq_libero_90 <YOUR_DATA_DIRECTORY>
```

### Step 2: Finetune VQ-VLA 
We use LoRA (Low-Rank Adaptation) to fine-tune the VLA model with a total batch size of 16:
```
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_vqvla.py \
  --vla_path openvla/openvla \  
  --data_root_dir <YOUR_DATA_DIRECTORY> \
  --dataset_name <DATASET_NAME> \
  --run_root_dir <PATH_TO_LOG/CHECKPOINT_DIRECTORY> \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --max_steps 400000 \
  --checkpoint_path <VQ_CHECKPOINT_DIRECTORY>
```
We have fine-tuned OpenVLA on Libero-90 dataset. The model weights are available on Hugging Face: [VQ-VLA/openvla-7b-finetuned-libero-90](https://huggingface.co/VQ-VLA/openvla-7b-finetuned-libero-90)

## :rocket: VQ-VLA Evaluation (LIBERO)

We train an action tokenizer (using the largest version of the datasets) using data from datasets: [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment), [RH20T](https://rh20t.github.io/), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [ManiSkill](https://github.com/haosulab/ManiSkill), [RLBench](https://github.com/stepjam/RLBench). The trained action tokenizer weights and VQ-VLA model fine-tuned on LIBERO-90 are available on [Hugging Face](https://huggingface.co/VQ-VLA/vq-vla-weight).
```bash
huggingface-cli download --resume-download VQ-VLA/vq-vla-weight --local-dir <YOUR_WEIGHT_DIRECTORY>

# LIBERO-90 eval
python experiments/robot/libero/run_libero_eval_vq_vla.py 
  --pretrained_checkpoint "<YOUR_WEIGHT_DIRECTORY>/vq-vla-weight/vqvla_weight" \
  --task_suite_name "libero_90" \
  --vqvae_ckpt "<YOUR_WEIGHT_DIRECTORY>/vq-vla-weight/action_tokenizer_weight/all_data_vq.pth"
```

## :sparkles: Acknowledgements
Our work is primarily built upon [OpenVLA](https://github.com/openvla/openvla), [Pyramid Flow](https://github.com/jy0205/Pyramid-Flow), [VQ-BeT](https://github.com/jayLEE0301/vq_bet_official), [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch), [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment), [RH20T](https://rh20t.github.io/), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [ManiSkill](https://github.com/haosulab/ManiSkill), [RLBench](https://github.com/stepjam/RLBench).

## :books: License
This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. For any questions, please email to tonghe90[at]gmail[dot]com.

## :pencil: Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2507.01016):

```bibtex
@inproceedings{wang25vqvla,
      title={VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers},
      author={Yating Wang, Haoyi Zhu, Mingyu Liu, Jiange Yang, Hao-Shu Fang, Tong He},
      booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
      year={2025}
}
```
