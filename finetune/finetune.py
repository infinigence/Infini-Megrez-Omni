# -*- encoding: utf-8 -*-
# File: finetune.py
# Description: None


import glob
import json
import logging
import os
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from glob import glob
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import transformers
from accelerate.utils import DistributedType
from dataset import SupervisedDataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from trainer import MegrezOTrainer
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers.integrations import deepspeed


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-V-2")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    dataset_prefix: str = field(default="data", metadata={"help": "Prefix for the multimodal data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    tune_vision_encoder: Optional[bool] = field(default=True)
    tune_vision_proj: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=True)
    tune_audio_encoder: Optional[bool] = field(default=True)
    tune_audio_proj: Optional[bool] = field(default=True)
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)
    scale_resolution: Optional[int] = field(default=448)
    remove_unused_columns: Optional[bool] = field(default=False)


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(trainer.model.named_parameters(), bias)
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def make_supervised_data_module(
    data_args,
    processor,
    process_func,
    data_collator=None,
    max_length=2048,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    rank0_print("Loading data...")

    with open(data_args.data_path, "r") as f:
        raw_data_list = [json.loads(line) for line in f]
        train_dataset = SupervisedDataset(
            raw_data_list,
            processor,
            process_func,
            data_args.dataset_prefix,
        )

    eval_dataset = None
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(data_collator, max_length=max_length, collate_labels=True),
    )


def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return {"Total": all_param, "Trainable": trainable_params}


local_rank = 0


def load_model_from_pretrained(model_path, dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, _attn_implementation="flash_attention_2", trust_remote_code=True, torch_dtype=dtype
    )
    return model


def load_tokenizer_from_pretrained(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    return tokenizer


def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))

    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 are not incompatible with QLoRA.")

    model = load_model_from_pretrained(model_args.model_name_or_path, dtype=compute_dtype)
    tokenizer = load_tokenizer_from_pretrained(model_args.model_name_or_path)
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    model.tune_llm = training_args.tune_llm
    model.tune_vision = training_args.tune_vision_encoder or training_args.tune_vision_proj
    model.tune_audio = training_args.tune_audio_encoder or training_args.tune_audio_proj

    if not training_args.tune_vision_encoder:
        model.vision.vpm.requires_grad_(False)
    if not training_args.tune_vision_proj:
        model.vision.resampler.requires_grad_(False)
    if not training_args.tune_llm:
        model.llm.requires_grad_(False)
    if not training_args.tune_audio_encoder:
        model.audio.requires_grad_(False)
        model.audio.audio.proj.requires_grad_(True)
        if model.audio.audio.audio_bos_eos_token is not None:
            model.audio.audio.audio_bos_eos_token.requires_grad_(True)
    if not training_args.tune_audio_proj:
        model.audio.audio.proj.requires_grad_(False)
        if model.audio.audio.audio_bos_eos_token is not None:
            model.audio.audio.audio_bos_eos_token.requires_grad_(False)

    rank0_print(get_parameter_number(model))
    data_module = make_supervised_data_module(
        data_args=data_args,
        processor=processor,
        process_func=None,
        data_collator=processor.data_collator,
        max_length=training_args.model_max_length,
    )
    if training_args.lr_scheduler_type == "cosine_with_min_lr":
        training_args.lr_scheduler_kwargs = {"min_lr_rate": 0.1}
    trainer = MegrezOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    train_dataset = trainer.train_dataset
    nr_data = len(train_dataset)
    rank0_print("nr dataset: {}".format(nr_data))

    checkpoint_path = os.path.join(training_args.output_dir, "checkpoint*")
    checkpoint_paths = sorted(list(glob(checkpoint_path)))

    valid_checkpoint_paths = []
    for checkpoint_path in checkpoint_paths:
        checkpoint_num = checkpoint_path.split("-")[-1]
        if checkpoint_num.isdigit():
            valid_checkpoint_paths.append(checkpoint_path)
    checkpoint_paths = sorted(list(valid_checkpoint_paths))
    checkpoint_paths = sorted(checkpoint_paths, key=lambda x: int(x.split("-")[-1]))
    checkpoint_paths = list(checkpoint_paths)
    load_checkpoint = True

    if load_checkpoint and checkpoint_paths:
        checkpoint_path = checkpoint_paths[-1]
        rank0_print("Continue Checkpoint Training: {}".format(checkpoint_path))
        trainer.train(checkpoint_path)
    else:
        trainer.train()

    trainer.save_state()
    final_path = os.path.join(training_args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    rank0_print("save final path to {}".format(final_path))
    safe_save_model_for_hf_trainer(trainer, final_path)


if __name__ == "__main__":
    train()
