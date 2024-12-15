# -*- encoding: utf-8 -*-
# File: dataset.py
# Description: None

import os

import numpy as np
from regex import F
import torch
from torch.utils.data import Dataset


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data_list,
        processor,
        process_func,
        dataset_prefix="",
    ):
        super(SupervisedDataset, self).__init__()
        self.raw_data_list = raw_data_list
        self.processor = processor
        self.process_func = process_func
        self.dataset_prefix = dataset_prefix

    def __len__(self):
        return len(self.raw_data_list)

    def check_ret(self, ret):
        flag = True
        for key in ret.keys():
            value_list = ret[key]
            if not isinstance(value_list, list):
                value_list = [value_list]
            for value in value_list:
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any():
                        flag = False
                    if torch.isinf(value).any():
                        flag = False
        return flag

    def check_audio(self, ret):
        flag = True
        for audio in ret["msgs_audio"]:
            if (audio["input_audio_lengths"][:, 1] == 0).any():
                flag = False
        return flag

    def prepare_labels(self, data):

        def prepare_labels(tokenizer, input_ids, padding_value=-100):
            # <|role_start|>assistant<|role_end|> 后面的内容才是需要算loss的部分
            def find_start_header_idxs():
                start_header_tokens = tokenizer.encode("<|role_start|>assistant<|role_end|>", add_special_tokens=False)
                start_header_idxs = np.where(input_ids == start_header_tokens[-1])[0]

                kept_start_header_idxs = []
                for start_header_idx in start_header_idxs:
                    keep = True
                    for i in range(1, len(start_header_tokens)):
                        if start_header_tokens[-(i + 1)] != input_ids[start_header_idx - i]:
                            keep = False
                            break
                    if keep:
                        kept_start_header_idxs.append(start_header_idx)
                return kept_start_header_idxs

            turn_end_token_id = tokenizer.encode("<|turn_end|>")[0]
            start_header_idxs = find_start_header_idxs()
            end_header_idxs = np.where(input_ids == turn_end_token_id)[0]
            label_mask = np.zeros_like(input_ids, dtype=np.bool_)

            def find_next_greater_number(lst, num):
                next_greater = None
                for n in lst:
                    if n > num:
                        if next_greater is None or n < next_greater:
                            next_greater = n
                return next_greater

            nr_tokens = len(input_ids)
            for start_head_idx in start_header_idxs:
                start_idx = start_head_idx + 1
                end_idx = find_next_greater_number(end_header_idxs, start_head_idx)
                end_idx = min(end_idx + 1, nr_tokens)
                label_mask[start_idx:end_idx] = True

            labels = torch.ones(input_ids.shape[0] + 1) * padding_value
            labels[: input_ids.shape[0]] = input_ids
            labels[: input_ids.shape[0]][~label_mask] = padding_value
            labels = labels[1:]
            return labels.long()

        return prepare_labels(self.processor.tokenizer, data["input_ids"])

    def add_dataset_prefix(self, item):
        conv = item["conversations"]
        for i in range(len(conv)):
            content = conv[i]["content"]
            if "image" in content:
                content["image"] = os.path.join(self.dataset_prefix, content["image"])
            if "audio" in content:
                content["audio"] = os.path.join(self.dataset_prefix, content["audio"])

        return conv

    def __getitem__(self, i):
        raw_data_item = self.raw_data_list[i]
        item = self.add_dataset_prefix(raw_data_item)
        processed_data = self.processor(
            item,
            add_generation_prompt=False,
            apply_data_collator=False,
        )
        if "labels" not in processed_data:
            processed_data["labels"] = self.prepare_labels(processed_data)

        return processed_data
