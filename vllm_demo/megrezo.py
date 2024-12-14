# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only MegrezO model compatible with HuggingFace weights."""

from functools import lru_cache
from functools import partial
from typing import Any, Callable, Iterable, List, Literal, Mapping, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.types
from PIL import Image
from torch import Tensor
from torch import nn
from torch.nn.init import trunc_normal_
from transformers import PretrainedConfig
from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig
from vllm.config import MultiModalConfig
from vllm.inputs import INPUT_REGISTRY
from vllm.inputs import DecoderOnlyInputs
from vllm.inputs import InputContext
from vllm.inputs import token_inputs
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.resampler import get_2d_sincos_pos_embed
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import VllmModelForTextGeneration
from vllm.model_executor.models.idefics2_vision_model import Idefics2VisionTransformer
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import LLMWrapper
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import IntermediateTensors
from vllm.sequence import SequenceData
from vllm.transformers_utils.processor import get_processor

RawImageType = Union[Image.Image, torch.Tensor]
RawAudioType = Union[bytes, torch.Tensor]

cached_get_processor = lru_cache(get_processor)


class MegrezORawImageInput(TypedDict):
    """Input mapper input with auxiliary data for computing image bounds."""

    image: RawImageType


class MegrezOAudioInput(TypedDict):
    type: Literal["audio"]

    data: RawAudioType


class MegrezOAudioTensorInput(TypedDict):
    type: Literal["audio_tensor"]

    input_audios: torch.Tensor
    input_audio_lengths: torch.Tensor
    audio_span_tokens: torch.Tensor


class MegrezOImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """
    Shape: `(batch_size * num_images, num_channels, height, width)`

    Note that the image size may vary, so we pass it as a list
    instead of a batched tensor.
    """

    tgt_sizes: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(height, width)` format.
    """

    patch_attention_mask: torch.Tensor
    """
    Shape: `(batch_size * num_images, num_patches, num_patches)`
    """


class MegrezOImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """
    Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    instead of a batched tensor.
    """

    image_bounds: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(start, stop)` format.
    """


def insert_audio_embeddings(text_embeddings, inserted_embeddings, inserted_bounds):

    inserted_bounds = inserted_bounds.long()

    for idx in range(len(inserted_embeddings)):
        bid = inserted_bounds[idx][0]
        start_id = inserted_bounds[idx][1]
        end_id = inserted_bounds[idx][2]
        embedding = inserted_embeddings[idx]
        text_embeddings[start_id + 1 : end_id] = embedding
    return text_embeddings


def insert_image_embeddings(text_embeddings, inserted_embeddings, inserted_bounds):

    inserted_bounds = inserted_bounds.long()
    for idx in range(len(inserted_embeddings)):
        bid = inserted_bounds[idx][0]
        start_id = inserted_bounds[idx][1]
        end_id = inserted_bounds[idx][2]
        embedding = inserted_embeddings[idx]
        text_embeddings[start_id:end_id] = embedding

    return text_embeddings


MegrezOImageInputs = Union[MegrezOImagePixelInputs]
MegrezOAudioInputs = Union[MegrezOAudioTensorInput]

# region: Resampler
DEFAULT_LN = partial(nn.LayerNorm, eps=1e-6)


class Resampler(nn.Module):

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
        max_size: Tuple[int, int] = (70, 70),
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=0.02)
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = ReplicatedLinear(kv_dim, embed_dim, bias=False, quant_config=quant_config, prefix=prefix)
        else:
            # Maintain the same return value with ReplicatedLinear.forward
            self.kv_proj = lambda *args, **kwargs: (  # type: ignore # noqa
                nn.Identity()(*args, **kwargs),
                None,
            )

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        self.do_post_projection = True
        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim**-0.5) * torch.randn(embed_dim, embed_dim))

        self.max_size = max_size
        self._set_2d_pos_cache(self.max_size)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

    def _set_2d_pos_cache(self, max_size: Tuple[int, int], device: torch.types.Device = "cpu") -> None:
        pos_embed_arr = get_2d_sincos_pos_embed(self.embed_dim, max_size, version=(2, 5))
        pos_embed = torch.from_numpy(pos_embed_arr).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes: torch.Tensor, device: torch.types.Device) -> None:
        max_h = tgt_sizes[:, 0].max().item()
        max_w = tgt_sizes[:, 1].max().item()
        assert isinstance(max_h, int) and isinstance(max_w, int)

        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = (
                max(max_h, self.max_size[0]),
                max(max_w, self.max_size[1]),
            )
            self._set_2d_pos_cache(self.max_size, device)

    def forward(self, x: torch.Tensor, tgt_sizes: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        max_patch_len = patch_len.max().item()
        assert isinstance(max_patch_len, int)

        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i].tolist()
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))  # patches * D
            key_padding_mask[i, patch_len[i] :] = True
        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(
            1, 0, 2
        )  # BLD => L * B * D
        x, _ = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x


# endregion

# region: AudioEncoder


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        # return super().forward(x.float()).type(x.dtype)
        return super().forward(x).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk += mask

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        output_dim: int = 512,
        avg_pool: bool = True,
        add_audio_bos_eos_token: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

        if avg_pool:
            self.avg_pooler = nn.AvgPool1d(2, stride=2)
        else:
            self.avg_pooler = None
        self.proj = nn.Linear(n_state, output_dim)
        if add_audio_bos_eos_token:
            self.audio_bos_eos_token = nn.Embedding(2, output_dim)
        else:
            self.audio_bos_eos_token = None
        self.output_dim = output_dim
        self.n_head = n_head

    def forward(self, x: Tensor, padding_mask: Tensor = None, audio_lengths: Tensor = None):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = x.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
        if audio_lengths is not None:
            input_mel_len = audio_lengths[:, 0] * 2
            max_mel_len_in_batch = input_mel_len.max()
            x = x[:, :, :max_mel_len_in_batch]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # B, L, D
        bsz = x.size(0)
        src_len = x.size(1)

        self.input_positional_embedding = self.positional_embedding[:src_len]
        assert (
            x.shape[1:] == self.input_positional_embedding.shape
        ), f"incorrect audio shape: {x.shape[1:], self.input_positional_embedding.shape}"
        x = (x + self.input_positional_embedding).to(x.dtype)
        if padding_mask is not None:
            padding_mask = padding_mask.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
            batch_src_len = padding_mask.size(1)
            x = x[:, :batch_src_len, :]
            padding_mask = padding_mask.view(bsz, -1, batch_src_len)
            padding_mask_ = padding_mask.all(1)
            x[padding_mask_] = 0
            key_padding_mask = (
                padding_mask_.view(bsz, 1, 1, batch_src_len)
                .expand(-1, self.n_head, -1, -1)
                .reshape(bsz, self.n_head, 1, batch_src_len)
            )
            new_padding_mask = torch.zeros_like(key_padding_mask, dtype=x.dtype)
            padding_mask = new_padding_mask.masked_fill(key_padding_mask, float("-inf"))

        for block in self.blocks:
            x = block(x, mask=padding_mask)

        if self.avg_pooler:
            x = x.permute(0, 2, 1)
            x = self.avg_pooler(x)
            x = x.permute(0, 2, 1)

        x = self.ln_post(x)
        x = self.proj(x)

        if self.audio_bos_eos_token is not None:
            bos = self.audio_bos_eos_token.weight[0][None, :]
            eos = self.audio_bos_eos_token.weight[1][None, :]
        else:
            bos, eos = None, None
        return x, bos, eos

    def encode(
        self,
        input_audios: Tensor,
        input_audio_lengths: Tensor,
        audio_span_tokens: List,
    ):
        real_input_audio_lens = input_audio_lengths[:, 0].tolist()
        max_len_in_batch = max(real_input_audio_lens)
        padding_mask = torch.ones([input_audios.size(0), max_len_in_batch]).to(
            dtype=self.conv1.weight.dtype, device=self.conv1.weight.device
        )
        for index in range(len(input_audios)):
            padding_mask[index, : input_audio_lengths[index][0].item()] = 0
        x, bos, eos = self(input_audios, padding_mask, input_audio_lengths)
        output_audios = []
        for i in range(len(audio_span_tokens)):
            audio_span = audio_span_tokens[i]
            audio = x[i][: audio_span - 2]
            if bos is not None:
                audio = torch.concat([bos, audio, eos])
            assert len(audio) == audio_span
            output_audios.append(audio)
        return output_audios


class AudioModel(torch.nn.Module):

    def __init__(self, config):
        super(AudioModel, self).__init__()
        self.config = config
        self.audio = AudioEncoder(**config.audio_config.to_dict())

    def forward(self, audio_info):
        audios = audio_info["input_audios"][0]
        input_audio_lengths = audio_info["input_audio_lengths"][0]
        audio_span_tokens = audio_info["audio_span_tokens"][0]
        audios_features = self.audio.encode(audios, input_audio_lengths, audio_span_tokens)
        return audios_features


# endregion


def get_max_megrezo_image_tokens(ctx: InputContext):
    hf_config = ctx.get_hf_config()
    return getattr(hf_config, "query_num", 64) * 10


def dummy_seq_data_for_minicpmv(seq_len: int, num_images: int):
    return SequenceData.from_prompt_token_counts((0, seq_len))


def dummy_image_for_minicpmv(ctx: InputContext, hf_config: PretrainedConfig, num_images: int):
    width = height = hf_config.vision_config.image_size
    imgs = [MegrezORawImageInput(image=Image.new("RGB", (width, height), color=0)) for _ in range(num_images)]
    return {"image": imgs}


def dummy_data_for_minicpmv(ctx: InputContext, seq_len: int, mm_counts: Mapping[str, int]):
    hf_config = ctx.get_hf_config()
    num_images = mm_counts["image"]

    seq_data = dummy_seq_data_for_minicpmv(seq_len, num_images)
    mm_data = dummy_image_for_minicpmv(ctx, hf_config, num_images)  # skip audio for now
    return (seq_data, mm_data)


def input_processor_for_megrezo(ctx: InputContext, inputs: DecoderOnlyInputs):
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or ("image" not in multi_modal_data and "audio" not in multi_modal_data):
        return inputs

    model_config = ctx.model_config
    tokenizer = cached_get_tokenizer(model_config.tokenizer, trust_remote_code=model_config.trust_remote_code)
    processor = cached_get_processor(model_config.model, trust_remote_code=model_config.trust_remote_code)

    prompt = inputs.get("prompt")
    token_ids = inputs.get("prompt_token_ids")
    if prompt is None:
        prompt = tokenizer.decode(token_ids)

    images = multi_modal_data.get("image")
    audios = multi_modal_data.get("audio")
    prompt, multimodal_inputs = processor.process_multimodal_inputs(
        prompt,
        images=images,
        audios=audios,
        return_tensors="pt",
    )
    text_encodings = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    encodings = processor.merge_encodings(text_encodings, multimodal_inputs)
    data = processor.data_collator([encodings])

    new_prompt = tokenizer.decode(data["input_ids"][0])
    new_multi_modal_data = {
        "image": data["image_encoding"],
        "audio": data["audio_encoding"],
    }

    return token_inputs(
        prompt_token_ids=data["input_ids"][0],
        prompt=new_prompt,
        multi_modal_data=new_multi_modal_data,
    )


def input_mapper_for_megrezo(ctx: InputContext, data: object):
    return MultiModalInputs(data)


@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_megrezo)
@MULTIMODAL_REGISTRY.register_input_mapper("audio", input_mapper_for_megrezo)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens("audio", 3000)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_megrezo_image_tokens)
@INPUT_REGISTRY.register_input_processor(input_processor_for_megrezo)
class MegrezOModel(nn.Module, VllmModelForTextGeneration, SupportsMultiModal, SupportsPP):

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        # All MiniCPM-V models disable `tie_word_embeddings` but
        # `PretrainedConfig.tie_word_embeddings` defaults to True; we cannot
        # check `tie_word_embeddings` until vLLM integrate MiniCPM-V model
        # and config class
        self.config = config
        self.multimodal_config = multimodal_config

        self.llm = self.init_llm(config, cache_config, quant_config, prefix="model")
        self.vision = self.init_vision_module(config, quant_config, prefix="vpm")
        param_dtype = torch.get_default_dtype()
        self.vision.to(dtype=param_dtype)

        self.audio = self.init_audio_module(config, quant_config)
        self.audio.to(dtype=param_dtype)

        self.vision_dim = self.vision.embeddings.embed_dim
        self.embed_dim = self.config.hidden_size
        self.resampler = self.init_resampler(
            self.embed_dim, self.vision_dim, quant_config=quant_config, prefix="vision.resampler"
        )
        self.resampler.to(device="cuda", dtype=param_dtype)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=quant_config, prefix="llm.lm_head"
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

        self.make_empty_intermediate_tensors = self.llm.make_empty_intermediate_tensors

        self._called_cnt = 0

    def get_vision_hidden_states(
        self,
        pixel_values,
        tgt_sizes,
        patch_attn_mask,
    ) -> torch.Tensor:

        device = self.vision.embeddings.position_embedding.weight.device
        dtype = self.vision.embeddings.position_embedding.weight.dtype
        pixel_values = torch.stack([(image.to(device) - 127.5) / 127.5 for image in pixel_values]).type(dtype)
        vision_embedding = self.vision(
            pixel_values.type(dtype),
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )

        return self.resampler(vision_embedding, tgt_sizes)

    def compose_embeddings(self, mini_batch):
        input_ids = mini_batch["input_ids"]
        image_encoding = mini_batch.get("image_encoding")
        audio_encoding = mini_batch.get("audio_encoding")

        embeddings_text = self.llm.model.embed_tokens(input_ids)
        input_embeds = embeddings_text
        if image_encoding:
            pixel_values = image_encoding["pixel_values"][0]
            tgt_sizes = image_encoding["tgt_sizes"][0]
            patch_attention_mask = image_encoding["patch_attention_mask"][0]
            bounds_image = image_encoding["image_bounds"][0]
            device = self.vision.embeddings.position_embedding.weight.device
            dtype = self.vision.embeddings.position_embedding.weight.dtype

            embeddings_image = self.get_vision_hidden_states(
                pixel_values.to(device, dtype),
                tgt_sizes,
                patch_attention_mask.to(device),
            )
            input_embeds = insert_image_embeddings(embeddings_text, embeddings_image, bounds_image)

        if audio_encoding:
            embeddings_audio = self.audio(audio_encoding)
            bounds_audio = audio_encoding["audio_bounds"][0]
            input_embeds = insert_audio_embeddings(embeddings_text, embeddings_audio, bounds_audio)

        return input_embeds

    def _parse_inputs(self, input_ids: torch.Tensor, **kwargs):
        if kwargs.get("pixel_values") is not None:
            image_encoding = {
                "pixel_values": kwargs.get("pixel_values"),
                "tgt_sizes": kwargs.get("tgt_sizes"),
                "patch_attention_mask": kwargs.get("patch_attention_mask"),
                "image_bounds": kwargs.get("image_bounds"),
            }
        else:
            image_encoding = None

        if kwargs.get("input_audios") is not None:
            audio_encoding = {
                "input_audios": kwargs.get("input_audios"),
                "input_audio_lengths": kwargs.get("input_audio_lengths"),
                "audio_span_tokens": kwargs.get("audio_span_tokens"),
                "audio_bounds": kwargs.get("audio_bounds"),
            }
        else:
            audio_encoding = None

        return {
            "input_ids": input_ids,
            "image_encoding": image_encoding,
            "audio_encoding": audio_encoding,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            embeddings = None
        else:
            mini_batch = self._parse_inputs(input_ids, **kwargs)
            embeddings = self.compose_embeddings(mini_batch)

        # always pass the input via `inputs_embeds`
        # to make sure the computation graph is consistent
        # for `torch.compile` integration
        input_ids = None

        output = self.llm(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=embeddings,
        )

        self._called_cnt += 1
        return output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        keys_to_modify_mapping = {
            "llm.lm_head": "lm_head",
            "vision.resampler": "resampler",
        }

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for key_to_modify, new_key in keys_to_modify_mapping.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if "audio.positional_embedding" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                else:
                    print(f"Skipping loading of {name}")

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                else:
                    print(f"Skipping loading of {name}")

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(language_model="llm", connector="resampler", tower_model="vpm")

    def init_llm(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:

        return LLMWrapper(
            LlamaModel(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            name=prefix,
        )

    def init_audio_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        return AudioModel(config)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        model = LLMWrapper(
            Idefics2VisionTransformer(config.vision_config),
            name=prefix,
        )
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]
        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        resampler = Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            quant_config=quant_config,
            prefix=prefix,
        )
        return resampler
