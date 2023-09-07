import contextlib
import copy
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.modeling_utils import unwrap_model
from transformers.utils import is_apex_available
from transformers import Trainer
import torch
from torch import nn

if is_apex_available():
    from apex import amp


class FsdpTrainer(Trainer):
    def _wrap_model(self, model, training=True, dataloader=None):
        print('Huggingface trainer hack for torch fsdp support')
        if self.args.use_ipex:
            dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
            model = self.ipex_optimize_model(model, training, dtype=dtype)
        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex and training:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization) / 8bit models does not support DDP
        if self.args.n_gpu > 1 and not getattr(model, "is_loaded_in_8bit", False):
            model = nn.DataParallel(model)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        if self.fsdp is not None: 
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                #from torch_xla.distributed.fsdp import checkpoint_module
                from torch.distributed.fsdp.wrap import (
                    always_wrap_policy,
                    size_based_auto_wrap_policy,
                    transformer_auto_wrap_policy,
                )
                print('willyshard fsdp')
            except ImportError:
                print('willyshard nofsdp')
                raise ImportError("Missing XLA FSDP related module; please make sure to use torch-xla >= 2.0.")
            auto_wrap_policy = None
            auto_wrapper_callable = None
            #if self.args.fsdp_config["fsdp_min_num_params"] > 0:
            transformer_cls_to_wrap = set()
            if False:
                auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy, min_num_params=100,
                    #self.args.fsdp_config["fsdp_min_num_params"]
                )
                auto_wrap_policy = always_wrap_policy
            else:
                for layer_class in self.args.fsdp_config["fsdp_transformer_layer_cls_to_wrap"]:
                    transformer_cls = get_module_class_from_name(model, layer_class)
                    if transformer_cls is None:
                        raise Exception("Could not find the transformer layer class to wrap in the model.")
                    else:
                        transformer_cls_to_wrap.add(transformer_cls)
                print(transformer_cls_to_wrap)
                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    # Transformer layer class to wrap
                    transformer_layer_cls=transformer_cls_to_wrap,
                )
            fsdp_kwargs = self.args.fsdp_config
            print(fsdp_kwargs)
            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                BackwardPrefetch,
                CPUOffload,
                FullOptimStateDictConfig,
                FullStateDictConfig,
                ShardingStrategy,
                StateDictType,
            )


            from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
            dtype = torch.float16
            mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
            sharding_strategy = ShardingStrategy(1)

            if True:
                import transformers
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, offload_wrapper, apply_activation_checkpointing
                if False:
                    rank = torch.distributed.get_rank()
                    world_size = torch.distributed.get_world_size()
                    torch.distributed.rpc.init_rpc("worker_{}".format(rank), rank=rank, world_size=world_size)
                    from torch.distributed.pipeline.sync import Pipe
                    model.model.layers = Pipe(model.model.layers, chunks=8, checkpoint='always')
                #check_fn = lambda x: isinstance(x, transformers.models.llama.modeling_llama.LlamaDecoderLayer)
                check_fn = lambda x: isinstance(x, tuple(transformer_cls_to_wrap))
                apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)

            # Wrap the base model with an outer FSDP wrapper
            self.model = model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                cpu_offload=CPUOffload(offload_params=False),
                mixed_precision=mixed_precision_policy,
                limit_all_gathers=True,
                use_orig_params=True,
                sharding_strategy=sharding_strategy,
                #auto_wrapper_callable=auto_wrapper_callable,
                #**fsdp_kwargs,
            )

        return model

