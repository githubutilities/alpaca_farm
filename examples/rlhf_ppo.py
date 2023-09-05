# Copyright 2023 The Alpaca Team
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

import os
import sys

import transformers
from accelerate import DistributedDataParallelKwargs

from alpaca_farm import accelerate_patch, data_utils, logging
from alpaca_farm.rl.ppo_trainer import PPOTrainer, make_models, make_tokenizer
from alpaca_farm.rl.ppo_utils import DataArguments, TrainingArguments

logger = logging.get_logger(__name__)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = transformers.HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()

    if True:
        from accelerate.utils import FullyShardedDataParallelPlugin
        fsdp_plugin = FullyShardedDataParallelPlugin()
        fsdp_plugin.set_mixed_precision('fp16')
        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
        fsdp_plugin.cpu_offload = CPUOffload(offload_params=training_args.cpu_offload)

        from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig, LocalStateDictConfig, LocalOptimStateDictConfig, ShardedStateDictConfig

        fsdp_plugin.state_dict_config = ShardedStateDictConfig(offload_to_cpu=True, use_dtensor=True)
        def my_import(name):
            components = name.split('.')
            mod = __import__(components[0])
            for comp in components[1:]:
                mod = getattr(mod, comp)
            return mod
        if True:
            print('accelerate enable offloading and llamaattention embedding fsdp')
            transformer_cls_to_wrap = set()
            import functools
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            layer_class_names = training_args.wrap_layer_name.split(",")
            for layer_class in layer_class_names:
                if layer_class.strip() == '':
                    continue
                transformer_cls = my_import(layer_class)
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
            def my_auto_wrap_policy(module, recurse, nonwrapped_numel):
                ret = transformer_auto_wrap_policy(module, recurse, nonwrapped_numel, transformer_cls_to_wrap)
                #ret = isinstance(module, tuple(transformer_cls_to_wrap))
                #print('wrap_policy', module, recurse, module.__class__.__name__, ret)
                return ret

            def custom_auto_wrap_policy(
                module,
                recurse,
                nonwrapped_numel,
                # Additional custom arguments
                min_num_params=int(1e8),
            ) -> bool:
                return nonwrapped_numel >= min_num_params
            #my_auto_wrap_policy = functools.partial(custom_auto_wrap_policy, min_num_params=int(1e5))
            fsdp_plugin.auto_wrap_policy = my_auto_wrap_policy

    accelerator = accelerate_patch.MyAccelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with=["wandb"],
        even_batches=True,  # Make sure the batch size on each device is the same.
        split_batches=False,  # Don't break a batch into smaller chunks.
        step_scheduler_with_optimizer=False,  # Untie optimizer and scheduler step.
        fsdp_plugin=fsdp_plugin,
        # Value model might not use all parameters (e.g., lm-head) in the forward pass.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    accelerator.init_trackers(
        training_args.wandb_project,
        init_kwargs={"wandb": {"name": training_args.run_name}},
        config=training_args.__dict__,
    )
    logger.warning(accelerator.state, main_process_only=False)  # Each process log their own state.

    tokenizer: transformers.PreTrainedTokenizer = make_tokenizer(args=training_args)
    model_module: dict = make_models(tokenizer=tokenizer, args=training_args, accelerator=accelerator)
    data_module: dict = data_utils.make_rl_data_module(
        tokenizer=tokenizer, data_args=data_args, training_args=training_args
    )

    trainer = PPOTrainer(
        args=training_args,
        accelerator=accelerator,
        **data_module,
        **model_module,
        tokenizer=tokenizer,
    )
    if training_args.do_export:
        step_list = [int(training_args.save_steps * i) for i in range(10)] + [1]
        for step in step_list:
            src_dir = training_args.output_dir + '/checkpoint-{}/local'.format(step)
            output_dir = training_args.output_dir + '/checkpoint-{}/full'.format(step)
            if os.path.exists(src_dir):
                print("Converting {} to {}".format(src_dir, output_dir))
                trainer.load_and_save_model(
                    src_dir=src_dir,
                    output_dir=output_dir,
                )
        sys.exit(0)
    trainer.train()


if __name__ == "__main__":
    main()
