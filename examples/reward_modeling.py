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

import contextlib
import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Literal

import transformers

from alpaca_farm import common, constants, data_utils, logging, utils
from alpaca_farm.models import reward_model
from alpaca_farm.reward_modeling_trainer import Trainer, compute_reward_modeling_metrics

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Name of or path to the base generative LM."},
    )


@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: str = field(
        #Literal["alpaca_human_preference", "alpaca_gpt4_preference", "alpaca_noisy_multi_preference"] = field(
        default="alpaca_noisy_multi_preference",
        metadata={"help": "Name of the dataset. Fetches the human or GPT-4 preference data."},
    )
    eval_size: int = field(
        default=500,
        metadata={"help": "Number of examples to split out from training to use for evaluation."},
    )
    prompt_dict_path: str = field(
        default=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
        metadata={"help": "Path to the dictionary for the prompt to format examples."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pad_token: str = field(default=constants.DEFAULT_PAD_TOKEN)
    cache_dir: str = field(default=constants.DEFAULT_CACHE_DIR)
    wandb_project: str = field(default=constants.WANDB_PROJECT)
    flash_attn: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    use_sft_loss: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    label_names: List[str] = field(
        default_factory=lambda: ["index_0", "index_1", "choice"],
        metadata={
            "help": "Names of the labels in the dataset. "
            "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
            "By default, the trainer throws away columns it doesn't recognize when creating the "
            "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    initialize_model_on_cpu: bool = field(
        default=False,
        metadata={
            "help": "Whether to initialize the model on CPU. "
            "If True, models on all processes will be first initialized on CPU; this is RAM-costly but faster."
        },
    )
    end_sequence_with_eos: bool = field(
        default=False,
        metadata={
            "help": "Whether to end sequences with EOS. "
            "Ending with EOS might help the reward model realize it's time to predict."
        },
    )
    resume_from_checkpoint: bool = field(default=False, metadata={"help": "If True, loads from last check point."})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Use fast tokenizer if True. "
            "Fast LLaMA tokenizer forces protobuf downgrade to 3.20.3. "
            "Use fast tokenizer only if you can live with that."
        },
    )


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ["WANDB_PROJECT"] = training_args.wandb_project

    if training_args.deepspeed is not None:
        ctx_mgr = contextlib.nullcontext()
        device_map = None
        low_cpu_mem_usage = None
    elif training_args.initialize_model_on_cpu:
        ctx_mgr = contextlib.nullcontext()
        device_map = None
        low_cpu_mem_usage = True
    else:
        ctx_mgr = common.staggered_object_creation(
            local_rank=training_args.local_rank, world_size=training_args.world_size
        )
        device_map = {"": training_args.device.index}
        low_cpu_mem_usage = True

    with ctx_mgr:
        config = reward_model.RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)
        model = reward_model.RewardModel(
            flash_attn=training_args.flash_attn,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            config=config,
            trust_remote_code=True,
        )
        common.let_model_save_mem_when_zero_grad(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",  # Ensure reward is always extracted at the last token embedding.
        use_fast=training_args.use_fast_tokenizer,
        trust_remote_code=True,
    )
    utils.fix_llama_tokenizer(tokenizer=tokenizer)
    tokenizer.padding = training_args.padding
    data_module = data_utils.make_binary_reward_modeling_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )
    print('data_size', len(data_module['train_dataset']))
    for idx, e in enumerate(data_module['train_dataset']):
        print(e)
        print(tokenizer.decode(e['input_ids'][0]))
        if idx > 10:
            break

    if training_args.use_sft_loss:
        model.use_sft_loss = True

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_reward_modeling_metrics,
        **data_module,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    logger.warning("hooray! training finished successfully! now on to model saving.", main_process_only=True)

    trainer.evaluate()

    trainer.save_state()
    common.safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    logger.warning("hooray again! model saving worked.", main_process_only=True)


if __name__ == "__main__":
    main()
