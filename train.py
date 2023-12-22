import os
import torch
import gc

from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType, 
    prepare_model_for_kbit_training
)

from transformers import Trainer, TrainingArguments, set_seed

from transformers.trainer_utils import get_last_checkpoint

from src.utils import get_datasets, get_model_and_tokenizer, parse_args
from src.data.dataset import DataPreprocessor, DataCollator
from src.metrics.competition_metric import mcrmse


def post_process_model(model):
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = prepare_model_for_kbit_training(model)

    return model


def apply_lora(model, lora_args):
    lora_config = LoraConfig(
        **lora_args,
        task_type=TaskType.SEQ_CLS
    )

    lora_model = get_peft_model(model, lora_config)

    lora_model.print_trainable_parameters()

    return lora_model


def load_checkpoint(training_args):
    """ Loads the latest checkpoint for resuming training.

    Args:
        training_args: TrainingArguments.
    Returns:
        checkpoint:
            * `None` if there is no checkpoint found.
            * (str): The path to the checkpoint.
    """

    if not os.path.isdir(training_args.output_dir):
        return None
    
    return get_last_checkpoint(training_args.output_dir)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    
    return logits

def compute_metrics(label_column_names):
    def forward(eval_preds):
        preds, labels = eval_preds

        preds = torch.tensor(preds)
        labels = torch.tensor(labels)

        mcrmse_loss = mcrmse(preds, labels)

        l1_colwise_loss = torch.mean(torch.abs(preds - labels), dim=0).tolist()

        return {
            "mcrmse": mcrmse_loss,
            **{f"{k}_l1_loss": v for k, v in zip(label_column_names, l1_colwise_loss)}
        }
    
    return forward


def main():
    """ Main function for fine-tuning the model. """

    # 1. Parse arguments
    args = parse_args()
    model_args = args["model"]
    lora_args = args["lora"]
    data_args = args["data"]
    training_args = TrainingArguments(**args["training"])

    # 2. Set seed before initializing model.
    set_seed(training_args.seed)

    # 3. Get the datasets
    datasets = get_datasets(data_args)

    # 4. Load pretrained model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_args)

    # 4.1 Post-process the model
    model = post_process_model(model)

    # 5. Apply LoRA
    model = apply_lora(model, lora_args)

    # 6. Preprocess datasets
    span_pooling_config = model_args.span_pooling_config
    data_preprocessor = DataPreprocessor(
        tokenizer, 
        data_args.max_len,
        span_pooling_config.span_type,
        span_pooling_config.ignored_span_id,
    )
    tokenized_datasets = data_preprocessor(datasets, data_args.text_column)

    # 7. Load checkpoint
    checkpoint = load_checkpoint(training_args)

    # 8. Initialize the trainer
    data_collator = DataCollator(
        tokenizer, 
        span_pooling_config.ignored_span_id, 
        data_args.label_column_names
    )

    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=tokenized_datasets["train"] if data_args.train_file else None,
        eval_dataset=tokenized_datasets["eval"] if data_args.eval_file else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics(data_args.label_column_names),
    )

    # 9. Train the model
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)

    # 10. Clean up
    torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()


if __name__ == "__main__":
    main()
