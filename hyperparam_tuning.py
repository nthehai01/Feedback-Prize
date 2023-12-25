import os
import optuna
import torch
import gc

from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType, 
    prepare_model_for_kbit_training
)

from transformers import (
    Trainer,
    set_seed,
    TrainingArguments
)

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


def objective(trial):
    hidden_dropout_prob = trial.suggest_categorical("hidden_dropout_prob", [0.0, 0.05])
    # attention_probs_dropout_prob = trial.suggest_categorical("attention_probs_dropout_prob", [0.0, 0.1])

    lora_r = trial.suggest_categorical("lora_r", [2, 4, 8])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32])

    learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 3e-4])
    

    # 1. Parse arguments
    args = parse_args()
    model_args = args["model"]
    lora_args = args["lora"]
    data_args = args["data"]
    training_args = TrainingArguments(**args["training"])

    # 1.1 Update hyperparameters
    model_args.hidden_dropout_prob = hidden_dropout_prob
    # model_args.attention_probs_dropout_prob = attention_probs_dropout_prob

    lora_args.r = lora_r
    lora_args.lora_alpha = lora_alpha

    training_args.learning_rate = learning_rate

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
    # checkpoint = load_checkpoint(training_args)

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
        data_collator=data_collator,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics(data_args.label_column_names),
    )

    # 9. Train the model
    trainer.train()

    # 10. Evaluate the model
    stats = trainer.evaluate(tokenized_datasets["eval"])

    # 11. Clean up
    torch.cuda.empty_cache()
    del model, tokenizer, trainer
    gc.collect()

    return stats["eval_mcrmse"]


def main():
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.RandomSampler(seed=42)
    )

    study.optimize(objective, n_trials=20)

    print("BEST PARAMS", study.best_params)

    f = optuna.visualization.plot_optimization_history(study)
    f.write_image("/kaggle/working/optuna_optimization_history.png")

    f = optuna.visualization.plot_parallel_coordinate(study)
    f.write_image("/kaggle/working/optuna_parallel_coordinate.png")

    f = optuna.visualization.plot_slice(study)
    f.write_image("/kaggle/working/optuna_slice.png")

    f = optuna.visualization.plot_param_importances(study)
    f.write_image("/kaggle/working/optuna_param_importances.png")


if __name__ == "__main__":
    main()
    