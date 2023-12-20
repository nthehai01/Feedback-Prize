from datasets import load_dataset

from transformers import AutoTokenizer

from src.model.modeling import FeedbackModel


def get_datasets(data_args):
    """ Gets the datasets for evaluation and/or training.

    Args:
        data_args: DataTrainingArguments.
    Returns:
        datasets (DatasetDict): The datasets with the keys of "train" (optional)
            and "eval".
    """

    # Load datasets from local files
    train_file = data_args.train_file
    eval_file = data_args.eval_file

    data_files = {}

    if train_file:
        data_files["train"] = train_file
    if eval_file:
        data_files["eval"] = eval_file

    file_extension = (
        train_file.split(".")[-1] if train_file 
        else eval_file.split(".")[-1]
    )
    if file_extension == "jsonl":
        file_extension = "json"

    datasets = load_dataset(file_extension, data_files=data_files)

    return datasets


def get_model_and_tokenizer(model_args):
    model = FeedbackModel.from_pretrained(**model_args)
    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)

    return model, tokenizer
