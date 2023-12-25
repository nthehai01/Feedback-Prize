import torch
import gc
from tqdm import tqdm
import numpy as np

from peft import PeftModel

from transformers import set_seed

from src.utils import get_datasets, get_model_and_tokenizer, parse_args
from src.data.dataset import DataPreprocessor, DataCollator


def load_lora_model(model, peft_model_id):
    model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16)
    model = model.merge_and_unload()
    
    model.half()

    return model


def post_process(outputs):
    return np.round(outputs * 2) / 2


def infer(model, tokenized_datasets, data_collator):
    model.eval()

    data_size = len(tokenized_datasets)

    res = []
    with torch.no_grad():
        for i in tqdm(range(data_size)):
            example = tokenized_datasets[i]
            example = {k: [v] for k, v in example.items()}

            inputs = data_collator(example)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs, return_dict=True).logits
            outputs = outputs.detach().cpu()
            # outputs = post_process(outputs)
            
            res.extend(outputs.tolist())

    return res


def main():
    """ Main function for fine-tuning the model. """

    # 1. Parse arguments
    args = parse_args()
    model_args = args["model"]
    data_args = args["data"]

    # 2. Set seed
    set_seed(42)

    # 3. Get the datasets
    datasets = get_datasets(data_args)

    # 4. Load pretrained model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_args)

    # 5. Load LoRA checkpoint
    model = load_lora_model(model, model_args.peft_model_id)

    # 6. Preprocess datasets
    span_pooling_config = model_args.span_pooling_config
    data_preprocessor = DataPreprocessor(
        tokenizer, 
        data_args.max_len,
        span_pooling_config.span_type,
        span_pooling_config.ignored_span_id,
    )
    tokenized_datasets = data_preprocessor(datasets, data_args.text_column)
    tokenized_datasets = tokenized_datasets["eval"]

    # 7. Inference
    data_collator = DataCollator(
        tokenizer, 
        span_pooling_config.ignored_span_id, 
        data_args.label_column_names
    )
    outputs = infer(model, tokenized_datasets, data_collator)

    # 8. Export results
    output_df = datasets["eval"].to_pandas()
    output_df[data_args.label_column_names] = outputs
    
    temp = ["text_id", "cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    output_df[temp].to_csv("/kaggle/working/submission.csv", index=False)

    # 9. Clean up
    torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()


if __name__ == "__main__":
    main()
