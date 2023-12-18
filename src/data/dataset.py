import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from src.data.data_utils import get_span_from_text

class DataPreprocessor:
    def __init__(self, tokenizer, max_len, span_type, ignore_span_id):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.span_type = span_type
        self.ignore_span_id = ignore_span_id

    def _preprocess_text(self, text_batch):
        return text_batch

    def _tokenize_dataset(self, text_batch):
        return self.tokenizer(
            text_batch, 
            max_length=self.max_len,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=True,
            return_length=True
        )

    def _get_span_ids(self, text_batch, offset_mapping_batch):
        def get_span_offset(text):
            return get_span_from_text(text, self.span_type)[1]
        
        def get_token_span_id(token_offset, spans):
            token_start, token_end = token_offset

            if token_start == token_end:
                return self.ignore_span_id

            for spans_id, (span_start, span_end) in enumerate(spans):
                if token_start >= span_start and token_end <= span_end:
                # if min(span_end, token_end) - max(span_start, token_start) > 0:
                    return spans_id
            
            # Return ignore_span_id if the token is not in any span.
            return self.ignore_span_id
        
        toke_span_ids_batch = []
        for text, offset_mapping in zip(text_batch, offset_mapping_batch):
            span_offset = get_span_offset(text)
            token_span_ids = [
                get_token_span_id(token_offset, span_offset)
                for token_offset in offset_mapping
            ]

            toke_span_ids_batch.append(token_span_ids)

        return {"span_ids": toke_span_ids_batch}

    def __call__(self, dataset, text_column):
        def process(example_batch):
            text_batch = example_batch[text_column]

            text_batch = self._preprocess_text(text_batch)
            
            res = self._tokenize_dataset(text_batch)
            res.update(
                self._get_span_ids(text_batch, res["offset_mapping"])
            )

            return res
        
        return dataset.map(process, batched=True)

class DataCollator:
    def __init__(self, 
                 tokenizer, 
                 span_pooling_config, 
                 label_column_names=None):
        self.tokenizer = tokenizer
        self.label_column_names = label_column_names
        self.span_pooling_config = span_pooling_config

    def __call__(self, examples):
        """ A data sample has:
                `input_ids`.
                `attention_mask`.
                `span_ids`: tokens are grouped into spans, tokens within the same 
                    span have the same span_id.
                `labels`: ground truth labels for six analytic measures.
        """

        pd_examples = pd.DataFrame(examples).to_dict(orient="list")

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token else self.tokenizer.eos_token_id

        input_ids = pad_sequence(
            [torch.tensor(input_ids) for input_ids in pd_examples["input_ids"]],
            batch_first=True,
            padding_value=pad_token_id
        )
        attention_mask = input_ids.ne(pad_token_id)

        span_ids = pad_sequence(
            [torch.tensor(span_ids) for span_ids in pd_examples["input_ids"]],
            batch_first=True,
            padding_value=self.span_pooling_config.ignore_id
        )
        
        if self.label_column_names and self.label_column_names in pd_examples:
            labels = torch.tensor(pd_examples[self.label_column_names].values)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "span_ids": span_ids,
                "labels": labels
            } 

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "span_ids": span_ids
        } 
