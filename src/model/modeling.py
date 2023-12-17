import torch
import torch.nn as nn

from typing import Optional, Tuple, Union

from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.deberta.modeling_deberta import DebertaModel, DebertaPreTrainedModel

from src.model.pooling import MeanPooling
from src.metrics.loss import FeedbackLoss


NUM_CLASSES = 6


class FeedbackModel(DebertaPreTrainedModel):
    def __init__(self, 
                 backbone_config, 
                 sliding_window_config,
                 span_pooling_config, 
                 final_pooling_config,
                 loss_weights):
        super().__init__(backbone_config)

        self.sliding_window_config = sliding_window_config
        self.loss_weights = loss_weights

        self.deberta = DebertaModel(backbone_config)

        self.span_pooling = MeanPooling(**span_pooling_config) if span_pooling_config.use_span_pooling else None

        self.final_pooling = MeanPooling(**final_pooling_config)

        self.fc = nn.Linear(backbone_config.hidden_size, NUM_CLASSES)

        self.post_init()

    def _sliding_window_encode(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        The `self.sliding_window_config` contains the following keys:
            * `window_size`: The size of the window (similar to `kernel` in CNN).
            * `inner_size`: The overlapping size of two consecutive segments.
            * `n_tokens`: The number of tokens to be taken from a segment if it 
                is not the first segment as we will take all the tokens in this one.

        For the final output, we concatenate the outputs of the first segment and
        the next consecutive segments as follows:
            * For the first segment, we take the all the `window_size` tokens.
            * For the next segments, we take the `n_tokens` tokens
                starting from the `inner_size`-th token.
            * If that is the last segment, we take all the tokens from the 
                `inner_size`-th token to the end of the sequence.
        """

        window_size = self.sliding_window_config.window_size
        inner_size = self.sliding_window_config.inner_size
        n_tokens = self.sliding_window_config.n_tokens

        assert n_tokens + inner_size <= window_size

        first_segment_outputs = self.deberta(
            input_ids[:, :window_size],
            attention_mask[:, :window_size],
        )
        hidden_states = [first_segment_outputs[0]]

        seq_len = input_ids.shape[1]

        start_window = window_size - inner_size
        end_window = start_window + window_size
        end_window = min(end_window, seq_len)

        # Slide the window until the end of the sequence
        while True:
            segment_outputs = self.deberta(
                input_ids[:, start_window:end_window],
                attention_mask[:, start_window:end_window],
            )
            hidden_states.append(
                segment_outputs[0][:, inner_size:inner_size+n_tokens]
            )

            if end_window == seq_len:
                break

            start_window = start_window + n_tokens
            end_window = start_window + window_size
            end_window = min(end_window, seq_len)

        hidden_states = torch.cat(hidden_states, dim=1)

        assert hidden_states.shape[:2] == input_ids.shape

        return hidden_states

    def _span_pooling_forward(self, hidden_states, span_ids, attention_mask):
        def forward(hidden_states, span_ids, attention_mask):
            ids = torch.unique(span_ids)

            outputs = []
            for id in ids:
                if id == -1:
                    continue

                mask = span_ids == id
                temp = self.final_pooling(
                    hidden_states[mask].unsqueeze(0),
                    attention_mask[mask].unsqueeze(0)
                )
                outputs.append(temp)

            outputs = torch.cat(outputs, dim=0)

            assert outputs.shape[0] == ids.shape[0] - 1

            return outputs

        batch_size, _, hidden_size = hidden_states.shape

        assert batch_size == 1, "The batch size MUST BE 1 or it will be crashed if there is variable range of spans for baches if use span pooling."

        outputs = []
        for batch_idx in range(batch_size):
            temp = forward(
                hidden_states[batch_idx], 
                span_ids[batch_idx], 
                attention_mask[batch_idx]
            )
            outputs.append(temp)

        return torch.cat(outputs, dim=0).reshape(batch_size, -1, hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        span_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.sliding_window_config.use_sliding_window:
            hidden_states = self._sliding_window_encode(input_ids, attention_mask)
        else:
            outputs = self.deberta(input_ids, attention_mask)
            hidden_states = outputs[0]

        if self.span_pooling is not None:
            hidden_states = self._span_pooling_forward(hidden_states, span_ids, attention_mask)
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.int64, device=hidden_states.device)

        hidden_states = self.final_pooling(hidden_states, attention_mask)

        logits = self.fc(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            labels = labels.contiguous()
            loss_fct = FeedbackLoss(self.loss_weights)
            loss = loss_fct(logits.view(-1, NUM_CLASSES), labels.view(-1, NUM_CLASSES))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss, 
            logits=logits, 
        )
