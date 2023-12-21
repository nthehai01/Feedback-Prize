import torch
import torch.nn as nn

from typing import Optional, Tuple, Union

from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.model.pooling import MeanPooling
from src.metrics.loss import FeedbackLoss
from src.model.configuration import FeedbackConfig


NUM_CLASSES = 6


class BaseFeedbackModel(PreTrainedModel):
    config_class = FeedbackConfig
    supports_gradient_checkpointing = True

    def __init__(self, 
                 config):
        super().__init__(config)
        self.config = config

        self.post_init()

    def _set_backbone(self, modules):
        setattr(self, self.backbone_name, modules)

    def _get_backbone(self):
        return getattr(self, self.backbone_name)

    def get_input_embeddings(self):
        # This function will be called by `model.enable_input_require_grads()`
        # Return the very first embedding layer since this layer directly takes the inputs
        for _, module in self._get_backbone().named_modules():
            if isinstance(module, nn.Embedding):
                return module

        raise ValueError("Cannot find the input embedding layer.")
    
    def set_input_embeddings(self, value):
        # This function will be called by `model.resize_token_embeddings(value)`

        def _get_current_layer_by_name(self, attribute_names):
            """
            For example:
                attribute_names = ["bert", "encoder", "layer"]
                return self.bert.encoder.layer
            """

            current_obj = self
            for name in attribute_names[:-1]:
                current_obj = getattr(current_obj, name)

            return current_obj
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                attribute_names = name.split('.')
                layer = _get_current_layer_by_name(attribute_names)
                setattr(layer, attribute_names[-1], value)
                break
                
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class FeedbackModel(BaseFeedbackModel):
    def __init__(self, 
                 config, 
                 sliding_window_config,
                 span_pooling_config, 
                 final_pooling_config,
                 loss_weights):
        super().__init__(config)
        self.config = config
        
        self.backbone_name = config.backbone_name

        self.sliding_window_config = sliding_window_config
        self.span_pooling_config = span_pooling_config
        self.loss_weights = torch.tensor(loss_weights)
        
        backbone_config = AutoConfig.from_pretrained(config._name_or_path)
        self._set_backbone(AutoModel.from_config(backbone_config))

        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.span_pooling = MeanPooling(**span_pooling_config) if span_pooling_config.use_span_pooling else None

        self.final_pooling = MeanPooling(**final_pooling_config)

        self.fc = nn.Linear(config.hidden_size, NUM_CLASSES)

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

        first_segment_outputs = self._get_backbone()(
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
            segment_outputs = self._get_backbone()(
                input_ids[:, start_window:end_window],
                attention_mask[:, start_window:end_window],
            )

            if end_window == seq_len:
                hidden_states.append(
                    segment_outputs[0][:, inner_size:]
                )
                break
            else:
                hidden_states.append(
                    segment_outputs[0][:, inner_size:inner_size+n_tokens]
                )

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
            for span_id in ids:
                if span_id == self.span_pooling_config.ignored_span_id:
                    continue

                mask = span_ids == span_id
                temp = self.span_pooling(
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
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        seq_len = input_ids.shape[1]
        if self.sliding_window_config.use_sliding_window and seq_len > self.sliding_window_config.window_size:
            hidden_states = self._sliding_window_encode(input_ids, attention_mask)
        else:
            outputs = self._get_backbone()(input_ids, attention_mask)
            hidden_states = outputs[0]

        hidden_states = self.hidden_dropout(hidden_states)

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
