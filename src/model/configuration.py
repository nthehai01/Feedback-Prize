from transformers.configuration_utils import PretrainedConfig


class FeedbackConfig(PretrainedConfig):
    def __init__(self,
                 backbone_name=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
