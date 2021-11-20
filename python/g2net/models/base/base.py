from g2net.models.base.architectures import SpectroCNN
from g2net.models.base.wavegram import CNNSpectrogram


def create_base_model() -> SpectroCNN:
    return SpectroCNN(model_name='tf_efficientnet_b6_ns',
                      pretrained=True,
                      num_classes=1,
                      spectrogram=CNNSpectrogram,
                      spec_params=dict(
                          base_filters=128,
                          kernel_sizes=(64, 16, 4),
                      ),
                      resize_img=None,
                      custom_classifier='gem',
                      upsample='bicubic')
