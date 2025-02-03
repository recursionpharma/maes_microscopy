import pytest
import torch

from maes_microscopy.huggingface_mae import MAEModel

huggingface_openphenom_model_dir = "."
# huggingface_modelpath = "recursionpharma/OpenPhenom"


@pytest.fixture
def huggingface_model():
    # Make sure you have the model/config downloaded from https://huggingface.co/recursionpharma/OpenPhenom to this directory
    # huggingface-cli download recursionpharma/OpenPhenom --local-dir=.
    huggingface_model = MAEModel.from_pretrained(huggingface_openphenom_model_dir)
    huggingface_model.eval()
    return huggingface_model


@pytest.mark.parametrize("C", [1, 4, 6, 11])
@pytest.mark.parametrize("return_channelwise_embeddings", [True, False])
def test_model_predict(huggingface_model, C, return_channelwise_embeddings):
    example_input_array = torch.randint(
        low=0,
        high=255,
        size=(2, C, 256, 256),
        dtype=torch.uint8,
        device=huggingface_model.device,
    )
    huggingface_model.return_channelwise_embeddings = return_channelwise_embeddings
    embeddings = huggingface_model.predict(example_input_array)
    expected_output_dim = 384 * C if return_channelwise_embeddings else 384
    assert embeddings.shape == (2, expected_output_dim)
