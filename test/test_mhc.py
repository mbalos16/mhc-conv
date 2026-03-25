from ..mhc import mHCBlock
import torch


def test_run_sinkhorn_knopp():
    """
    Validate if the run_sinkhorn_knopp with and without max return the same output.
    """
    conv_1 = torch.nn.Conv2d(
        in_channels=3, out_channels=3, kernel_size=3, stride=1, padding="same"
    )
    mhc_block = mHCBlock(layer=conv_1, num_channels=3, num_streams=4)

    rand_matrix = (
        torch.rand(8, 20, 30, 4, 3) * 10
    ) - 5  # Shape needed for h_tilde: B, H, W, C, N
    old_result = mhc_block.run_sinkhorn_knopp(h_tilde=rand_matrix, stable_softmax=False)
    new_result = mhc_block.run_sinkhorn_knopp(h_tilde=rand_matrix, stable_softmax=True)

    assert torch.allclose(old_result, new_result, atol=0.001)
