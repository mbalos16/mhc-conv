import torch


class mHCBlock(torch.nn.Module):
    def __init__(self, layer, num_channels, num_streams):
        super().__init__()
        self.num_channels = num_channels
        self.num_streams = num_streams

        self.alpha_pre = torch.nn.parameter.Parameter(
            data=torch.randn(1) * 0.01, requires_grad=True
        )
        self.alpha_res = torch.nn.parameter.Parameter(
            data=torch.randn(1) * 0.01, requires_grad=True
        )
        self.alpha_post = torch.nn.parameter.Parameter(
            data=torch.randn(1) * 0.01, requires_grad=True
        )

        self.phi_pre_mapping = torch.nn.Linear(
            in_features=num_channels * num_streams, out_features=num_streams, bias=False
        )  # B, ..., C*N -> B, ..., N
        self.phi_post_mapping = torch.nn.Linear(
            in_features=num_channels * num_streams, out_features=num_streams, bias=False
        )  # B, ..., C*N -> B, ..., N
        self.phi_res_mapping = torch.nn.Linear(
            in_features=num_channels * num_streams,
            out_features=num_streams * num_streams,
            bias=False,
        )  # B, ..., C*N  -> B, ..., N^2

        self.b_pre = torch.nn.parameter.Parameter(
            data=torch.randn(1, num_streams) * 0.01, requires_grad=True
        )
        self.b_res = torch.nn.parameter.Parameter(
            data=torch.randn(num_streams * num_streams) * 0.01, requires_grad=True
        )
        self.b_post = torch.nn.parameter.Parameter(
            data=torch.randn(1, num_streams) * 0.01, requires_grad=True
        )

        self.layer = layer  # Convolutional layer
        self.rmsnorm = torch.nn.modules.normalization.RMSNorm(
            self.num_channels * self.num_streams
        )

    def forward(self, x):
        # Expected x shape (B, H, W, N, C)
        h_pre, h_res, h_post = self.calculate_mhc_mapping(x)
        h_post_t = h_post.unsqueeze(-1)

        x_pre = h_pre @ x  # B, H, W, 1, C
        x_pre = x_pre.squeeze(-2)  # B, H, W, C
        x_pre = x_pre.movedim(-1, 1)  # B, C, H, W

        layer_out = self.layer(x_pre)  # B, C, H, W
        layer_out = layer_out.movedim(1, -1)  # B, H, W, C
        layer_out = layer_out.unsqueeze(-2)  # B, H, W, 1, C

        output = h_res @ x + h_post_t @ layer_out
        return output

    def calculate_mhc_mapping(self, x):
        """
        A function that uses the instantiated learnable elements and calculates the HyperConnections
        (HC) mappings with which, latter one, calculates the manifold Hyper Connections(mHC).
        """
        # x shape (B, ..., C, N)
        # Flatten (B, ..., C, N) --> (B, ..., C*N)
        x_arrow = torch.flatten(x, start_dim=-2, end_dim=-1)

        # RMSNorm
        x_norm = self.rmsnorm(x_arrow)

        # HC: h_tilde_pre, h_tilde_post, h_tilde_res
        h_tilde_pre = (
            self.alpha_pre * self.phi_pre_mapping(x_norm) + self.b_pre
        )  # # B, ..., C*N, -> B, ..., N,
        h_tilde_post = (
            self.alpha_post * self.phi_post_mapping(x_norm) + self.b_post
        )  # B, ..., C*N, -> B, ..., N,
        res_shape = (
            x.shape[0],
            x.shape[1],
            x.shape[2],
            self.num_streams,
            self.num_streams,
        )
        h_tilde_res = self.alpha_res * torch.reshape(
            self.phi_res_mapping(x_norm), res_shape
        ) + self.b_res.reshape(
            self.num_streams, self.num_streams
        )  # B, ..., C*N  -> B, ..., C, N

        # Define the mHC: H_pre, H_res and H_post
        h_pre = torch.sigmoid(h_tilde_pre)
        h_res = self.run_sinkhorn_knopp(h_tilde_res)
        h_post = 2 * torch.sigmoid(h_tilde_post)

        h_pre = h_pre.unsqueeze(-2)

        return h_pre, h_res, h_post

    @staticmethod
    def run_sinkhorn_knopp(h_tilde, iterations=20, epsilon=1e-6, stable_softmax=True):
        """
        The Sinkhorn–Knopp algorithm iteratively rescales the rows and columns of a
        non-negative matrix so that they sum to one, converging to a doubly stochastic matrix.
        """
        # Stabilise exp by subtracting the global max. # Makes all elements in the matrix positive.
        if stable_softmax:
            h_tilde = h_tilde - h_tilde.amax(dim=(-2, -1), keepdim=True)
        h_res = torch.exp(h_tilde)

        for _ in range(iterations):
            # Column normalisation (T_c)
            h_res = h_res / (h_res.sum(dim=-2, keepdim=True) + epsilon)
            # Row normalisation (T_r)
            h_res = h_res / (h_res.sum(dim=-1, keepdim=True) + epsilon)

        return h_res


class mHCResNet(torch.nn.Module):
    def __init__(self, num_streams=4, num_blocks=10, num_outputs=1000):
        super(mHCResNet, self).__init__()
        self.num_streams = num_streams
        self.conv_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.conv_2 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        )
        self.conv_3 = torch.nn.Conv2d(
            in_channels=128 * 4, out_channels=256, kernel_size=3, stride=2, padding=1
        )

        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.mhc_blocks = torch.nn.ModuleList()

        for i in range(num_blocks):
            mhc_conv = torch.nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding="same"
            )
            mhc_layer = mHCBlock(layer=mhc_conv, num_channels=128, num_streams=4)
            self.mhc_blocks.append(mhc_layer)

        self.fully_connected = torch.nn.Linear(
            in_features=256, out_features=num_outputs
        )
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        # Input size: (batch_size, 3, 224, 224)
        x = self.conv_1(x)  # Shape: (batch_size, 64, 112, 112)
        x = self.activation(x)
        x = self.conv_2(x)  # Shape: (batch_size, 128, 56, 56)
        x = self.activation(x)

        # MHC
        x = self.move_channels_last(
            x
        )  # Shape: in:(batch_size, 128, 56, 56)  out: (batch_size, 56, 56, 128)
        x = x.unsqueeze(-2).repeat(
            1, 1, 1, 4, 1
        )  # Add N dimension | Shape: (batch_size, 56, 56, 4, 128)

        for layer in (
            self.mhc_blocks
        ):  # Shape: in: (batch_size, 56, 56, 4, 128) out: (batch_size, 56, 56, 4, 128)
            x = layer(x)
            x = self.activation(x)

        x = x.reshape(
            x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4]
        )  # Shape : (batch_size, 56, 56, 4*128)
        x = self.move_channels_first(x)  # Shape: (batch_size, 4*128, 56, 56)

        x = self.conv_3(x)  # Shape: (batch_size, 256, 28, 28)
        x = self.activation(x)

        x = self.adaptive_avg_pool(x)  # Shape: (batch_size, 256, 1, 1)
        x = torch.squeeze(x)  # Shape: (batch_size, 256)
        x = self.fully_connected(x)  # Shape: (batch_size, 256)
        return x

    @staticmethod
    def move_channels_first(x):
        return x.movedim(3, 1)

    @staticmethod
    def move_channels_last(x):
        return x.movedim(1, 3)
