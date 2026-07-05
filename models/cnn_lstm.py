import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A single Conv1D -> BatchNorm -> ReLU -> (optional) MaxPool block.
    Operates on tensors shaped (batch, channels, time).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 5, pool: bool = False):
        super().__init__()
        padding = kernel_size // 2  # 'same' padding to preserve time length before pooling

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class CNNLSTM(nn.Module):
    """
    CNN-LSTM model for HAR classification.

    Architecture:
        ConvBlock(9  -> 64,  no pool)
        ConvBlock(64 -> 64,  pool)      -> T: 128 -> 64
        ConvBlock(64 -> 128, no pool)
        ConvBlock(128-> 128, pool)      -> T: 64  -> 32
        BiLSTM(128 -> 128, num_layers=2)
        Mean-pool over time
        Linear(256 -> 64) -> ReLU -> Dropout -> Linear(64 -> num_classes)
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 6,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # --- Convolutional feature extractor ---
        self.conv_stack = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=5, pool=False),
            ConvBlock(64, 64, kernel_size=5, pool=True),    # T: 128 -> 64
            ConvBlock(64, 128, kernel_size=5, pool=False),
            ConvBlock(128, 128, kernel_size=5, pool=True),  # T: 64 -> 32
        )

        # --- Temporal modeling ---
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        lstm_output_dim = lstm_hidden_size * 2  # bidirectional -> concat both directions

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, time=128, channels=9)
        """
        # Conv1d expects (batch, channels, time)
        x = x.permute(0, 2, 1)          # (batch, 9, 128)
        x = self.conv_stack(x)          # (batch, 128, 32)

        # LSTM expects (batch, time, features)
        x = x.permute(0, 2, 1)          # (batch, 32, 128)

        lstm_out, (h_n, c_n) = self.lstm(x)   # lstm_out: (batch, 32, 256)

        # Mean-pool over the time dimension instead of using only h_n
        pooled = lstm_out.mean(dim=1)   # (batch, 256)

        logits = self.classifier(pooled)  # (batch, num_classes)
        return logits


if __name__ == "__main__":
    # Sanity check: forward pass with a dummy batch
    model = CNNLSTM(in_channels=9, num_classes=6)

    dummy_batch = torch.randn(32, 128, 9)  # (batch=32, time=128, channels=9)
    output = model(dummy_batch)

    print(f"Output shape: {output.shape}")  # expected: torch.Size([32, 6])
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")