import torch

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self.max_seq_len_cached = 0

    def build_rope_cache(self, seq_len, device):
        if self.max_seq_len_cached >= seq_len:
            return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires the embedding dimension to be even.")

        half_dim = self.head_dim // 2
        freq_seq = torch.arange(half_dim, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.base ** (freq_seq / half_dim))
        pos = torch.arange(seq_len, dtype=torch.float32, device=device)
        sinusoid = torch.einsum('i,j->ij', pos, inv_freq)
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)

        sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, self.head_dim)
        cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, self.head_dim)

        self.cos_cached = cos
        self.sin_cached = sin
        self.max_seq_len_cached = seq_len

        return cos.to(device), sin.to(device)

    def apply_rope(self, x, cos, sin):
        """
        应用 RoPE 到输入张量 x。
        x: [B, H, L, D]
        cos, sin: [L, D]
        """
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires the embedding dimension to be even.")
        
        x1 = x[..., ::2]  # even
        x2 = x[..., 1::2]  # odd
        x_rotated = torch.stack([-x2, x1], dim=-1).reshape_as(x)

        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, L, D]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, L, D]

        return x * cos + x_rotated * sin

    def forward(self, x):
        """
        输入 x: [B, H, L, D]
        返回值: 应用 RoPE 后的 x
        """
        B, H, L, D = x.shape
        if D != self.head_dim:
            raise ValueError(f"Expected head_dim={self.head_dim}, but got {D}")
        device = x.device
        cos, sin = self.build_rope_cache(L, device)
        return self.apply_rope(x, cos, sin)