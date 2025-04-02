# --- Core Imports ---
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint # For gradient checkpointing

# --- Hugging Face Ecosystem ---
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
# from transformers import CLIPTextModel, CLIPTokenizer # Example for future text conditioning

# --- Helpers ---
from einops import rearrange # pip install einops
import warnings

# Try importing xformers for memory efficient attention
try:
    import xformers
    import xformers.ops
    XFORMERS_AVAILABLE = True
    print("xformers found. Memory-efficient attention enabled.")
except ImportError:
    XFORMERS_AVAILABLE = False
    print("WARNING: xformers not found. Falling back to default PyTorch attention. pip install xformers for potential speedups.")

# --- Configuration ---
CONFIG = {
    # --- Data ---
    "dataset_name": "HuggingFaceFV/finevideo",
    "dataset_split": "train", # Adjust as needed (e.g., 'train[:1%]' for testing)
    "video_col": "video",
    "cache_dir": "./processed_cache", # Directory to cache processed dataset
    "force_preprocess": False,      # Set True to ignore cache and re-preprocess
    # --- Video Params ---
    "max_frames": 16,
    "resolution": 128, # Start small (64, 128)
    # --- VAE ---
    "vae_model_name": "stabilityai/sd-vae-ft-mse",
    "vae_scale_factor": 0.18215,
    "latent_channels": 4, # Should match VAE output channels
    # --- DiT Architecture ---
    "patch_size_t": 2,
    "patch_size_hw": 4,
    "hidden_size": 256, # Keep small initially (e.g., 256, 512, 768)
    "depth": 8,         # Keep small initially (e.g., 8, 12, 24)
    "num_heads": 8,       # Must divide hidden_size
    "mlp_ratio": 4.0,
    "learn_sigma": False, # Keep False for simplicity first. If True, doubles output channels.
    # --- Conditioning ---
    "cond_dim": 768,     # Dimension of conditioning embeddings (e.g., CLIP text embedding size). Set to 0 if no conditioning.
    # --- Training ---
    "batch_size": 2,      # Adjust based on VRAM (Video is memory intensive!)
    "map_batch_multiplier": 2, # Batch size multiplier for dataset .map() operation
    "epochs": 5,          # Number of training epochs
    "learning_rate": 1e-4,
    "optimizer": "AdamW", # AdamW is standard
    "gradient_accumulation_steps": 4,
    "mixed_precision": "fp16", # 'fp16', 'bf16', or 'no'
    "use_gradient_checkpointing": True, # Enable gradient checkpointing in DiT blocks
    "use_flash_attention": True, # Use xformers memory_efficient_attention if available
    "noise_schedule": "squaredcos_cap_v2", # Or 'linear', 'scaled_linear'
    "num_train_timesteps": 1000,
    "dataloader_num_workers": 4, # Adjust based on CPU cores/IO
    "pin_memory": True,         # Usually good for GPU training
    # --- Logging/Saving ---
    "output_dir": "./video_gen_output",
    "log_interval": 10,
    "save_interval": 500, # Save checkpoint every N steps
    "seed": 42,
}

# --- Safety & Efficiency Utilities ---

# set_seed is now imported from accelerate.utils

def get_video_transforms(resolution):
    """Define transformations for video frames."""
    from torchvision.transforms import v2 as transforms # Use v2 for better video support
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resolution, antialias=True),
        transforms.CenterCrop(resolution),
        transforms.ToImage(), # Convert back to tensor (CHW format)
        transforms.ToDtype(torch.float32, scale=True), # Normalize to [0.0, 1.0]
        transforms.Normalize([0.5], [0.5]), # Normalize to [-1.0, 1.0]
    ])

# --- Data Loading and Preprocessing ---

def preprocess_video(batch, config, transforms_fn):
    """Preprocesses a batch of videos. Assumes 'video' column contains loadable video data."""
    processed_videos = []
    video_key = config["video_col"]

    if video_key not in batch:
         print(f"Warning: Video column '{video_key}' not found in batch keys: {batch.keys()}. Skipping batch.")
         return {"pixel_values": None} # Use None to easily filter later

    for video_data in batch[video_key]:
        if video_data is None:
             print("Warning: Encountered None video data. Skipping.")
             continue

        # --- Video Decoding ---
        # This part is CRITICAL and dataset dependent.
        # `datasets` often handles basic loading if `av` is installed.
        # Assuming `video_data` is usable by torchvision transforms (e.g., list of frames or path)
        # If it's raw bytes or specific format, manual decoding with pyav/decord needed here.
        try:
            # Example: Assuming video_data might be {'path': '...', 'bytes': ...} or just a path
            # Or it could already be decoded frames by the dataset loading script
            # Let's assume it yields frame tensors directly [T, H, W, C] for this example
            # We rely on the dataset loader + `av` to provide something usable by ToPILImage
            if isinstance(video_data, dict) and 'bytes' in video_data:
                # Requires specific decoding logic based on format
                # Placeholder: Skip complex cases for now
                print(f"Warning: Skipping video with complex data structure: {type(video_data)}")
                continue
            elif isinstance(video_data, list) and len(video_data) > 0:
                 # Assume list of PIL Images or Tensors
                 video_frames = video_data
            else:
                 # Attempt direct loading if it's a path or other simple type
                 # This is highly speculative without knowing the exact dataset format
                 video_frames = video_data # This line likely needs dataset-specific handling

            # --- Frame Sampling & Transformation ---
            num_frames = len(video_frames)
            if num_frames < config["max_frames"]:
                # Simple repeat padding if too short
                indices = torch.arange(num_frames)
                indices = indices.repeat(math.ceil(config["max_frames"] / num_frames))
                indices = indices[:config["max_frames"]]
            else:
                # Uniform sampling if too long
                indices = torch.linspace(0, num_frames - 1, config["max_frames"], dtype=torch.long)

            sampled_frames = [video_frames[i] for i in indices]

            # Transform frames
            transformed_frames = torch.stack([transforms_fn(frame) for frame in sampled_frames])
            # Output shape: [max_frames, C, resolution, resolution]
            processed_videos.append(transformed_frames)

        except Exception as e:
            print(f"Error processing video item: {e}. Skipping item.")
            # Log the type of video_data for debugging: print(f"Data type was: {type(video_data)}")
            continue

    if not processed_videos:
        return {"pixel_values": None}

    # Stack videos into a batch -> [Batch_size, T, C, H, W]
    # Need to handle cases where some videos failed processing within the batch
    return {"pixel_values": torch.stack(processed_videos)}


# --- Model Architecture ---

def modulate(x, shift, scale):
    # x: (B, N, D) or (B, D)
    # shift, scale: (B, D)
    # Unsqueeze depends on input shape
    if x.ndim == 3: # B, N, D
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    elif x.ndim == 2: # B, D
         return x * (1 + scale) + shift
    else:
         raise ValueError(f"Unsupported input shape for modulate: {x.shape}")


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTBlock(nn.Module):
    """A Diffusion Transformer Block with AdaLN-Zero conditioning and optional memory-efficient attention."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_flash_attention=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_flash_attention = use_flash_attention and XFORMERS_AVAILABLE

        # Define projection weights for Q, K, V and Output, matching nn.MultiheadAttention
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # We won't instantiate nn.MultiheadAttention if using xformers directly for QKV projection
        # self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True) # Replaced logic

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True) # 6 for scale/shift for norm1, attn proj, norm2, mlp
        )
        self._init_modulation_weights() # Best practice init

    def _init_modulation_weights(self):
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)

    def _attention(self, x):
        B, N, C = x.shape
        qkv = self.Wqkv(x) # (B, N, 3*C)

        if self.use_flash_attention:
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4) # (3, B, N, H, D)
            q, k, v = qkv[0], qkv[1], qkv[2] # (B, N, H, D) each
             # Need to reshape for xformers: (B, N, H, D) -> (B, H, N, D) ? Check docs.
             # xformers expects (batch_size, seq_len, num_heads, head_dim)
            attn_output = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
            # attn_output shape (B, N, H, D) -> Reshape back
            attn_output = attn_output.reshape(B, N, C)
        else:
            # Fallback to standard PyTorch attention mechanism split
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B, H, N, D
            q, k, v = qkv[0], qkv[1], qkv[2] # B, H, N, D
            # Scale dot product attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v) # Uses Flash Attn internally if available in PyTorch >= 2.0
            # attn_output shape (B, H, N, D) -> Reshape back
            attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, C)

        return self.out_proj(attn_output)


    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # Modulated Norm + Attention
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output = self._attention(x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_output

        # Modulated Norm + MLP
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x

class FinalLayer(nn.Module):
    """Final layer for DiT: LayerNorm, modulation, and linear projection."""
    def __init__(self, hidden_size, patch_size_t, patch_size_hw, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size_t * patch_size_hw * patch_size_hw * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self._init_modulation_weights() # Best practice init

    def _init_modulation_weights(self):
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.linear.weight, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class VideoDiffusionTransformer(nn.Module):
    """Video Diffusion Transformer with Spacetime Patching and AdaLN Conditioning."""
    def __init__(
        self,
        input_size=(CONFIG["max_frames"], CONFIG["latent_channels"], CONFIG["resolution"] // 8, CONFIG["resolution"] // 8),
        patch_size=(CONFIG["patch_size_t"], CONFIG["patch_size_hw"], CONFIG["patch_size_hw"]),
        hidden_size=CONFIG["hidden_size"],
        depth=CONFIG["depth"],
        num_heads=CONFIG["num_heads"],
        mlp_ratio=CONFIG["mlp_ratio"],
        learn_sigma=CONFIG["learn_sigma"],
        cond_dim=CONFIG["cond_dim"],
        use_gradient_checkpointing=CONFIG["use_gradient_checkpointing"],
        use_flash_attention=CONFIG["use_flash_attention"],
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = input_size[1]
        self.out_channels = self.in_channels * 2 if learn_sigma else self.in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.cond_dim = cond_dim

        self.input_depth, self.input_height, self.input_width = input_size[0], input_size[2], input_size[3]
        grid_dims = (self.input_depth // patch_size[0], self.input_height // patch_size[1], self.input_width // patch_size[2])
        self.num_patches = math.prod(grid_dims)

        # --- Embeddings ---
        self.patch_embed = nn.Conv3d(self.in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, hidden_size, bias=True) # Project condition embedding
            print(f"Conditioning enabled with dimension: {cond_dim} -> {hidden_size}")
        else:
            self.cond_proj = None
            print("Conditioning disabled (cond_dim=0).")

        # --- Transformer Blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_flash_attention=use_flash_attention)
            for _ in range(depth)
        ])

        # --- Final Layer ---
        self.final_layer = FinalLayer(hidden_size, patch_size[0], patch_size[1], self.out_channels)

        # --- Initialization ---
        self._initialize_weights()
        print(f"DiT Initialized: Depth={depth}, Hidden={hidden_size}, Heads={num_heads}, Patches={self.num_patches}")

    def _initialize_weights(self):
        # Initialize patch embedding like ViT (`trunc_normal_`)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.constant_(self.patch_embed.bias, 0)
        # Initialize positional embedding (`trunc_normal_`)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Initialize condition projection if exists
        if self.cond_proj is not None:
             nn.init.xavier_uniform_(self.cond_proj.weight)
             nn.init.constant_(self.cond_proj.bias, 0)
        # Other layers (DiTBlock, FinalLayer) handle their own init.

    def unpatchify(self, x):
        """Reconstruct latent video from patch tokens."""
        B = x.shape[0]
        T_p, H_p, W_p = self.patch_size
        T_g, H_g, W_g = self.input_depth // T_p, self.input_height // H_p, self.input_width // W_p
        C_out = self.out_channels

        try:
            x = rearrange(x, 'b (tg hg wg) (tp hp wp c) -> b c (tg tp) (hg hp) (wg wp)',
                          tg=T_g, hg=H_g, wg=W_g, tp=T_p, hp=H_p, wp=W_p, c=C_out)
            return x
        except ImportError:
             warnings.warn("einops not installed. Manual unpatchify is less robust. pip install einops")
             # Fallback (less tested)
             x = x.view(B, T_g, H_g, W_g, T_p, H_p, W_p, C_out)
             x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous() # B, C_out, T_g, T_p, H_g, H_p, W_g, W_p
             x = x.view(B, C_out, self.input_depth, self.input_height, self.input_width)
             return x

    def forward(self, x, t, condition_embed=None):
        """
        x: Latent video (B, C_in, T_in, H_in, W_in)
        t: Timesteps (B,)
        condition_embed: Optional condition embeddings (B, cond_dim)
        """
        B, C_in, T_in, H_in, W_in = x.shape
        if (T_in, H_in, W_in) != (self.input_depth, self.input_height, self.input_width):
            # TODO: Implement optional resizing/interpolation if needed, or error out.
            raise ValueError(f"Input latent size mismatch: {x.shape} vs expected {(self.input_depth, self.input_height, self.input_width)}")

        # 1. Embeddings
        x = self.patch_embed(x)           # (B, D, T_g, H_g, W_g)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D) N = T_g*H_g*W_g
        x = x + self.pos_embed            # Add positional embedding

        t_embed = self.t_embedder(t)      # (B, D) Timestep embedding

        # Combine timestep and condition embeddings
        if self.cond_proj is not None:
            if condition_embed is None:
                # TODO: Implement guidance dropout (set condition_embed to zeros/learned embedding)
                 warnings.warn("Conditioning enabled but no condition_embed provided. Using zeros.")
                 condition_embed = torch.zeros(B, self.cond_dim, device=x.device, dtype=x.dtype) # Placeholder
                # else: check shape: condition_embed.shape == (B, self.cond_dim)
            cond_embed_proj = self.cond_proj(condition_embed) # (B, D)
            c = t_embed + cond_embed_proj # Simple addition
        else:
            c = t_embed # Only timestep

        # 2. Apply Transformer Blocks
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                # use_reentrant=False is often more memory efficient with recent PyTorch versions
                x = checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)

        # 3. Final Layer and Unpatchify
        x = self.final_layer(x, c)      # (B, N, T_p*H_p*W_p*C_out)
        x = self.unpatchify(x)          # (B, C_out, T_in, H_in, W_in)

        return x


class VideoGeneratorModel(nn.Module):
    """Main model integrating VAE and VideoDiffusionTransformer."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vae_scale_factor = config["vae_scale_factor"]
        self.latent_channels = config["latent_channels"]

        # --- VAE ---
        self.vae = AutoencoderKL.from_pretrained(config["vae_model_name"])
        self.vae.requires_grad_(False) # Freeze VAE
        print(f"Loaded frozen VAE: {config['vae_model_name']}")

        # --- Calculate Latent Size ---
        # Assuming VAE downsampling factor is 8 (common for SD VAEs)
        downsample_factor = 8 # TODO: Could try to infer this from vae.config if available
        latent_h = config["resolution"] // downsample_factor
        latent_w = config["resolution"] // downsample_factor
        latent_t = config["max_frames"]
        self.latent_size = (latent_t, self.latent_channels, latent_h, latent_w)
        print(f"Calculated latent size (T, C, H, W): {self.latent_size}")

        # --- DiT ---
        self.dit = VideoDiffusionTransformer(
            input_size=self.latent_size,
            patch_size=(config["patch_size_t"], config["patch_size_hw"], config["patch_size_hw"]),
            hidden_size=config["hidden_size"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            learn_sigma=config["learn_sigma"],
            cond_dim=config["cond_dim"],
            use_gradient_checkpointing=config["use_gradient_checkpointing"],
            use_flash_attention=config["use_flash_attention"]
        )

    @torch.no_grad() # Ensure VAE encoding doesn't track gradients
    def encode_video(self, pixel_values):
        """Encodes video frames into latents using VAE."""
        b, t, c, h, w = pixel_values.shape
        pixel_values = rearrange(pixel_values, 'b t c h w -> (b t) c h w')
        self.vae.eval() # Set VAE to eval mode

        # Use float32 for VAE encoding stability if using mixed precision
        if pixel_values.dtype != torch.float32:
             pixel_values = pixel_values.to(torch.float32)

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae_scale_factor

        # Reshape back: (B*T, C_l, H_l, W_l) -> (B, C_l, T, H_l, W_l)
        latent_c, latent_h, latent_w = latents.shape[1], latents.shape[2], latents.shape[3]
        latents = rearrange(latents, '(b t) c h w -> b c t h w', b=b, t=t)

        # Dynamic check (optional, can be removed after verification)
        if (t, latent_c, latent_h, latent_w) != self.latent_size:
            warnings.warn(f"Actual VAE latent size {(t, latent_c, latent_h, latent_w)} differs from expected {self.latent_size}. Check VAE/config.")

        return latents # Shape: [B, C_latent, T_latent, H_latent, W_latent]

    def forward(self, noisy_latents, timesteps, condition_embed=None):
        """Forward pass through the DiT."""
        return self.dit(noisy_latents, timesteps, condition_embed)

# --- Main Training Function ---

def train(config):
    print("Initializing Training...")
    set_seed(config["seed"])

    # 1. Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=config["mixed_precision"],
        log_with="tensorboard",
        project_dir=config["output_dir"]
    )
    accelerator.print(f"Accelerator config: {accelerator.state}")

    # 2. Load and Preprocess Dataset
    accelerator.print("Loading dataset...")
    transforms_fn = get_video_transforms(config["resolution"])

    # --- Caching Logic ---
    dataset_id = config["dataset_name"].split('/')[-1]
    cache_file_name = f"{dataset_id}_{config['dataset_split']}_{config['resolution']}p_{config['max_frames']}f.arrow"
    cache_path = os.path.join(config["cache_dir"], cache_file_name)
    os.makedirs(config["cache_dir"], exist_ok=True)

    if os.path.exists(cache_path) and not config["force_preprocess"]:
        accelerator.print(f"Loading processed dataset from cache: {cache_path}")
        processed_dataset = load_from_disk(cache_path)
        processed_dataset.set_format("torch") # Ensure format is set after loading
    else:
        accelerator.print("Preprocessing dataset (cache not found or forced)...")
        # Load raw dataset
        try:
            dataset = load_dataset(config["dataset_name"], split=config["dataset_split"], trust_remote_code=True)
        except Exception as e:
            accelerator.print(f"Error loading raw dataset: {e}")
            accelerator.print("Check dataset name, network connection, and potentially 'av' dependency ('pip install av').")
            return

        if config["video_col"] not in dataset.features:
            accelerator.print(f"ERROR: Video column '{config['video_col']}' not found! Available: {list(dataset.features.keys())}")
            return

        # Define preprocessing function with config access
        preprocess_fn = lambda batch: preprocess_video(batch, config, transforms_fn)

        # Apply preprocessing map
        try:
             map_batch_size = config["batch_size"] * config["map_batch_multiplier"]
             columns_to_remove = [col for col in dataset.column_names] # Remove all original columns
             processed_dataset = dataset.map(
                 preprocess_fn,
                 batched=True,
                 batch_size=max(1, map_batch_size // accelerator.num_processes), # Adjust batch size for map
                 num_proc=config["dataloader_num_workers"], # Use multiple procs for mapping too
                 remove_columns=columns_to_remove,
                 # keep_in_memory=False # If dataset is huge
             )
             # Filter out entries where preprocessing failed (returned None)
             processed_dataset = processed_dataset.filter(lambda x: x['pixel_values'] is not None)
             if len(processed_dataset) == 0:
                  accelerator.print("ERROR: Dataset empty after filtering failed preprocessing steps.")
                  return
             processed_dataset.set_format("torch")
             accelerator.print("Dataset preprocessing complete.")
             # Save to cache
             if accelerator.is_main_process:
                 accelerator.print(f"Saving processed dataset to cache: {cache_path}")
                 processed_dataset.save_to_disk(cache_path)

        except Exception as e:
            accelerator.print(f"Error during dataset preprocessing: {e}")
            if "Ran out of memory" in str(e):
                accelerator.print("Try reducing 'map_batch_multiplier', 'resolution', or 'max_frames'.")
            return

    accelerator.print(f"Final dataset size: {len(processed_dataset)}")

    # 3. Create DataLoader
    dataloader = DataLoader(
        processed_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["dataloader_num_workers"],
        pin_memory=config["pin_memory"],
        # persistent_workers=True if config["dataloader_num_workers"] > 0 else False # Can speed up iteration
    )
    accelerator.print("DataLoader created.")

    # 4. Initialize Model, Optimizer, Scheduler
    model = VideoGeneratorModel(config)

    # Optimize only the DiT parameters
    optimizer = torch.optim.AdamW(model.dit.parameters(), lr=config["learning_rate"])
    accelerator.print(f"Optimizer: {config['optimizer']} targeting DiT parameters.")

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule=config["noise_schedule"],
        prediction_type="epsilon" # Default, predicting noise
    )
    accelerator.print(f"Noise Scheduler: {noise_scheduler.__class__.__name__} ({config['noise_schedule']})")

    # TODO: Add LR Scheduler (e.g., Cosine, Linear)
    # lr_scheduler = ...

    # 5. Prepare with Accelerator
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader # Add lr_scheduler here if used
    )
    accelerator.print("Components prepared with Accelerator.")

    # --- Optional: Text Conditioning Setup (Example) ---
    # if config["cond_dim"] > 0:
    #     tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    #     text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device)
    #     text_encoder.requires_grad_(False) # Freeze text encoder
    #     # TODO: Need to get text prompts paired with videos in the dataset
    #     # and tokenize/encode them in the batch loading/preprocessing step.

    # 6. Training Loop
    accelerator.print("Starting training loop...")
    global_step = 0
    total_steps = len(dataloader) * config["epochs"] // config["gradient_accumulation_steps"]
    accelerator.print(f"Total training steps: {total_steps}")

    for epoch in range(config["epochs"]):
        model.train() # Set DiT to train mode (VAE remains eval)
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                pixel_values = batch.get('pixel_values')
                if pixel_values is None or pixel_values.numel() == 0: continue # Should not happen after filtering

                B = pixel_values.shape[0]

                # --- Get Latents ---
                # No need for unwrap_model if prepare handles device placement
                # VAE encoding is done within the model's encode_video, which uses no_grad
                latents = model.module.encode_video(pixel_values) if accelerator.num_processes > 1 else model.encode_video(pixel_values)
                latents = latents.to(dtype=model.module.dit.dtype if accelerator.num_processes > 1 else model.dit.dtype) # Match DiT dtype for mixed precision

                # --- Prepare for DiT ---
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # --- TODO: Prepare Conditioning Embeddings ---
                condition_embed = None
                # if config["cond_dim"] > 0:
                #    # Get text prompts from batch['text'] (needs to be added in dataset handling)
                #    # tokenized_prompts = tokenizer(...)
                #    # condition_embed = text_encoder(tokenized_prompts.input_ids)[0] # Get CLIP last hidden state
                #    # Handle classifier-free guidance (dropout): randomly set some cond_embed to null/zero embedding
                #    pass # Placeholder

                # --- Forward Pass ---
                model_pred = model(noisy_latents, timesteps, condition_embed)

                # --- Loss Calculation ---
                if config["learn_sigma"]:
                    # Predicts both epsilon and variance
                    pred_epsilon, pred_variance = torch.chunk(model_pred, 2, dim=1)
                    # For simplicity, using standard epsilon loss. VLB loss for variance is more complex.
                    # TODO: Implement VLB loss if variance prediction is critical.
                    loss = torch.nn.functional.mse_loss(pred_epsilon, noise)
                else:
                    # Predicts only epsilon
                    loss = torch.nn.functional.mse_loss(model_pred, noise)

                # --- Backpropagation ---
                accelerator.backward(loss)

                if accelerator.sync_gradients: # Only clip when gradients are synced
                    # Clip gradients for the parameters being optimized
                    accelerator.clip_grad_norm_(model.module.dit.parameters() if accelerator.num_processes > 1 else model.dit.parameters(), 1.0)

                optimizer.step()
                # lr_scheduler.step() # If using LR scheduler
                optimizer.zero_grad()

            # --- Logging & Saving ---
            if accelerator.sync_gradients: # Log/Save only after optimizer step
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % config["log_interval"] == 0:
                        logs = {"loss": loss.item(), "step": global_step, "epoch": epoch}
                        accelerator.print(f"Epoch {epoch}/{config['epochs']}, Step {global_step}/{total_steps}, Loss: {loss.item():.4f}")
                        accelerator.log(logs, step=global_step)

                    if global_step % config["save_interval"] == 0:
                        save_path = os.path.join(config["output_dir"], f"checkpoint-{global_step}")
                        accelerator.save_state(save_path) # Saves model, opt, scheduler state etc.
                        accelerator.print(f"Checkpoint saved to {save_path}")

            # Basic check to stop if loss is NaN/inf
            if torch.isnan(loss) or torch.isinf(loss):
                 accelerator.print(f"ERROR: Loss is NaN/Inf at step {global_step}. Stopping training.")
                 return # Stop training

    # --- End of Training ---
    accelerator.print("Training finished.")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_save_path = os.path.join(config["output_dir"], "final_model")
        accelerator.save_state(final_save_path)

        # Optionally save just the trained DiT weights
        unwrapped_model = accelerator.unwrap_model(model)
        dit_state_dict = unwrapped_model.dit.state_dict()
        torch.save(dit_state_dict, os.path.join(config["output_dir"], "final_dit_weights.pt"))
        accelerator.print(f"Final state saved to {final_save_path}")
        accelerator.print(f"Final DiT weights saved to {os.path.join(config['output_dir'], 'final_dit_weights.pt')}")


if __name__ == "__main__":
    # Create output & cache directories if they don't exist
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    train(CONFIG)