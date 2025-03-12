import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Modal Encoders
# ---------------------------

class TextEncoder(nn.Module):
    """Simple text encoder using embeddings and a feedforward layer."""
    def __init__(self, vocab_size, hidden_dim):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        # x: [batch_size, sequence_length]
        embedded = self.embedding(x)          # [batch_size, seq_len, hidden_dim]
        pooled = embedded.mean(dim=1)         # Mean pooling over sequence length
        return F.relu(self.fc(pooled))

class ImageEncoder(nn.Module):
    """Basic CNN-based image encoder."""
    def __init__(self, in_channels, hidden_dim):
        super(ImageEncoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        x = F.relu(self.conv(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)  # Flatten to [batch_size, hidden_dim]

class AudioEncoder(nn.Module):
    """Simple feedforward encoder for audio signals."""
    def __init__(self, input_dim, hidden_dim):
        super(AudioEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        # x: [batch_size, input_dim]
        return F.relu(self.fc(x))

class VideoEncoder(nn.Module):
    """Basic encoder for video represented as a feature vector."""
    def __init__(self, input_dim, hidden_dim):
        super(VideoEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        # x: [batch_size, input_dim]
        return F.relu(self.fc(x))

# ---------------------------
# Mixture-of-Experts (MoE) Layer
# ---------------------------

class MoELayer(nn.Module):
    """
    An MoE layer that routes input through a subset (top-k) of experts.
    Each expert is a simple linear layer.
    """
    def __init__(self, input_dim, output_dim, num_experts=4, k=2):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.k = k  # Number of experts to select per input sample
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        # x: [batch_size, input_dim]
        gate_scores = self.gate(x)                # [batch_size, num_experts]
        gate_probs = F.softmax(gate_scores, dim=1)  # [batch_size, num_experts]

        # Select top-k experts for each sample
        topk_vals, topk_idx = gate_probs.topk(self.k, dim=1)
        
        output = torch.zeros(x.size(0), self.experts[0].out_features, device=x.device)
        
        # Route each sample through its selected experts and weight the outputs
        for i in range(self.num_experts):
            mask = (topk_idx == i)
            if mask.any():
                expert_input = x[mask]
                expert_output = self.experts[i](expert_input)
                weights = gate_probs[mask, i].unsqueeze(1)
                output[mask] += expert_output * weights
        return output

# ---------------------------
# StelAI 1 Model
# ---------------------------

class StelAI1(nn.Module):
    """
    A multi-modal model that processes text, image, audio, and video.
    It combines modality-specific encodings using an MoE layer, then
    uses a final reasoning layer to produce the output.
    """
    def __init__(self, config):
        super(StelAI1, self).__init__()
        hidden_dim = config['hidden_dim']
        vocab_size = config['vocab_size']
        
        # Modal encoders
        self.text_encoder = TextEncoder(vocab_size, hidden_dim)
        self.image_encoder = ImageEncoder(config['image_channels'], hidden_dim)
        self.audio_encoder = AudioEncoder(config['audio_input_dim'], hidden_dim)
        self.video_encoder = VideoEncoder(config['video_input_dim'], hidden_dim)
        
        # Combined feature dimension: one feature vector per modality
        combined_input_dim = hidden_dim * 4  
        self.moe = MoELayer(combined_input_dim, hidden_dim,
                            num_experts=config['num_experts'],
                            k=config['top_k'])
        self.reasoning = nn.Linear(hidden_dim, config['output_dim'])
    
    def forward(self, inputs):
        """
        Expects a dictionary with keys: 'text', 'image', 'audio', 'video'.
        If a modality is missing, zeros are used as a placeholder.
        """
        batch_size = next(iter(inputs.values())).shape[0]
        device = next(self.parameters()).device
        
        text_out = self.text_encoder(inputs['text']) if 'text' in inputs else torch.zeros(batch_size, self.text_encoder.fc.out_features, device=device)
        image_out = self.image_encoder(inputs['image']) if 'image' in inputs else torch.zeros(batch_size, self.image_encoder.conv.out_channels, device=device)
        audio_out = self.audio_encoder(inputs['audio']) if 'audio' in inputs else torch.zeros(batch_size, self.audio_encoder.fc.out_features, device=device)
        video_out = self.video_encoder(inputs['video']) if 'video' in inputs else torch.zeros(batch_size, self.video_encoder.fc.out_features, device=device)
        
        # Concatenate modality features
        combined = torch.cat([text_out, image_out, audio_out, video_out], dim=1)
        moe_out = self.moe(combined)
        output = self.reasoning(moe_out)
        return output

    # ---------------------------
    # Additional Capabilities
    # ---------------------------
    
    def internet_search(self, query):
        """
        Placeholder for internet search functionality.
        """
        print(f"Searching the internet for: {query}")
        return {"results": f"Results for query '{query}'."}
    
    def rlhf_update(self, feedback, optimizer):
        """
        Placeholder for an RLHF update.
        """
        loss = torch.tensor(0.0, requires_grad=True)
        for key, reward in feedback.items():
            loss = loss - reward  # Replace with an actual loss based on feedback
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss