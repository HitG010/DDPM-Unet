import torch
import torch.nn as nn

def get_time_embedding(time_steps, t_emb_dim):
    """
    Generates time embeddings for the given time steps.
    """
    
    assert t_emb_dim % 2 == 0, "t_emb_dim must be even"
    
    factor = 10000 ** (torch.arange(
        start = 0, end = t_emb_dim / 2, device = time_steps.device) / (t_emb_dim // 2))
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, down_sample = True, num_head = 4, num_layers = 1):
        super().__init__()
        self.down_sample = down_sample
        self.num_layers = num_layers
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = 8, num_channels = in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
            ) for i in range(num_layers)
        ])
                
        
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels),
            )
            for _ in range(num_layers)
        ])
        
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = 8, num_channels = out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
            ) for _ in range(num_layers)
        ])
        
        self.attention_norm = nn.ModuleList([
            nn.GroupNorm(num_groups = 8, num_channels = out_channels) for _ in range(num_layers)
        ])
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim = out_channels, num_heads = num_head, batch_first = True) for _ in range(num_layers)
        ])
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size = 1) for i in range(num_layers)
        ])
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size = 4, stride = 2, padding = 1) if self.down_sample else nn.Identity()
    
    def forward(self, x, t_emb,):
        """
        Forward pass for the DownBlock.
        """
        out = x
        
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            # Attention Block
            batch_size, channels, height, width = out.shape
            in_attn = out.reshape(batch_size, channels, height * width)
            in_attn = self.attention_norm[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attention[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
            out = out + out_attn
        
        out = self.down_sample_conv(out)
        return out
        
        
class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_head = 4, num_layers = 1):
        self.num_layers = num_layers
        super().__init__()
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = 8, num_channels = in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
            ) for i in range(num_layers + 1)
        ])
        self.t_emb_layer = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels),
            )
            for _ in range(num_layers + 1)
        ])
        
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = 8, num_channels = out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
            ) for _ in range(num_layers + 1)
        ])
        
        
        self.attention_norm = nn.ModuleList([
            nn.GroupNorm(num_groups = 8, num_channels = out_channels) for _ in range(num_layers)
        ])
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim = out_channels, num_heads = num_head, batch_first = True) for _ in range(num_layers)
        ])
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers+1)
            ]
        )
        
    def forward(self, x, t_emb):
        """
        Forward pass for the MidBlock.
        """
        out = x
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layer[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](x)
        
        for i in range(self.num_layers):
            # Attention Block
            batch_size, channels, height, width = out.shape
            in_attn = out.reshape(batch_size, channels, height * width)
            in_attn = self.attention_norm[i](in_attn)
            in_attn = in_attn.transpose(1, 2)  # (batch_size, height * width, channels)
            out_attn, _ = self.attention[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
            out = out + out_attn
            
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            out = out + self.t_emb_layer[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)    
        
        return out
        
        
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up_sample = True, num_head = 4, num_layers = 1):
        
        super().__init__()
        self.up_sample = up_sample
        self.num_layers = num_layers
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = 8, num_channels = in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
            ) for i in range(num_layers)
        ])
        
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels),
            ) for _ in range(num_layers)
        ])
        
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = 8, num_channels = out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
            ) for _ in range(num_layers)
        ])
        
        self.attention_norm = nn.ModuleList([
            nn.GroupNorm(num_groups = 8, num_channels = out_channels) for _ in range(num_layers)
        ])
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim = out_channels, num_heads = num_head, batch_first = True) for _ in range(num_layers)
        ])
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size = 1) for i in range(num_layers)
        ])
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size = 4, stride = 2, padding = 1) if self.up_sample else nn.Identity()
        
    def forward(self, x, out_down, t_emb):
        
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim = 1)  # Concatenate along channel dimension
       
        out = x
        
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            # Attention Block
            batch_size, channels, height, width = out.shape
            in_attn = out.reshape(batch_size, channels, height * width)
            in_attn = self.attention_norm[i](in_attn)
            in_attn = in_attn.transpose(1, 2)  # (batch_size, height * width, channels)
            out_attn, _ = self.attention[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
            out = out + out_attn
        
        return out
        
        
class Unet(nn.Module):
    def __init__(self, model_config):
        """ Initializes the UNet model with the given configuration. """
        super().__init__()
        in_channels = model_config['in_channels']
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['t_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        
        assert self.mid_channels[0] == self.down_channels[-1], "First mid channel must match last down channel"
        assert self.mid_channels[-1] == self.down_channels[-2], "Last mid channel must match second last down channel"
        assert len(self.down_sample) == len(self.down_channels) - 1, "Down sample list must match down channels"

        
        
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(in_channels, self.down_channels[0], kernel_size=3, padding= (1, 1))
        
        self.downs = nn.ModuleList([])
        
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(
                in_channels=self.down_channels[i],
                out_channels=self.down_channels[i + 1],
                time_emb_dim=self.t_emb_dim,
                down_sample=self.down_sample[i],
                num_layers = self.num_down_layers
            ))
        
        self.mids = nn.ModuleList([])
        
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(
                self.mid_channels[i],
                self.mid_channels[i + 1],
                time_emb_dim=self.t_emb_dim,
                num_layers=self.num_mid_layers,
            ))
            
        self.ups = nn.ModuleList([])
        
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(UpBlock(
                in_channels=self.down_channels[i] * 2,
                out_channels=self.down_channels[i - 1] if i != 0 else 16,
                time_emb_dim=self.t_emb_dim,
                up_sample=self.up_sample[i],
                num_layers=self.num_up_layers
            ))
            
        self.norm_out = nn.GroupNorm(num_groups=8, num_channels=16)
        self.conv_out = nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1)
        
        
    def forward(self, x, t):
        """
        Forward pass for the UNet model.
        """
        out = self.conv_in(x)
        
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        down_outs = []
        
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)
            
        for mid in self.mids:
            out = mid(out, t_emb)
            
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        
        return out