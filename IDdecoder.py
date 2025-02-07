import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class IDFusionDecoder(nn.Module):
    def __init__(self, num_ids, id_embed_size, target_embed_size, nhead=8, num_decoder_layers=6):
        super(IDFusionDecoder, self).__init__()
        self.id_embed_size = id_embed_size
        self.target_embed_size = target_embed_size
        self.id_embedding = nn.Embedding(num_ids, id_embed_size)
        self.gate_layer = nn.Linear(id_embed_size + target_embed_size, target_embed_size)
        decoder_layer = TransformerDecoderLayer(d_model=target_embed_size, nhead=nhead)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_layer = nn.Linear(target_embed_size, num_ids)

    def forward(self, id_label, history_target_embed, current_target_embed):

        id_embed = self.id_embedding(id_label)  # (batch_size, id_embed_size)
        concat_embed = torch.cat((id_embed, history_target_embed),
                                 dim=-1)  # (batch_size, id_embed_size + target_embed_size)
        gate = torch.sigmoid(self.gate_layer(concat_embed))  # (batch_size, target_embed_size)
        fused_embed = gate * id_embed + (1 - gate) * history_target_embed  # (batch_size, target_embed_size)
        fused_embed = fused_embed.unsqueeze(0)  # (1, batch_size, target_embed_size)
        current_target_embed = current_target_embed.unsqueeze(0)  # (1, batch_size, target_embed_size)
        decoder_output = self.decoder(current_target_embed, fused_embed)  # (1, batch_size, target_embed_size)
        logits = self.output_layer(decoder_output.squeeze(0))  # (batch_size, num_ids)

        return logits