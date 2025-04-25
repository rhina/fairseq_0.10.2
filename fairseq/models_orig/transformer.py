from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from fairseq.models.fairseq_encoder_decoder import FairseqEncoderDecoderModel
from fairseq.models.fairseq_encoder import FairseqEncoder
from fairseq.models.fairseq_decoder import FairseqDecoder
from fairseq.modules import (
    MultiheadAttention,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)

@register_model("transformer")
class TransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from "Attention Is All You Need".
    
    Args:
        encoder: Transformer encoder
        decoder: Transformer decoder
    """
    
    def __init__(self, encoder: FairseqEncoder, decoder: FairseqDecoder):
        super().__init__(encoder, decoder)
        
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-embed-dim", type=int, metavar="N",
            help="encoder embedding dimension"
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim", type=int, metavar="N",
            help="encoder embedding dimension for FFN"
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N",
            help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads", type=int, metavar="N",
            help="num encoder attention heads"
        )
        parser.add_argument(
            "--decoder-embed-dim", type=int, metavar="N",
            help="decoder embedding dimension"
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim", type=int, metavar="N",
            help="decoder embedding dimension for FFN"
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N",
            help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads", type=int, metavar="N",
            help="num decoder attention heads"
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D",
            help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout", type=float, metavar="D",
            help="dropout probability for attention weights"
        )
        parser.add_argument(
            "--activation-dropout", type=float, metavar="D",
            help="dropout probability after activation in FFN"
        )
        parser.add_argument(
            "--activation-fn",
            choices=["relu", "gelu"],
            help="activation function to use"
        )
        
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # Build encoder
        encoder = TransformerEncoder(args, task.source_dictionary)
        # Build decoder
        decoder = TransformerDecoder(args, task.target_dictionary)
        # Build model
        return cls(encoder, decoder)

class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder.
    
    Args:
        args: Model arguments
        dictionary: Source dictionary
    """
    
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.dropout = args.dropout
        
        embed_dim = args.encoder_embed_dim
        self.embed_tokens = nn.Embedding(len(dictionary), embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions,
            embed_dim,
            dictionary.pad(),
        )
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(args)
            for _ in range(args.encoder_layers)
        ])
        
    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens: Source tokens
            src_lengths: Source lengths
            
        Returns:
            Encoder output
        """
        # Embed tokens and positions
        x = self.embed_tokens(src_tokens)
        x = x + self.embed_positions(src_tokens)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
            
        return {
            "encoder_out": x,
            "encoder_padding_mask": src_tokens.eq(self.dictionary.pad()),
        }

class TransformerDecoder(FairseqDecoder):
    """
    Transformer decoder.
    
    Args:
        args: Model arguments
        dictionary: Target dictionary
    """
    
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.dropout = args.dropout
        
        embed_dim = args.decoder_embed_dim
        self.embed_tokens = nn.Embedding(len(dictionary), embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions,
            embed_dim,
            dictionary.pad(),
        )
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(args)
            for _ in range(args.decoder_layers)
        ])
        
    def forward(self, prev_output_tokens, encoder_out):
        """
        Args:
            prev_output_tokens: Previous output tokens
            encoder_out: Encoder output
            
        Returns:
            Decoder output
        """
        # Embed tokens and positions
        x = self.embed_tokens(prev_output_tokens)
        x = x + self.embed_positions(prev_output_tokens)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, encoder_out["encoder_out"], encoder_out["encoder_padding_mask"])
            
        return x 