from typing import Optional, Any, Union, Callable, Tuple
from typing import TYPE_CHECKING
import warnings
if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int

import torch
import math
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn.init import xavier_uniform_
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from torch.nn import Embedding
from torch.nn import CrossEntropyLoss
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.transformer import Transformer


# This model is from torch github https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
# I just edited some parts to use it as transformer_VAE
# Transformer VAE paper : https://ieeexplore.ieee.org/document/9054554
# notion : https://www.notion.so/binne/YAI-X-POZAlabs-852ef538af984d99abee33037751547c

class Transformer_CVAE(Module):
    r"""
    arguments are replaced to cfg which is in commu/model/config_helperCVAE
    Args at cfg:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
    """

    def __init__(self, cfg, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer_CVAE, self).__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")

        d_model = cfg.MODEL.units
        d_latent = cfg.MODEL.latent_dim
        nhead = cfg.MODEL.num_heads
        num_encoder_layers = cfg.MODEL.num_layers
        num_decoder_layers = cfg.MODEL.num_layers
        dim_feedforward = cfg.MODEL.inner_size
        pad_idx = cfg.MODEL.pad_index
        vocab_size = cfg.MODEL.vocab_size
        seq_len = cfg.TRAIN.tgt_length
        cdt_len = cfg.MODEL.meta_length
        dropout = cfg.MODEL.dropout
        activation = F.relu
        layer_norm_eps = cfg.MODEL.layer_norm_eps
        beta = cfg.MODEL.beta
        batch_first = False
        norm_first = False
        num_buckets = cfg.MODEL.num_buckets

        encoder_layer = T5TransformerEncoderLayer(d_model, nhead, num_buckets, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                **factory_kwargs)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.sample_layer = VAEsample(d_model, d_latent, batch_first, **factory_kwargs)

        decoder_layer = T5TransformerDecoderLayer(d_model, nhead, num_buckets, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                **factory_kwargs)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.d_latent = d_latent
        self.seq_len = seq_len
        self.nhead = nhead
        self.batch_first = batch_first
        self.device = device
        self.cdt_len = cdt_len
        self.num_buckets = num_buckets

        self.local_encoder = Linear(d_model, d_model, **factory_kwargs)
        self.local_decoder = Linear(d_model, vocab_size, **factory_kwargs)
        self.condition_embed = TokenEmbedding(d_latent, vocab_size=vocab_size, **factory_kwargs)
        self.src_tgt_embed = TokenEmbedding(d_model, vocab_size=vocab_size, **factory_kwargs)
        self.criterion = VAE_Loss(beta=beta, pad_idx=pad_idx) # beta = 1 from Transformer VAE paper, pad_idx from ComMU dataset

    def forward_(self, src: Tensor, tgt: Tensor, cdt: Tensor, latent: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the Tensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the Tensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the Tensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
              `(N, S, E)` if `batch_first=True`.
            - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.
            - src_mask: :math:`(S, S)` or :math:`(N\cdot\text{num\_heads}, S, S)`.
            - tgt_mask: :math:`(T, T)` or :math:`(N\cdot\text{num\_heads}, T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decoder.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> # xdoctest: +SKIP
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        encoder_out = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        if latent is not None:
            encoder_out += latent
        
        latent_sample, latent_mu, latent_logvar = self.sample_layer(encoder_out, cdt)
        output = self.decoder(tgt, latent_sample, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return (output, latent_mu, latent_logvar)
    
    # src : word sequence
    # tgt : word sequence that starts with start token
    # cdt : condition sequence
    def forward(self, src: Tensor, tgt: Tensor, cdt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        
        if self.batch_first:
            seq_len = src.size(1)
        else:
            seq_len = src.size(0)
        src_tgt_mask = self.generate_square_subsequent_mask(seq_len, device=self.device)
        cross_attention_mask = self.generate_rectangle_subsequent_mask(seq_len, self.cdt_len, device=self.device)
        
        src_embed = self.src_tgt_embed(src)
        tgt_embed = self.src_tgt_embed(tgt)
        cdt_embed = self.condition_embed(cdt)

        src_embed = self.local_encoder(src_embed)
        tgt_embed = self.local_encoder(tgt_embed)
        out, latent_mu, latent_logvar = self.forward_(tgt_embed, tgt_embed, cdt_embed, None, src_tgt_mask, src_tgt_mask, cross_attention_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        pred = F.log_softmax(self.local_decoder(out), dim = 2)
        loss, nll = self.criterion(pred, src, latent_mu, latent_logvar)
        return (loss, nll, pred)
    
    # tgt : word sequence that starts with start token, filled with padding at the first step of generation
    def forward_generate(self, tgt: Tensor, latent: Tensor, cdt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        if self.batch_first:
            seq_len = tgt.size(1)
        else:
            seq_len = tgt.size(0)
        src_tgt_mask = self.generate_square_subsequent_mask(seq_len, device=self.device)
        src_tgt_mask = self.generate_square_subsequent_mask(seq_len, device=self.device)
        cross_attention_mask = self.generate_rectangle_subsequent_mask(seq_len, self.cdt_len, device=self.device)
        
        tgt_embed = self.src_tgt_embed(tgt)
        cdt_embed = self.condition_embed(cdt)

        tgt_embed = self.local_encoder(tgt_embed)
        out, latent_mu, latent_logvar = self.forward_(tgt_embed, tgt_embed, cdt_embed, latent, src_tgt_mask, src_tgt_mask, cross_attention_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        pred = F.log_softmax(self.local_decoder(out), dim = 2)
        return pred



    @staticmethod
    def generate_square_subsequent_mask(sz: int, device='cpu') -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
    
    @staticmethod
    def generate_rectangle_subsequent_mask(sz: int, cdt: int, device='cpu') -> Tensor:
        return torch.triu(torch.full((sz, sz + cdt), float('-inf'), device=device), diagonal=cdt + 1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class T5TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:
        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)
        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, num_buckets: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.device = device

        self.relative_attention_num_buckets = num_buckets
        self.relative_attention_bias_sa = Embedding(self.relative_attention_num_buckets, nhead, **factory_kwargs)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as src_mask.
              Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = _canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=_none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )


        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        mask_bias = attn_mask + self.compute_bias(x.size(1), x.size(0), x.size(0), self.relative_attention_bias_sa)
        x = self.self_attn(x, x, x,
                           attn_mask=mask_bias,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    def _relative_position_bucket(self, relative_position, bidirectional=False, num_buckets=128, max_distance=128):
        r"""
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        return relative_position

    def compute_bias(self, batch_size, query_length, key_length, relative_attention_bias):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length),
            num_buckets=self.relative_attention_num_buckets,
        )
        values = relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).repeat(batch_size, 1, 1) # shape (num_heads, query_length, key_length)
        return values



class T5TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, num_buckets: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.device = device

        self.relative_attention_num_buckets = num_buckets
        self.relative_attention_bias_sa = Embedding(self.relative_attention_num_buckets, nhead, **factory_kwargs)
        self.relative_attention_bias_mha = Embedding(self.relative_attention_num_buckets + 11, nhead, **factory_kwargs)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing tgt_mask. Default: ``False``.
            memory_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing memory_mask. Default: ``False``.
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        mask_bias = attn_mask + self.compute_bias(x.size(1), x.size(0), x.size(0), self.relative_attention_bias_sa)
        x = self.self_attn(x, x, x,
                           attn_mask=mask_bias,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        mask_bias = attn_mask + self.compute_bias(x.size(1), x.size(0), mem.size(0), self.relative_attention_bias_mha)
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=mask_bias,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
    
    def _relative_position_bucket(self, relative_position, bidirectional=False, num_buckets=128):
        r"""
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        q_len = relative_position.size(0)
        k_len = relative_position.size(1)
        if q_len != k_len:
            relative_position[:, k_len - q_len:] = relative_position[:, :q_len]
            relative_position[:, :k_len - q_len] = torch.arange(0, k_len - q_len, dtype=torch.long, device=self.device)[None, :].repeat(q_len, 1) + num_buckets
        return relative_position

    def compute_bias(self,batch_size, query_length, key_length, relative_attention_bias):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length),
            num_buckets=self.relative_attention_num_buckets,
        )
        values = relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).repeat(batch_size, 1, 1) # shape (num_heads, query_length, key_length)
        return values


class VAEsample(Module):
    def __init__(self, d_model: int, d_latent: int, batch_first: bool, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(VAEsample, self).__init__()
        self.device = device
        self.d_latent = d_latent
        self.d_model = d_model
        if batch_first:
            self.seq_pos = 1
        else:
            self.seq_pos = 0
        self.model2sample = Linear(d_model, d_latent * 2, **factory_kwargs)
        self.latent2model = Linear(d_latent, d_model, **factory_kwargs)
    
    # input size : 
    # (batch_size, seq_length, d_model) if batch_first is True
    # (seq_length, batch_size, d_model) if batch_first is False
    def forward(self, x, cdt):
        lin_out = self.model2sample(x)
        latent_mean = lin_out[:, :, :self.d_latent]
        latent_logvar = lin_out[:, :, self.d_latent:]
        latent_std = torch.exp(0.5*latent_logvar)
        norm_sample = torch.normal(mean = 0, std = 1, size=latent_mean.size(), device=self.device)
        out = latent_mean + latent_std * norm_sample
        out = torch.concat((cdt, out), dim = self.seq_pos)
        out = self.latent2model(out)
        return out, latent_mean, latent_logvar


class TokenEmbedding(Module):
    def __init__(self, d_model, vocab_size, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TokenEmbedding, self).__init__()
        self.embedding = Embedding(vocab_size, d_model, **factory_kwargs)
        self.d_model = d_model
    
    # input size : 
    # (batch_size, seq_length) if batch_first is True
    # (seq_length, batch_size) if batch_first is False
    def forward(self, x) -> Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


# class PositionalEncoding(Module):
#     def __init__(self, d_model, max_len, batch_first: bool = False, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(PositionalEncoding, self).__init__()

#         self.batch_first = batch_first
#         encoding = torch.zeros(max_len, d_model)
#         encoding.requires_grad = False
#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
#         encoding[:, 0::2] = torch.sin(position * div_term)
#         encoding[:, 1::2] = torch.cos(position * div_term)
#         if batch_first:
#             self.encoding = encoding.unsqueeze(0).to(device)
#         else:
#             self.encoding = encoding.unsqueeze(1).to(device)
    
#     # input size : 
#     # (batch_size, seq_length, d_model) if batch_first is True
#     # (seq_length, batch_size, d_model) if batch_first is False
#     def forward(self, x) -> Tensor:
#         if self.batch_first : # input size : (batch_size, seq_length, d_model)
#             seq_len = x.size(1)
#             pos_embed = self.encoding[:, :seq_len, :]
#         else :                # input size : (seq_length, batch_size, d_model)
#             seq_len = x.size(0)
#             pos_embed = self.encoding[:seq_len, :, :]
#         return x + pos_embed

# class TransformerEmbedding(Module):
#     def __init__(self, d_model: int, vocab_size: int, max_len: int, batch_first: bool = False, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(TransformerEmbedding, self).__init__()
#         self.embed = TokenEmbedding(d_model=d_model, vocab_size=vocab_size, **factory_kwargs)
#         self.positional = PositionalEncoding(d_model=d_model, max_len=max_len, batch_first=batch_first, **factory_kwargs)
    
#     # input size : 
#     # (batch_size, seq_length) if batch_first is True
#     # (seq_length, batch_size) if batch_first is False
#     def forward(self, x) -> Tensor:
#         return self.positional(self.embed(x))

class VAE_Loss(Module):
    def __init__(self, beta: int, pad_idx: int) -> None:
        super(VAE_Loss, self).__init__()
        self.beta = beta
        self.reconstruction_loss = CrossEntropyLoss(ignore_index=pad_idx, reduction='mean')
    
    def forward(self, pred: Tensor, output: Tensor, latent_mu: Tensor, latent_logvar: Tensor) -> Tensor:
        kld_loss = - 0.5 * torch.mean(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())
        sz = output.size(0) * output.size(1)
        rec_loss = self.reconstruction_loss(pred.contiguous().view(sz, -1), output.contiguous().view(sz))

        return kld_loss * self.beta + rec_loss, rec_loss

def _canonical_mask(
        mask: Optional[Tensor],
        mask_name: str,
        other_type: Optional[DType],
        other_name: str,
        target_type: DType,
        check_other: bool = True,
) -> Optional[Tensor]:

    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported")
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask = (
                torch.zeros_like(mask, dtype=target_type)
                .masked_fill_(mask, float("-inf"))
            )
    return mask

def _none_or_dtype(input: Optional[Tensor]) -> Optional[DType]:
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))