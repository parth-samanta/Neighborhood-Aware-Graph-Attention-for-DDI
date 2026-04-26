"""
model.py

Implements the full DDI prediction architecture with ablation flags:

  GatedFusionLayer
      ↓
  RelationAwareMultiHeadNeighborhoodAttention
      ↓
  DDIInteractionMLP

Ablation toggles (passed to DDIModel.__init__):
  use_semantic_only      - disables TransE, uses only Word2Vec
  use_naive_concat       - replaces GatedFusion with plain concatenation + linear
  use_flat_attention     - disables relation-aware attention; uses uniform avg pooling
  use_naive_pair_concat  - replaces 4-part interaction layer with simple [hA || hB]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

EMB_DIM = 128
NUM_NEIGHBORS = 15


# 1. Gated Fusion Layer

class GatedFusionLayer(nn.Module):
    """
    Learns a soft gate g ∈ (0,1)^d to fuse two 128-dim vectors:
        g      = sigmoid( W_g [v1 || v2] + b_g )
        output = g ⊙ v1 + (1-g) ⊙ v2

    The gate is conditioned on both vectors, allowing the model to learn
    which embedding source is more informative per-dimension.
    """

    def __init__(self, dim: int = EMB_DIM):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v1, v2 : [B, dim]  (Word2Vec, TransE)
        Returns:
            fused  : [B, dim]
        """
        g = torch.sigmoid(self.gate(torch.cat([v1, v2], dim=-1)))  # [B, dim]
        return g * v1 + (1.0 - g) * v2                             # [B, dim]


class NaiveConcatFusion(nn.Module):
    """Ablation: concatenate then project back to dim."""

    def __init__(self, dim: int = EMB_DIM):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim, bias=True)

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        return F.relu(self.proj(torch.cat([v1, v2], dim=-1)))



# 2. Relation-Aware Multi-Head Neighborhood Attention


class RelationAwareMultiHeadAttention(nn.Module):
    """
    Multi-head attention that aggregates neighborhood TransE vectors,
    conditioned on real TransE relation embeddings from the knowledge graph.

    Architecture per head h:
        q_h = W_q^h · drug_vec                         [B, d_head]
        k_h = W_k^h · neighbor_vec[n]                  [B, N, d_head]
        r_h = W_r^h · relation_transe_vec[n]           [B, N, d_head]
        score[n] = (q_h · (k_h[n] + r_h[n])) / √d_head
        alpha    = softmax(score)                        [B, H, N]
        v_h      = W_v^h · neighbor_vec                 [B, N, d_head]
        out_h    = Σ_n alpha[n] · v_h[n]                   [B, d_head]

    Heads concatenated and projected back to [B, dim].
    Residual + LayerNorm applied on top of drug_vec.

    The relation embedding is not a learnable lookup . It is the 128-dim TransE
    vector for each typed edge, projected by W_r to allow adaptation while
    preserving the pre-trained geometric structure.
    """

    def __init__(
        self,
        dim: int = EMB_DIM,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim    = dim
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.scale  = math.sqrt(self.d_head)

        # Learnable projections
        self.W_q = nn.Linear(dim, dim, bias=False)   # query from drug
        self.W_k = nn.Linear(dim, dim, bias=False)   # key from neighbor node
        self.W_v = nn.Linear(dim, dim, bias=False)   # value from neighbor node
        self.W_r = nn.Linear(dim, dim, bias=False)   # projects relation TransE vec

        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_q, self.W_k, self.W_v, self.W_r, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, drug_vec, neighbor_vecs, relation_vecs):
      B, N, D = neighbor_vecs.shape
      H, d = self.n_heads, self.d_head

      Q = self.W_q(drug_vec).view(B, H, d).unsqueeze(2)              # [B, H, 1, d]
      K = self.W_k(neighbor_vecs).view(B, N, H, d).permute(0,2,1,3) # [B, H, N, d]
      V = self.W_v(neighbor_vecs).view(B, N, H, d).permute(0,2,1,3) # [B, H, N, d]
      R = self.W_r(relation_vecs).view(B, N, H, d).permute(0,2,1,3) # [B, H, N, d]

      scores = torch.matmul(Q, (K + R).transpose(-1, -2)) / self.scale  # [B, H, 1, N]
      alpha  = self.dropout(torch.softmax(scores, dim=-1))

      agg = torch.matmul(alpha, V).squeeze(2).reshape(B, D)  # [B, D]
      agg = self.out_proj(agg)

      return self.norm(drug_vec + agg)        


class FlatAttentionAggregation(nn.Module):
    """
    Ablation: simple mean pooling of neighbor vectors — no attention weights,
    no relation conditioning. Residual + LayerNorm retained for fair comparison.

    Accepts the same (drug_vec, neighbor_vecs, relation_vecs) signature as
    RelationAwareMultiHeadAttention so it is a drop-in swap; relation_vecs ignored.
    """

    def __init__(self, dim: int = EMB_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        drug_vec: torch.Tensor,       # [B, dim]
        neighbor_vecs: torch.Tensor,  # [B, N, dim]
        relation_vecs: torch.Tensor,  # [B, N, dim]  – ignored in flat ablation
    ) -> torch.Tensor:
        agg = neighbor_vecs.mean(dim=1)   # [B, dim]
        return self.norm(drug_vec + agg)  # [B, dim]


class NoRelationsAttention(nn.Module):
    """
    Ablation: relation-agnostic multi-head attention over neighbor nodes.

    Identical to RelationAwareMultiHeadAttention but the W_r projection is
    removed and relation embeddings are replaced with zeros at runtime.
    This isolates the contribution of typed-edge (relation) information:

        score[n] = (q_h · k_h[n]) / √d_head       ← no R term
        alpha        = softmax(score)
        out      = Σ_n alpha[n] · v_h[n]

    """

    def __init__(
        self,
        dim: int = EMB_DIM,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim     = dim
        self.n_heads = n_heads
        self.d_head  = dim // n_heads
        self.scale   = math.sqrt(self.d_head)

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        
        # No W_r - relation signal is entirely absent

        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_q, self.W_k, self.W_v, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        drug_vec: torch.Tensor,       # [B, dim]
        neighbor_vecs: torch.Tensor,  # [B, N, dim]
        relation_vecs: torch.Tensor,  # [B, N, dim]  – ignored; zeroed out
    ) -> torch.Tensor:
        B, N, D = neighbor_vecs.shape
        H, d    = self.n_heads, self.d_head

        Q = self.W_q(drug_vec).view(B, H, d).unsqueeze(2)               # [B, H, 1, d]
        K = self.W_k(neighbor_vecs).view(B, N, H, d).permute(0, 2, 1, 3)  # [B, H, N, d]
        V = self.W_v(neighbor_vecs).view(B, N, H, d).permute(0, 2, 1, 3)  # [B, H, N, d]
        # R is intentionally absent — keys are not relation-conditioned

        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale       # [B, H, 1, N]
        alpha  = self.dropout(torch.softmax(scores, dim=-1))

        agg = torch.matmul(alpha, V).squeeze(2).reshape(B, D)            # [B, D]
        agg = self.out_proj(agg)

        return self.norm(drug_vec + agg)                                  # [B, dim]


# 3. DDI Interaction MLP


class DDIInteractionMLP(nn.Module):
    """
    4-part interaction:  [hA || hB || hA⊙hB || |hA-hB|]  -> MLP -> scalar logit
    input_dim = 4 * drug_dim
    """

    def __init__(
        self,
        drug_dim: int = EMB_DIM,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 64]

        in_dim = drug_dim * 4
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, hA: torch.Tensor, hB: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hA, hB : [B, drug_dim]
        Returns:
            logit  : [B]
        """
        interaction = torch.cat(
            [hA, hB, hA * hB, torch.abs(hA - hB)], dim=-1
        )  # [B, 4*dim]
        return self.mlp(interaction).squeeze(-1)   # [B]


class NaivePairConcatMLP(nn.Module):
    """Ablation: simple [hA || hB] -> MLP -> scalar logit."""

    def __init__(
        self,
        drug_dim: int = EMB_DIM,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 64]

        in_dim = drug_dim * 2
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, hA: torch.Tensor, hB: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([hA, hB], dim=-1)).squeeze(-1)



# 4. Full DDI Model (with ablation switches)


class DDIModel(nn.Module):
    """
    End-to-end DDI prediction model.

    Ablation flags
    ──────────────
    use_semantic_only     : bool  - Word2Vec identity only; no TransE fusion AND
                                    no graph neighbourhood aggregation. Pure
                                    semantic baseline.
    use_naive_concat      : bool  - replace GatedFusion with NaiveConcatFusion
                                    (keeps graph context, changes fusion method)
    use_flat_attention    : bool  - replace relation-aware attention with mean
                                    pooling (keeps graph context, drops relation
                                    signal and attention weighting)
    use_naive_pair_concat : bool  - replace 4-part interaction with [hA||hB]
    use_no_neighborhood   : bool  - skip neighbourhood aggregation entirely;
                                    drug rep = fused (W2V + TransE) only.
                                    Proves that neighbourhood reasoning helps.
    use_no_relations      : bool  - use relation-agnostic attention (NoRelationsAttention);
                                    neighbours are attended but edge types are ignored.
                                    Proves that typed-relation info helps.

    Inference
    ─────────
    forward(drug_a_idx, drug_b_idx, store) -> logit [B]
    """

    def __init__(
        self,
        dim: int = EMB_DIM,
        n_heads: int = 4,
        n_neighbors: int = NUM_NEIGHBORS,
        mlp_hidden: list[int] | None = None,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.3,
        # Ablation flags 
        use_semantic_only: bool = False,
        use_structural_only: bool = False,
        use_naive_concat: bool = False,
        use_flat_attention: bool = False,
        use_naive_pair_concat: bool = False,
        use_no_neighborhood: bool = False,
        use_no_relations: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.use_semantic_only  = use_semantic_only
        self.use_structural_only = use_structural_only
        self.use_no_neighborhood = use_no_neighborhood
        self.use_no_relations    = use_no_relations

        self.register_buffer("_device_probe", torch.zeros(1))

        # Fusion layer
        if use_semantic_only:
            self.fusion = None
            self.neighborhood_attn = None
        elif use_no_neighborhood:
            # Keep W2V+TransE fusion; skip neighbourhood aggregation entirely.
            # Drug rep = fused vector only - no graph context.
            if use_naive_concat:
                self.fusion = NaiveConcatFusion(dim)
            else:
                self.fusion = GatedFusionLayer(dim)
            self.neighborhood_attn = None
        elif use_structural_only:
            self.fusion = None
            if use_flat_attention:
                self.neighborhood_attn = FlatAttentionAggregation(dim)
            elif use_no_relations:
                self.neighborhood_attn = NoRelationsAttention(
                    dim=dim, n_heads=n_heads, dropout=attn_dropout,
                )
            else:
                self.neighborhood_attn = RelationAwareMultiHeadAttention(
                    dim=dim, n_heads=n_heads, dropout=attn_dropout,
                )
        else:
            if use_naive_concat:
                self.fusion = NaiveConcatFusion(dim)
            else:
                self.fusion = GatedFusionLayer(dim)

            if use_flat_attention:
                self.neighborhood_attn = FlatAttentionAggregation(dim)
            elif use_no_relations:
                self.neighborhood_attn = NoRelationsAttention(
                    dim=dim, n_heads=n_heads, dropout=attn_dropout,
                )
            else:
                self.neighborhood_attn = RelationAwareMultiHeadAttention(
                    dim=dim, n_heads=n_heads, dropout=attn_dropout,
                )
        # Interaction MLP
        if use_naive_pair_concat:
            self.interaction = NaivePairConcatMLP(dim, mlp_hidden, mlp_dropout)
        else:
            self.interaction = DDIInteractionMLP(dim, mlp_hidden, mlp_dropout)

    # Drug encoding (shared for A and B)

    def _encode_drug(
        self,
        drug_indices: torch.Tensor,    # [B]
        store,                         # EmbeddingStore
    ) -> torch.Tensor:
        """Returns contextualized drug vector [B, dim]."""

        device = self._device_probe.device

        w2v, transe = store.get_drug_embeddings(drug_indices)  # [B, dim] each
        
        w2v    = w2v.to(device)
        transe = transe.to(device)

        # Step 1: Fuse W2V + TransE (skipped for semantic-only)
        if self.use_semantic_only:
            return w2v # [B, dim]
        if self.use_structural_only:
            fused = transe
        else:
            fused = self.fusion(w2v, transe) # [B, dim]

        # Step 2: Relation-aware neighbourhood aggregation
        if self.use_no_neighborhood:

            return fused  # [B, dim]

        nb_vecs, rel_vecs = store.get_neighbor_and_relation_embeddings(drug_indices)
        nb_vecs  = nb_vecs.to(device)
        rel_vecs = rel_vecs.to(device)

        contextualized = self.neighborhood_attn(fused, nb_vecs, rel_vecs)  # [B, dim]
        return contextualized

    # Forward

    def forward(
        self,
        drug_a_idx: torch.Tensor,   # [B]
        drug_b_idx: torch.Tensor,   # [B]
        store,                      # EmbeddingStore
    ) -> torch.Tensor:
        """
        Returns raw logit [B].  Apply sigmoid externally for probabilities.
        """
        hA = self._encode_drug(drug_a_idx, store)   # [B, dim]
        hB = self._encode_drug(drug_b_idx, store)   # [B, dim]
        return self.interaction(hA, hB)             # [B]


    @property
    def config_str(self) -> str:
        flags = []
        if self.use_semantic_only:    flags.append("SemanticOnly")
        if self.use_structural_only:  flags.append("StructuralOnly")
        if self.use_no_neighborhood:  flags.append("NoNeighborhood")
        if self.use_no_relations:     flags.append("NoRelations")
        if isinstance(self.fusion, NaiveConcatFusion): flags.append("NaiveConcat")
        if isinstance(self.neighborhood_attn, FlatAttentionAggregation): flags.append("FlatAttn")
        if isinstance(self.neighborhood_attn, NoRelationsAttention): pass  # already captured above
        if isinstance(self.interaction, NaivePairConcatMLP): flags.append("NaivePair")
        return "Full" if not flags else "+".join(flags)