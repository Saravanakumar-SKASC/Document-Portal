"""Maximal Marginal Relevance retrieval: relevant chunks that are also diverse."""

from utils.config_loader import load_config


def build_mmr_retriever(vectorstore, k: int | None = None, fetch_k: int | None = None,
                        lambda_mult: float | None = None):
    """
    Fetch ``fetch_k`` candidates by similarity, then pick ``k`` of them that balance
    relevance and diversity. ``lambda_mult`` = 1.0 is pure relevance, 0.0 is pure diversity.
    With several documents this stops one file from crowding out all the others.
    """
    cfg = load_config()["retriever"]
    mmr_cfg = cfg.get("mmr", {})
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k or cfg["top_k"],
            "fetch_k": fetch_k or mmr_cfg.get("fetch_k", 20),
            "lambda_mult": lambda_mult if lambda_mult is not None else mmr_cfg.get("lambda_mult", 0.5),
        },
    )
