"""(3,6)-regular systematic LDPC: H/G generation + vectorized sum-product BP.

BP decoder takes an externally supplied LLR vector (channel LLR, or channel+LLM
prior after injection) so that the same decoder can be reused for the
BP-only baseline and for the LLM-aided re-decode.
"""
import numpy as np
import pyldpc

LLR_CLIP = 30.0
TANH_EPS = 1e-7


def build_ldpc(n: int, d_v: int = 3, d_c: int = 6, seed: int = 42):
    """Returns H (m,n), G (n,k). Requires regular H (checked)."""
    H, G = pyldpc.make_ldpc(n, d_v=d_v, d_c=d_c, systematic=True, sparse=False, seed=seed)
    row_w = np.unique(H.sum(axis=1))
    col_w = np.unique(H.sum(axis=0))
    assert len(row_w) == 1 and row_w[0] == d_c, f"H not d_c-regular: {row_w}"
    assert len(col_w) == 1 and col_w[0] == d_v, f"H not d_v-regular: {col_w}"
    return H.astype(np.int8), G.astype(np.int8)


def encode(G: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Systematic encode: x[:k] == u."""
    return (G.dot(u) % 2).astype(np.int8)


class BPDecoder:
    """Precomputes edge-index tables for a fixed H so BP calls are cheap."""

    def __init__(self, H: np.ndarray):
        self.H = H
        m, n = H.shape
        self.m, self.n = m, n
        d_c = int(H.sum(axis=1)[0])
        d_v = int(H.sum(axis=0)[0])
        self.d_c, self.d_v = d_c, d_v

        rows, cols = np.nonzero(H)  # row-major order -> grouped by check
        self.row_idx = cols.reshape(m, d_c)  # row_idx[i, a] = var index

        rowsT, colsT = np.nonzero(H.T)  # grouped by var
        self.col_idx = colsT.reshape(n, d_v)  # col_idx[j, b] = check index

        E = m * d_c
        edges_check_row = np.repeat(np.arange(m), d_c)
        edges_var_row = self.row_idx.flatten()
        edge_a = np.tile(np.arange(d_c), m)
        key_row = edges_check_row.astype(np.int64) * n + edges_var_row

        edges_var_col = np.repeat(np.arange(n), d_v)
        edges_check_col = self.col_idx.flatten()
        edge_b = np.tile(np.arange(d_v), n)
        key_col = edges_check_col.astype(np.int64) * n + edges_var_col

        sort_row = np.argsort(key_row)
        sort_col = np.argsort(key_col)
        assert np.array_equal(key_row[sort_row], key_col[sort_col]), "edge key mismatch"

        f_of_e = np.empty(E, dtype=np.int64)
        f_of_e[sort_row] = sort_col
        var_pos_flat = edge_b[f_of_e]
        self.var_pos = var_pos_flat.reshape(m, d_c)  # var_pos[i,a] = position b in col_idx[j]

        e_of_f = np.empty(E, dtype=np.int64)
        e_of_f[sort_col] = sort_row
        check_pos_flat = edge_a[e_of_f]
        self.check_pos = check_pos_flat.reshape(n, d_v)  # check_pos[j,b] = position a in row_idx[i]

    def decode(self, llr_ch: np.ndarray, max_iter: int = 50, mode: str = "sum_product",
               msg_clip: float | None = None, min_sum_alpha: float = 0.8, diagnostics: bool = False):
        """BP decode. mode='sum_product' (default) or 'min_sum' (normalized min-sum,
        more robust to saturation). msg_clip: optional per-iteration internal message
        clamp (separate from the LLR_CLIP applied once to the channel input) -- this is
        the Phase-2 A-1 numerical-diagnosis knob. diagnostics=True returns saturation/
        nan/inf event counters so pathological BP behavior can be distinguished from a
        floating-point artifact.
        Returns dict with bits, converged, n_iter, syndrome_weights, post_llr[, diag]."""
        H, m, n, d_c, d_v = self.H, self.m, self.n, self.d_c, self.d_v
        row_idx, col_idx, var_pos, check_pos = self.row_idx, self.col_idx, self.var_pos, self.check_pos

        llr_ch = np.clip(llr_ch, -LLR_CLIP, LLR_CLIP)
        Vmsg = np.tile(llr_ch[:, None], (1, d_v))  # Vmsg[j,b]

        syndrome_weights = []
        bits_hat = (llr_ch < 0).astype(np.int8)
        converged = False
        n_iter_used = 0
        diag = {"n_nan": 0, "n_inf": 0, "n_saturated": 0, "n_div_clipped": 0}

        for it in range(max_iter):
            n_iter_used = it + 1
            vmsg_gather = Vmsg[row_idx, var_pos]  # (m, d_c)

            if mode == "sum_product":
                arg = np.clip(vmsg_gather, -LLR_CLIP, LLR_CLIP) / 2.0
                t_raw = np.tanh(arg)
                if diagnostics:
                    diag["n_saturated"] += int(np.sum(np.abs(t_raw) >= 1 - TANH_EPS))
                t = np.clip(t_raw, -1 + TANH_EPS, 1 - TANH_EPS)
                prod_row = np.prod(t, axis=1, keepdims=True)
                prod_excl_raw = prod_row / t
                if diagnostics:
                    diag["n_nan"] += int(np.sum(np.isnan(prod_excl_raw)))
                    diag["n_inf"] += int(np.sum(np.isinf(prod_excl_raw)))
                    diag["n_div_clipped"] += int(np.sum(np.abs(prod_excl_raw) > 1 - TANH_EPS))
                prod_excl = np.nan_to_num(prod_excl_raw, nan=0.0, posinf=1 - TANH_EPS, neginf=-1 + TANH_EPS)
                prod_excl = np.clip(prod_excl, -1 + TANH_EPS, 1 - TANH_EPS)
                Cmsg = 2.0 * np.arctanh(prod_excl)  # (m, d_c)
            elif mode == "min_sum":
                signs = np.where(vmsg_gather >= 0, 1.0, -1.0)
                abs_vals = np.abs(vmsg_gather)
                overall_sign = np.prod(signs, axis=1, keepdims=True)
                sign_excl = overall_sign * signs
                sorted_abs = np.sort(abs_vals, axis=1)
                min1 = sorted_abs[:, 0:1]
                min2 = sorted_abs[:, 1:2]
                idx1 = np.argmin(abs_vals, axis=1)
                is_min1 = np.arange(d_c)[None, :] == idx1[:, None]
                min_excl = np.where(is_min1, min2, min1)
                Cmsg = min_sum_alpha * sign_excl * min_excl
            else:
                raise ValueError(f"unknown mode {mode}")

            if msg_clip is not None:
                Cmsg = np.clip(Cmsg, -msg_clip, msg_clip)

            # --- variable update ---
            cmsg_gather = Cmsg[col_idx, check_pos]  # (n, d_v)
            total_var = llr_ch + cmsg_gather.sum(axis=1)
            Vmsg = total_var[:, None] - cmsg_gather
            if msg_clip is not None:
                Vmsg = np.clip(Vmsg, -msg_clip, msg_clip)

            bits_hat = (total_var < 0).astype(np.int8)
            syn = H.dot(bits_hat) % 2
            w = int(syn.sum())
            syndrome_weights.append(w)
            if w == 0:
                converged = True
                break

        result = {
            "bits": bits_hat,
            "post_llr": total_var,
            "converged": converged,
            "n_iter": n_iter_used,
            "syndrome_weights": syndrome_weights,
        }
        if diagnostics:
            result["diag"] = diag
        return result
