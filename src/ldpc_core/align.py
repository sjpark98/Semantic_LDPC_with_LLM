"""Needleman-Wunsch global alignment with affine gap penalty (Gotoh's algorithm).

Used to re-align an LLM correction output (which may not respect the exact
length M, since the base model has no fine-tuning to enforce it) back onto
the fixed-length corrupted reference so char-position correspondence survives.
"""
NEG_INF = float("-inf")


def nw_align(a: str, b: str, s_match: float = 2.0, s_mismatch: float = -1.0,
             gap_open: float = -2.0, gap_extend: float = -0.5):
    """Align a (reference, e.g. corrupted/received text) against b (LLM output).
    Returns (aligned_a, aligned_b, score) with '-' marking gaps."""
    la, lb = len(a), len(b)
    M = [[NEG_INF] * (lb + 1) for _ in range(la + 1)]
    Ix = [[NEG_INF] * (lb + 1) for _ in range(la + 1)]  # gap in b (a advances)
    Iy = [[NEG_INF] * (lb + 1) for _ in range(la + 1)]  # gap in a (b advances)
    M[0][0] = 0.0
    for i in range(1, la + 1):
        Ix[i][0] = gap_open + (i - 1) * gap_extend
    for j in range(1, lb + 1):
        Iy[0][j] = gap_open + (j - 1) * gap_extend

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            s = s_match if a[i - 1] == b[j - 1] else s_mismatch
            M[i][j] = max(M[i - 1][j - 1], Ix[i - 1][j - 1], Iy[i - 1][j - 1]) + s
            Ix[i][j] = max(M[i - 1][j] + gap_open, Ix[i - 1][j] + gap_extend)
            Iy[i][j] = max(M[i][j - 1] + gap_open, Iy[i][j - 1] + gap_extend)

    # pick best final state
    finals = {"M": M[la][lb], "Ix": Ix[la][lb], "Iy": Iy[la][lb]}
    state = max(finals, key=finals.get)
    score = finals[state]

    aligned_a, aligned_b = [], []
    i, j = la, lb
    while i > 0 or j > 0:
        if state == "M":
            aligned_a.append(a[i - 1])
            aligned_b.append(b[j - 1])
            s = s_match if a[i - 1] == b[j - 1] else s_mismatch
            prev_scores = {"M": M[i - 1][j - 1], "Ix": Ix[i - 1][j - 1], "Iy": Iy[i - 1][j - 1]}
            i, j = i - 1, j - 1
            if i == 0 and j == 0:
                break
            state = max(prev_scores, key=prev_scores.get)
        elif state == "Ix":
            aligned_a.append(a[i - 1])
            aligned_b.append("-")
            open_score = M[i - 1][j] + gap_open
            ext_score = Ix[i - 1][j] + gap_extend
            i -= 1
            state = "M" if open_score >= ext_score else "Ix"
        else:  # Iy
            aligned_a.append("-")
            aligned_b.append(b[j - 1])
            open_score = M[i][j - 1] + gap_open
            ext_score = Iy[i][j - 1] + gap_extend
            j -= 1
            state = "M" if open_score >= ext_score else "Iy"

    aligned_a.reverse()
    aligned_b.reverse()
    return "".join(aligned_a), "".join(aligned_b), score


def project_correction(reference: str, llm_output: str, **kw):
    """Return a length(reference) string: for each position in reference, the
    aligned llm_output character, or the reference char itself if that
    position aligns to a gap (no correction proposed)."""
    aligned_ref, aligned_out, _ = nw_align(reference, llm_output, **kw)
    result = []
    ref_pos = 0
    for ra, oa in zip(aligned_ref, aligned_out):
        if ra == "-":
            continue  # insertion in llm output, doesn't correspond to a reference position
        if oa == "-":
            result.append(ra)  # deletion in llm output -> keep original char
        else:
            result.append(oa)
        ref_pos += 1
    return "".join(result)
