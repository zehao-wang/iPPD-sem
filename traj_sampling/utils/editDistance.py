"""
A modified version of edit distance to evaluate diff
between path observation and objects mentioned in instruction
"""

def levenshteinDistance(s, t):
    """
    Args:
        s: instruction obj list, list
        t: path observations, list of list
    """
    s = [' '] + s
    t = [[]] + t
    d = {}
    S = len(s)
    T = len(t)
    for i in range(S):
        d[i, 0] = i
    for j in range (T):
        d[0, j] = j
    for j in range(1,T):
        for i in range(1,S):
            if s[i] in t[j]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j], d[i, j-1], d[i-1, j-1]) + 1
    return d[S-1, T-1], d