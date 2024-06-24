from MILSH_KDE import g_MI

def cond_indep_test(data, target, var, cond_set=[], alpha=0.01):
    pval, dep = g_MI(data, target, var, cond_set, alpha)
    return pval, dep
