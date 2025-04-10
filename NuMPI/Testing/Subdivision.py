import numpy as np
from primefac import primefac


def suggest_subdivisions(nb_dims, nb_procs):
    facs = list(primefac(nb_procs))
    if len(facs) < nb_dims:
        return facs + [1] * (nb_dims - len(facs))
    return facs[: nb_dims - 1] + [np.prod(facs[nb_dims - 1:])]
