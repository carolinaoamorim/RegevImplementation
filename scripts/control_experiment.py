"""Reproduce the quantum-vs-random control table in the README."""
import random
from regev.postprocess import regev_relation_lattice, is_relation, regev_factor
from regev.bases import generate_regev_bases

N, d, M = 77, 3, 32
bases, primes, trivial = generate_regev_bases(N, d)
assert not trivial

VEC = [(4, 2, 17), (28, 30, 15), (19, 26, 13), (13, 6, 19), (2, 17, 9)]
POOL = [(4,2,17),(28,30,15),(19,26,13),(13,6,19),(2,17,9),(30,15,23),
        (15,23,28),(17,9,4),(23,28,30),(9,4,2),(2,17,8),(1,8,20)]

def run(sampler, label, trials=30):
    succ, tot, norms = 0, 0, []
    for _ in range(trials):
        sv = sampler()
        c = regev_relation_lattice(sv, M, d)
        rels = [(nm, e) for nm, e, _ in c if is_relation(e, bases, N)]
        tot += len(rels)
        if rels:
            norms.append(min(nm for nm, _ in rels))
        if regev_factor(sv, M, d, N, bases, primes, verbose=False):
            succ += 1
    print(f"{label:16s} {succ:2d}/{trials}  rel/basis {tot/trials:.2f}  "
            f"mean min |e|1 {sum(norms)/len(norms):.1f}" if norms else
            f"{label:16s} {succ:2d}/{trials}  rel/basis {tot/trials:.2f}  n/a")

random.seed(7)
run(lambda: random.sample(POOL, 5), "quantum (ideal)")
random.seed(7)
run(lambda: [tuple(random.randrange(M) for _ in range(d)) for _ in range(5)], "uniform random")
