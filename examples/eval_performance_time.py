"""
hdfmap performance benchmarks

### BENCHMARKS
times in ms
| Version | Description | Populate | expression 1 | expression 2 |
| --- | --- | --- | --- | --- |
| 1.0.0 | linux workstation ws559 | 126.9 | 0.85 | 1.01 |


By Dan Porter
29/07/2025 (hdfmap v1.0.0)
"""

import hdfmap
from timeit import timeit


def time_eval(expression):
    result = m.eval(hdf, expression)
    eval_time = timeit(f"m.eval(hdf, '{expression}')", number=iterations, globals=globals())
    print(f"\nevaluating: '{expression}")
    print(f"average time to evaluate in {iterations} iterations: {1000 * eval_time / iterations:.2f} ms")
    # print(f"result correct? {result[0] - 300 < 0.01}")
    print(f"result = {result}")


if __name__ == '__main__':
    filename = '../tests/data/1040323.nxs'

    with hdfmap.load_hdf(filename) as hdf:
        m = hdfmap.NexusMap()
        iterations = 10
        pop_time = timeit('m.populate(hdf)', number=iterations, globals=globals())
        print(f"average populate time in {iterations} iterations: {1000 * pop_time / iterations:.2f} ms")

        time_eval('signal / (counttime|count_time?(1.0)) / Transmission / (rc / 300)')
        time_eval('d_IMAGE[..., pil3_centre_j-30:pil3_centre_j+30, pil3_centre_i-30:pil3_centre_i+30].sum(axis=(-1, -2))')




