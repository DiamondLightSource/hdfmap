"""
hdfmap example

Generate metadata using namespace expressions

By Dan Porter
27/11/2024 (hdfmap v0.6.2)
"""

import hdfmap

print(hdfmap.version_info())

VALUES = {
    'cmd': '(cmd|scan_command)',
    'date': 'start_time',
    'pol': 'polarisation?("lh")',
    'iddgap': 'iddgap',
    'rowphase': 'idutrp if iddgap == 100 else iddtrp',
    'endstation': 'instrument_name',
    'temp': '(T_sample|lakeshore336_sample?(300))',
    'rot': '(scmth|xabs_theta|ddiff_theta?(0))',
    'field': 'field_x?(0), field_y?(0), field_z?(0)',
    'energy': '(fastEnergy|pgm_energy|energy)',
    'monitor': '(C2|ca62sr)',
    'tey': '(C1|ca61sr)', # / (C2|ca62sr)',
    'tfy': '(C3|ca63sr)', # / (C2|ca62sr)',
}

LABELS = {
    'name': '{filename}: {scan_command}',
    'temp': 'T = {(T_sample|lakeshore336_sample@(300)):.2f}',
    'energy': 'E = {np.mean((fastEnergy|pgm_energy|energy@(0))):.2f} eV',
}

if __name__ == '__main__':
    filename = '../tests/data/i06-353130.nxs'

    m = hdfmap.create_nexus_map(filename)
    with m.load_hdf() as nxs:
        values = {name: m.eval(nxs, expr) for name, expr in VALUES.items()}
        labels = {name: m.format_hdf(nxs, expr) for name, expr in LABELS.items()}

    print('Values:')
    print('\n'.join(f"{name:20}: {value}" for name, value in values.items()))
    print('\n\nLabels:')
    print('\n'.join(f"{name:20}: {value}" for name, value in labels.items()))

    print('\n\nFinished!')