import hdfmap

m = hdfmap.create_nexus_map('i06-12345.nxs')

print(m.scannables['energy'])
# >>> '/entry/medipix/energy'

print(m['scan_command'])
# >>> '/entry/scan_command'

with m.load_hdf() as nxs:
    xdata, ydata, error = m.eval(nxs, 'axes0, signal, np.sqrt(signal)')
    energy_label = m.format_hdf(nxs, 'energy = {pgm_energy:.2f}{pgm_energy@units}')
    temp_label = m.format_hdf(nxs, 'T = {(T_sample|lakeshore336_sample?(300))} K')

