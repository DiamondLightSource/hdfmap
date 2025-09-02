"""
HdfMap Example - Define Regions of Interest

Requires HdfMap V1.0.1
"""

import hdfmap
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filename = '../tests/data/1049598.nxs'

    m = hdfmap.create_nexus_map(filename)
    # Nexus file contains datasets 'pil3_centre_j' defining the detector centre
    cen_i = 'pil3_centre_j'
    cen_j = 'pil3_centre_i'
    cen_j2 = 'pil3_centre_i-50'
    wid_i = 61
    wid_j = 21
    m.add_roi('my_roi1', cen_i, cen_j, wid_i, wid_j)
    m.add_roi('my_roi2', cen_i, cen_j2, wid_i, wid_j)
    with m.load_hdf() as nxs:
        ij_box1 = m.eval(nxs, 'my_roi1_box')
        ij_box2 = m.eval(nxs, 'my_roi2_box')
        roi1 = m.eval(nxs, 'my_roi1')
        roi2 = m.eval(nxs, 'my_roi2')
        roi1_total = m.eval(nxs, 'my_roi1_total')
        roi2_total = m.eval(nxs, 'my_roi2_total')
        total = m.eval(nxs, 'total')
        idx1 = np.argmax(roi1_total)
        idx2 = np.argmax(roi2_total)
        image1 = m.get_image(nxs, idx1)
        image2 = m.get_image(nxs, idx2)

    # Plot the ROIs
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), dpi=100)
    fig.subplots_adjust(top=0.93, bottom=0.08)
    axes[0].imshow(image1)
    axes[0].plot(ij_box1[:, 1], ij_box1[:, 0], 'k-', lw=2, label='roi1')
    axes[0].plot(ij_box2[:, 1], ij_box2[:, 0], 'w-', lw=2, label='roi2')
    axes[0].legend()
    axes[0].set_title(f"index = {idx1}")
    axes[1].imshow(image2)
    axes[1].plot(ij_box1[:, 1], ij_box1[:, 0], 'k-', lw=2, label='roi1')
    axes[1].plot(ij_box2[:, 1], ij_box2[:, 0], 'w-', lw=2, label='roi2')
    axes[1].legend()
    axes[1].set_title(f"index = {idx2}")
    axes[2].plot(total, label='total')
    axes[2].plot(roi1_total, label='roi1 total')
    axes[2].plot(roi2_total, label='roi2 total')
    axes[2].legend()

    plt.show()

