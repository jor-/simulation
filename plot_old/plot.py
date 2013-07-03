import numpy as np
from pylab import *

J = np.load('/mnt/work_j2_rz/NDOP/oed_data/parameter_set_1/J_12.npy', 'r')
J_PO4 = J[1]
J_DOP = J[0]

sensitivity_DOP = np.sum(np.abs(J[0]), 4)
sensitivity_DOP_annual = np.sum(sensitivity_DOP, 0)
sensitivity_PO4 = np.sum(np.abs(J[1]), 4)
sensitivity_PO4_annual = np.sum(sensitivity_PO4, 0)

for z in range(15):
    subplot(2,1,1)
    imshow(sensitivity_DOP_annual[z], origin='lower', extent=(0, 360, -90, 90))
    title('DOP')
#     xlabel('x')
#     ylabel('y')
    colorbar()
    subplot(2,1,2)
    title('PO4')
    imshow(sensitivity_PO4_annual[z], origin='lower')
    colorbar()
    pause(2)
    clf()

show()



def plot_all_z(S):
    # plot NANs in black
    colormap = cm.jet
    colormap.set_bad('k',1.)
    
    max_S = np.nanmax(S)
    
    for z in range(15):
        imshow(S[z], origin='lower', extent=(0, 360, -90, 90), vmin=0, vmax=max_S)
    #         colorbar()
        pause(2)
        clf()
    
    
