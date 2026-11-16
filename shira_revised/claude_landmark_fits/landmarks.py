import numpy as np, scipy.io as sio
G = "/mnt/user-data/uploads/VSD_FM/legacy_code/Retinotopic model Joanna/Gandalf/"
P = sio.loadmat(G + "VSDI points/201118/opt_points_8_5_22.mat")["opt_points"].astype(float)
VF_R, VF_C, BR_R, BR_C = P[:,0], P[:,1], P[:,4], P[:,5]          # stimulus pixel (row,col); camera (row,col)
CAM = BR_C + 1j*BR_R
PPD = 35
X0, X1, Y0, Y1 = 0.0071428571, 6.0642857143, 0.0142857143, 6.6142857143
sech = lambda z: 1/np.cosh(z)
def schira(z, a, al):
    E, th = np.abs(z), np.angle(z); T = th/al
    return np.log(E*np.exp(1j*T*sech(T)**(sech(np.log(E/a)*0.7609)*0.1821)) + a)
def deg_correct(r, c): return c/PPD - 1j*r/PPD
def deg_legacy(r, c):  return (X0 + (c-1)*(X1-X0)/511) - 1j*(Y0 + (r-1)*(Y1-Y0)/300)
def idx(r, c): return int(np.where((VF_R == r) & (VF_C == c))[0][0])
