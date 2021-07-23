##############################################################################
# This code is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: sac_method.py
#
# Description: Generates spherical point set from integer-pair sequences using 
#   spherical area coordinate (SAC) method
#
# Example commands to run: 
#   python sac_method.py --mn 1,1 2,0 2,0 2,0 2,0 2,0 --plot
#   python sac_method.py --mn 10,0 --plot
#   python sac_method.py --mn 4,0 4,0 --plot
#
##############################################################################

import numpy as np
import numba as nb
from numpy import array as npa
from scipy.spatial import SphericalVoronoi,ConvexHull
from numpy.linalg import solve
from numpy import tan,cos,sin
from numpy import arccos as acos
import matplotlib.pyplot as plt
from matplotlib import cm,colors


def dotv(v1,v2):
    return np.sum(v1*v2,axis=-1)

def norm2(v1): #squared L2-norm
    return dotv(v1,v1)

def normv(v1): #L2-norm
    return np.sqrt(norm2(v1))

def convex_hull_out(V):
    hull = ConvexHull(V)
    K = hull.simplices
    A = V[K[:,0]]
    B = V[K[:,1]]
    C = V[K[:,2]]
    nor = np.cross(B-A,C-A)
    cent = 1/3*(A+B+C)
    rev = np.sum(cent*nor,axis=-1)<0
    K[rev,:]=K[rev,::-1]
    return K

def get_edge_len(V,K):
    edge_len = np.c_[\
               normv(V[K[:,0],:]-V[K[:,1],:]),\
               normv(V[K[:,1],:]-V[K[:,2],:]),\
               normv(V[K[:,2],:]-V[K[:,0],:])]
    return edge_len

def get_triangle_quality(edge_len):
    tri_quality = np.min(edge_len,axis=-1)/np.max(edge_len,axis=-1)
    return tri_quality

def get_covering_radii(V):
    sv = SphericalVoronoi(V)
    vor_V = sv.vertices
    vor_K = sv.regions
    assert np.allclose(normv(vor_V),1.)
    assert len(vor_K) == V.shape[0]

    cover_rad = np.zeros((len(vor_K),))
    for i in range(len(vor_K)):
        cover_rad[i] = np.max(normv(vor_V[vor_K[i],:]-V[i,:]))
    return cover_rad,sv

def get_mesh_ratio(cover_rad,edge_len):
    return np.max(cover_rad)/np.min(edge_len)

def get_icosahedron():
    PHI = 0.5*(1.0+np.sqrt(5.0)) #golden ratio
    V = npa([[0.,1,PHI],[0,-1,PHI],[0,-1,-PHI],\
             [0,1,-PHI],[PHI,0,1],[-PHI,0,1],\
             [-PHI,0,-1],[PHI,0,-1],[1,PHI,0],\
             [-1,PHI,0],[-1,-PHI,0],[1,-PHI,0]])/np.sqrt((1+PHI**2))
    return V

def get_octahedron():
    V = npa([[1.,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    return V

def sph_tri_area(p1,p2,p3):
    d12 = dotv(p1,p2)
    d23 = dotv(p2,p3)
    d31 = dotv(p3,p1)
    return 2*acos((1+d12+d23+d31)/np.sqrt(2*(1+d12)*(1+d23)*(1+d31)))

def sph_lerp(p1,p2,a):
   assert a>0 
   assert a<1 
   dp = dotv(p1,p2)
   assert np.all(np.abs(dp)<=1)
   alp1 = a*acos(dp)
   alp2 = (1.0-a)*acos(dp)
   denom = dp**2 - 1.0
   ca1 = cos(alp1)
   ca2 = cos(alp2)
   c1 = (dp*ca2-ca1)/denom
   c2 = (dp*ca1-ca2)/denom
   return p1*c1[:,None] + p2*c2[:,None]

@nb.jit(nopython=True,parallel=True)
def nb_solve(P,mu,r1,r2,r3,t1,t2,t3,s1,s2,s3,la,lb,lz):
    # see Equations (8)--(10) from https://doi.org/10.3390/app10020655
    for i in nb.prange(P.shape[0]):
        g1 = tan(0.5*mu[i]*la)
        g2 = tan(0.5*mu[i]*lb)
        g3 = tan(0.5*mu[i]*lz)
        Mv = npa([[r1[i][0]-g1*t1[i][0],r1[i][1]-g1*t1[i][1],r1[i][2]-g1*t1[i][2]],\
                  [r2[i][0]-g2*t2[i][0],r2[i][1]-g2*t2[i][1],r2[i][2]-g2*t2[i][2]],\
                  [r3[i][0]-g3*t3[i][0],r3[i][1]-g3*t3[i][1],r3[i][2]-g3*t3[i][2]]])
        bv = npa([g1*(1+s1[i]),g2*(1+s2[i]),g3*(1+s3[i])])
        P[i]=solve(Mv,bv)

def get_sac_point_set(V,K,m,n):
    assert m>0
    assert n>=0
    assert m>=n
    gmn = m*m+m*n+n*n
    N = (V.shape[0]-2)*gmn

    M = npa([[m+n,n],[-n,m]])
    edge0 = K[:,1]>K[:,2]
    edge1 = K[:,2]>K[:,0]
    edge2 = K[:,0]>K[:,1]

    A = V[K[:,0]]
    B = V[K[:,1]]
    Z = V[K[:,2]]

    Vp = np.copy(V)

    # see Equations (8)--(10) from https://doi.org/10.3390/app10020655
    p1 = A
    p2 = B
    p3 = Z

    r1 = np.cross(p2,p3)
    r2 = np.cross(p3,p1)
    r3 = np.cross(p1,p2)

    t1 = p2+p3
    t2 = p3+p1
    t3 = p1+p2

    s1 = dotv(p2,p3)
    s2 = dotv(p3,p1)
    s3 = dotv(p1,p2)
    mu = sph_tri_area(A,B,Z);

    for lp in range(m+n+1):
        for qp in range(m+n+1):
            mp = lp-qp
            ll2 = M @ npa([mp,qp])
            ll3 = npa([ll2[0],ll2[1],gmn-np.sum(ll2)])
            nnz = np.sum(ll3>0) #1 is vertex, #2 is edge, #3 is inside

            if np.any(ll3<0): #outside T_mn
                continue
            if nnz==1: #original vertex
                continue

            la,lb = ll2/gmn
            lz = 1.0-(la+lb)
            if nnz==3: #face
                P = np.zeros((A.shape[0],3))
                nb_solve(P,mu,r1,r2,r3,t1,t2,t3,s1,s2,s3,la,lb,lz)
                Vp = np.r_[Vp,P]
                del P
            elif nnz==2:
                if ll3[2]==0: #edge2
                    assert lz==0
                    P = sph_lerp(A[edge2],B[edge2],la)
                elif ll3[1]==0: #edge1
                    assert lb==0
                    P = sph_lerp(A[edge1],Z[edge1],la)
                elif ll3[0]==0: #edge0
                    assert la==0
                    P = sph_lerp(B[edge0],Z[edge0],lb)

                P /= normv(P)[:,None]
                Vp = np.r_[Vp,P]
                del P
    assert Vp.shape[0] == 2+(V.shape[0]-2)*gmn
    Vp /= normv(Vp)[:,None]
    return Vp


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true',help='draw spherical point set')
    parser.add_argument('--mn', nargs='+',help='list of integers e.g. --mn 1,1 2,0 2,0 2,0 2,0)')
    parser.set_defaults(plot=False)
    parser.set_defaults(mn=[])

    args = parser.parse_args()
    print(args)

    mn_array = npa([[],[]],dtype=int)
    for mn_str in args.mn:
        mn = np.fromstring(mn_str,sep=',',dtype=int)
        m,n = mn
        assert m>0
        assert n>=0
        assert m>=n
        mn_array = np.c_[mn_array,mn]
    mn_array = mn_array.T
    print(f'{mn_array=}')

    #get icosahedron to start 
    V = get_icosahedron()

    #get octahedron to start 
    #V = get_octahedron()

    assert np.allclose(np.sqrt(np.sum(V**2,axis=-1)),1.)
    K = convex_hull_out(V)

    #recursive point-set generation
    for mn in mn_array:
        m,n = mn
        V = get_sac_point_set(V,K,m,n)
        K = convex_hull_out(V)

    edge_len = get_edge_len(V,K)
    tri_quality = get_triangle_quality(edge_len)
    cover_rad,sv = get_covering_radii(V)
    mesh_ratio = get_mesh_ratio(cover_rad,edge_len)
    N = V.shape[0]
    print(f'{N=}')
    print(f'mesh ratio = {mesh_ratio:.5f}')
    #areas = sv.calculate_areas()
    #print(f'min/max voronoi area = {np.min(areas)/np.max(areas):.5f}')

    if args.plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        hh = ax.plot_trisurf(*V.T[:2],K,V.T[2])
        hh.set_edgecolor((0,0,0))
        qmin = np.min(tri_quality)
        qmax = np.max(tri_quality)
        #manual cmap (tmin->tmax, white->black)
        hh.set_facecolors(1-np.tile(npa([1,1,1]),(K.shape[0],1))*((tri_quality-qmin)/(qmax-qmin))[:,None])

        hh.set_linewidth(4/np.sqrt(N))
        ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_zlim(-1,1);
        ax.set_box_aspect((4,4,4))
        plt.title(f'{N=}, {mesh_ratio=:.3f}, {qmin=:.3f}, {qmax=:.3f}')
        plt.show()

if __name__ == '__main__':
    main()
