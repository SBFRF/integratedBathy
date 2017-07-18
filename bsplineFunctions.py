import numpy as np
# this is a Python conversion of Meg Palmsten's (NRL) splining functions used in her interpD3DFRFBackground.m script
# DLY 7/18/2017


def bspline_pertgrid(Zi, w, splinebctype=0, lc=4, dxm=2, dxi=1):
    """
    :param Zi: the gridded perturbation data (Nx1 or NxM) 
    :param w: weights for Zi (same shape)
    :param splinebctype: 
            2 - second derivative goes to zero at boundary
            1 - first derivative goes to zero at boundary
            0 - value is zero at boundary
            10 - force value and derivative (first?!?) to zero at boundary
    :param lc: smoothing constraint value (usually 4)
    :param dxm: coarsening of the grid (e.g., 2 means calculate with a dx that is 2x input dx)
    :param dxi: fining of the grid (e.g., 0.1 means return spline on a grid that is 10x input dx)
    
    :return: Zi - splined version of Zi 
    """

    sz = np.shape(Zi)

    assert len(sz) <= 3, 'bspline_pertgrid Error: Zi must be array with three or fewer dimensions!'
    assert np.shape(Zi) == np.shape(w), 'bspline_pertgrid Error: dimensions of w must match Zi!'
    spline_types = [2, 1, 0, 10]
    assert splinebctype in spline_types, 'bspline_pertgrid Error: Unacceptable splinebctype.'

    Ny = sz[0]
    Nx = sz[1]

    # allow 1D input
    if ((Ny > 1) & (Nx == 1)):
        Zi = np.transpose(Zi)
        w = np.transpose(w)
        sz = np.shape(Zi)
        Ny = sz[0]
        Nx = sz[1]
    else:
        pass

    # get rid of nan weights
    id = np.where(np.isnan(w))
    w[id] = 0


    # specify bc condition
    fix_ends = [0]
    if ((Ny > 1) & (Nx >1) & (type(splinebctype) == int)):
        splinebctype = [splinebctype, splinebctype]
        fix_ends = [0, 0]
    else:
        splinebctype = [splinebctype]


    # do I need to fix dxm???!?!?!?!
    if ((Ny > 1) & (Nx >1) & (type(dxm) == int)):
        dxm = [dxm, dxm]
    elif ((Ny > 1) & (Nx >1) & (type(dxm) == float)):
        dxm = [dxm, dxm]
    elif isinstance(dxm, list):
        pass
    else:
        dxm = [dxm]

    # do I need to fix dxi???!?!?!?!
    if ((Ny > 1) & (Nx >1) & (type(dxi) == int)):
        dxi = [dxi, dxi]
    elif ((Ny > 1) & (Nx > 1) & (type(dxm) == float)):
        dxi = [dxi, dxi]
    elif isinstance(dxi, list):
        pass
    else:
        dxi = [dxi]


    # check to see if we are "pinning" bc
    for ii in range(0, len(fix_ends)):
        if splinebctype[ii] == 10:
            splinebctype[ii] = 1
            fix_ends[ii] = 1
        else:
            pass


    # input grid
    dx = 1
    x = np.arange(1, Nx+dx, dx)

    # output grid
    xi = np.arange(1, Nx+dxi[0], dxi[0])
    Nxi = len(xi)

    # put xm on boundary
    nxm = np.fix((x[-1] - x[0])/float(dxm[0]))
    dxm[0] = (x[-1] - x[0])/float(nxm)
    xm = x[0]+dxm[0]*np.arange(0, nxm+dx, dx)
    xm = np.transpose(xm)


    # can proceed in 1D or 2D
    if ((Ny > 1) & (Nx > 1)):
        y = np.arange(1, Ny + dx, dx)

        # repeat the above stuff for y
        if len(dxm) > 1:
            test = dxm
            del dxm
            dym = [test[1]]
            dxm = [test[0]]
        else:
            dym = dxm

        nym = np.fix((y[-1] - y[0]) / float(dym[0]))
        dym[0] = (y[-1] - y[0]) / float(nym)
        ym = y[0] + dym[0] * np.arange(0, nym + dx, dx)
        ym = np.transpose(ym)

        if len(dxi) > 1:
            test = dxi
            del dxi
            dyi = test[1]
            dxi = test[0]
        else:
            dyi = dxi

        yi = np.transpose(np.arange(1, Ny + dyi, dyi))
        Nyi = len(yi)

        Am = np.zeros((len(ym), Nx))
        Cm = np.zeros((len(ym), Nx))

        # spline alongshore first
        ztmp0 = np.zeros(Ny)
        fac = np.finfo(float).eps
        for ii in range(0, Nx):
            id = np.where(((~np.isnan(Zi[:, ii])) & (w[:, ii] > fac)))
            ztmp = ztmp0
            ztmp[id] = Zi[id, ii]
            temp = bspline_compute(y, ztmp, w[:, ii], ym, dym, lc, splinebctype[1])
            am = temp['am']
            aci = temp['aci']
            J = temp['J']
            Am[:, ii] = am
            aci[aci == J] == np.nan
            Cm[:, ii] = np.divide(aci, max(aci))

        if fix_ends[1]:
            Am[0, :] = -0.5*Am[1, :]
            Am[-1, :] = -0.5*Am[-2, :]
            Cm[0, :] = 0
            Cm[-1, :] = 0
        else:
            pass

        Zi = Am  #spline the Ams
        # update weights
        minw = np.min(w)
        w = minw + 1 - np.divide(Cm, 1 + Cm)
        w[np.isnan(w)] = 1/float(len(w))
        yprime = y
        y = ym
        Ny = len(ym)


    # now run the spline along the x-direction
    Zprime = np.zeros((Ny, Nxi))
    ztmp0 = np.zeros(np.shape(Zi[0, :]))
    fac = np.finfo(float).eps

    for ii in range(0, Ny):
        # run spline
        id = np.where(((~np.isnan(Zi[ii, :])) & (w[ii, :] > fac)))
        ztmp = ztmp0
        ztmp[id] = Zi[ii, id]
        temp = bspline_compute(np.transpose(x), np.transpose(ztmp), np.transpose(w[ii, :]), xm, dxm, lc, splinebctype[0])
        # check to see if we are also forcing boundary
        am = temp['am']
        if fix_ends[0]:
            am[0] = -0.5*am[1]
            am[-1] = -.05*am[-2]
        else:
            pass

        # get result on fine grid
        temp = bspline_curve(np.transpose(xi), xm, am, dxm, splinebctype[0])
        zm = temp['z']
        Zprime[ii, :] = zm

    Zi = Zprime

    # now convert back, Zi are really Am^y
    if ((Ny > 1) & (Nx > 1)):
        y = yprime
        Ny = len(y)
        Ziprime = Zi
        # check of we are also forcing zero at boundary
        if len(fix_ends) == 1:
            test_c = (fix_ends[0] == 1)
        elif len(fix_ends) == 2:
            test_c = ((fix_ends[0] == 1) & (fix_ends[1] == 1))
        else:
            pass
        if test_c:
            Ziprime[0, :] = -0.5*Ziprime[1, :]
            Ziprime[-1, :] = -.05*Ziprime[-2, :]

        Zi = np.zeros((Nyi, Nxi))
        for ii in range(0, Nxi):
            temp = bspline_curve(yi, ym, Ziprime[:, ii], dym, splinebctype[1])
            zm = temp['z']
            Zi[:, ii] = zm


    return Zi


def bspline_compute(x, z, w, xm, dxm, lc, bctype):
    # this fits 1D data to a spline, like a boss....
    """
    
    :param x: the N x 1 data locations
    :param z: the N x 1 observations
    :param w: the N x 1 observations weights (i.e., rms(true)/ (rms(error) + rms(true)))
    :param xm: the M x 1 EQUALLY SPACED!!!! grid nodes
                xm[1] should be smaller than the smallest x and xm[-1] should be larger than the largest x
                modify the xm or the input x to satisfy such that incosistant use of bc does not flare up
    :param dxm: the 1 x 1 grid spacing of xm, used to scale
    :param lc: the 1 x 1 spline curvature penalty weight
                lc = 4 wipes out wavelengths less than 2*dxm (Nyquist)
    :param bctype: this may either be
                2 - second derivative vanishes at boundary
                1 - first derivative vanishes at boundary
                0 - value is specified at the boundary
    :return: dictionaty containing
            am - the M x 1 spline amplitides
            aci - the M x 1 error estimates
            J - the 1 x 1 rms error of the spline surface
            zm - the M x 1 spline values at the location xm
            zs - the N x 1 spline values at the locations x
            Note: the spline values can be computed anywhere in the domain using
            zs = bspline_curve(xi, xm, am, dxm)
    """

    # input grid N
    N = len(x)

    #output grid N
    M = len(xm)

    # normalize inputs
    x = x/float(dxm[0])
    xm = xm/float(dxm[0])
    z = np.multiply(w, z)

    # bc coef
    bc = np.zeros(M)
    if bctype == 2:
        bc[0:2] = [2, -1]
        bc[-2:] = [-1, 2]
    elif bctype == 1:
        bc[0:2] = [0, 1]
        bc[-2:] = [1, 0]
    elif bctype == 0:
        bc[0:2] = [-4, 1]
        bc[-2:] = [-1, -4]
    else:
        pass

    # initial boundary (-1)
    fb1 = bspline_basis(x-(xm[0]-1))
    # end boundary (M)
    fbM = bspline_basis(x - (xm[-1] + 1))

    #compute matrix coefficients
    b = np.zeros(M) # data correlation
    p = np.zeros((M, M)) # model-model correlation at data
    g1 = 0
    g2 = 0
    g3 = 0

    for m in range(0, M):
        # forward sweep
        # set boundary condition
        if (m < 2):
            # initial boundary
            fb = fb1
        elif (m >= (M-4)):
            fb = fbM
        else:
            fb = 0

        # superimpose boundary and central elements, or just central
        if (m > 0):
            # if we are past the boundary get the lagged terms we have computed already
            f = g1
        else:
            # generate function
            f = bspline_basis((x - xm[m]) + bc[m]*fb)

        b[m] = np.dot(np.transpose(f), z) # spline-data covariance
        p[m, m] = np.dot(np.transpose(f), np.multiply(w, f)) # spline - spline covariance, diagonal term

        # do first off-diagonal terms
        if (m < M-1):
            mm = m + 1
            if ((m > 0) & (m < (M-3))):
                g1 = g2
            else:
                # include boundary influence here
                g1 = bspline_basis((x - xm[mm]) + bc[mm]*fb)
            p[m, mm] = np.dot(np.transpose(f), np.multiply(w, g1)) # spline - spline covariance, diagonal term
            p[mm, m] = p[m, mm]

        # do second off diagonal terms
        if (m < (M-2)):
            mm = mm + 1
            if ((m > 0) & (m < (M-3))):
                g2 = g3
            else:
                g2 = bspline_basis((x - xm[mm]) + bc[mm]*fb)
            p[m, mm] = np.dot(np.transpose(f), np.multiply(w, g2))  # spline - spline covariance, diagonal term
            p[mm, m] = p[m, mm]

        # do third off diagonal terms
        if (m < (M-3)):
            mm = mm + 1
            g3 = bspline_basis((x - xm[mm]) + bc[mm]*fb)
            p[m, mm] = np.dot(np.transpose(f), np.multiply(w, g3))  # spline - spline covariance, diagonal term
            p[mm, m] = p[m, mm]
        else:
            pass

    # q is the second derivative of the covariance matrix
    # it does not depend on the data
    q = (-27/float(8))*np.diag(np.ones(M-1), 1) + (-27/float(8))*np.diag(np.ones(M-1), -1) + (6)*np.diag(np.ones(M)) + (3/float(8))*np.diag(np.ones(M-3), 3) + (3/float(8))*np.diag(np.ones(M-3), -3)

    # implement the appropriate boundary conditions
    q[0, 0] = (3/float(4))*np.square(bc[0]) - (9/float(4))*bc[0] + 3
    q[-1, -1] = q[0, 0]  # take advantage of symmetry???

    q[0, 1] = (-9/float(4)) + (3/float(4))*bc[0]*bc[1] - (-9/float(8))*bc[1]
    q[1, 0] = q[0, 1]
    q[-1, -2] = q[0, 1]
    q[-2, -1] = q[0, 1]

    q[0, 2] = (3/float(8))*bc[0]
    q[2, 0] = q[0, 2]
    q[-1, -3] = q[0, 2]
    q[-3, -1] = q[0, 2]

    q[1, 1] = (3/float(4))*np.square(bc[1]) + (21/float(4))
    q[-2, -2] = q[1, 1]

    q[1, 2] = (-27/float(8)) + (3/float(8))*bc[1]
    q[2, 1] = q[1, 2]
    q[-2, -3] = q[1, 2]
    q[-3, -2] = q[1, 2]

    # compute the curvature penalty from lc
    alpha = np.power((lc/float(2*np.pi)), 4)

    # this normalization allows q to kick in when sumw is small, else q is not needed
    if len(w) == 1:
        sumw = N*w + 1
    else:
        sumw = sum(w) + 1

    # add curvature terms
    # this form enforces constant scaling of freq cutoff at about 0.25/lc
    r = np.divide(p, sumw) + alpha*(q/float(M))
    b = np.divide(b, sumw)

    # check matrix conditioning
    # if no good we can try for a direct solution, but no error estimate
    mrc = 1/float(np.linalg.cond(r))
    fac = np.finfo(float).eps
    if mrc > 100*fac:
        # this might be why her solution is super sensitive to perturbations?
        # her threshold for going ahead with the inversion is way way low.
        # According to the matlab documetation, this number should be close to one!
        r = np.linalg.inv(r)
        am = np.dot(r, b)
    else:
        am, resid, rank, s = np.linalg.lstsq(r, b)
        r = np.ones(np.shape(r))

    msz = np.divide(np.dot(np.transpose(z), z), sumw)
    J = msz - np.dot(np.transpose(am), np.dot(np.divide(p, sumw) + np.dot(alpha, np.divide(q, M)), am))
    aci = np.sqrt(np.dot(np.diag(r), np.divide(J, sumw)))

    zm_dict = bspline_curve(xm, xm, am, 1, bctype, aci)
    zs_dict = bspline_curve(x, xm, am, 1, bctype, aci)

    out = {}
    out['am'] = am
    out['aci'] = aci
    out['J'] = J
    out['zm'] = zm_dict['z']
    out['zs'] = zs_dict['z']

    return out


def bspline_basis(y):

    """
    :param y: the N x 1 independent variable locations
                y is normalized by dy (grid node spacing) and centered on xm (node of interest)
    :return: y - the basis function for x - xm = 0
    """

    ya = abs(y)

    if (isinstance(y, int)) or (isinstance(y, float)):
        y = 0
        if (1 <= ya) and (ya < 2):
            y = 0.25 * np.power((2 - ya), 3)
        elif ya < 1:
            y = 0.25 * np.power((2 - ya), 3) - np.power((1 - ya), 3)

    else:
        y = np.zeros(np.shape(y))

        # and defined piecewise
        id12 = np.where(((1 <= ya) & (ya < 2)))
        id1 = np.where(ya < 1)
        y[id12] = 0.25 * np.power((2 - ya[id12]), 3)
        y[id1] = 0.25 * np.power((2 - ya[id1]), 3) - np.power((1 - ya[id1]), 3)

    return y


def bspline_curve(y, ym, am, dx, bctype, aci=None):

    """
    
    :param y: the N x 1 locations of interest
    :param ym:  the M x 1 EQUALLY SPACED!!! locations of the basis function
    :param am:  the corresponding basis function amplitudes
    :param dx:  the spacing of grid xm
    :param bctype:  this may either be
                2 - second derivative vanishes at boundary
                1 - first derivative vanishes at boundary
                0 - value is specified at the boundary
    :param aci: something to do with the errors, if you aren't interested in the errors this doesn't matter
    
    :return: dictionary containing
            z - the N x 1 spline values at each yi
            e - the N x 1 error values at each yi
    """

    if aci is None:
        aci = np.zeros(np.shape(y))
    else:
        pass

    M = len(ym)

    # bc coef
    bc = np.zeros(M)
    if bctype == 2:
        bc[0:2] = [2, -1]
        bc[-2:] = [-1, 2]
    elif bctype == 1:
        bc[0:2] = [0, 1]
        bc[-2:] = [1, 0]
    elif bctype == 0:
        bc[0:2] = [-4, 1]
        bc[-2:] = [-1, -4]
    else:
        pass

    # normalize length scale by dx
    y = np.divide(y, dx)
    ym = np.divide(ym, dx)
    z = np.zeros(np.shape(y))
    e = np.zeros(np.shape(y))

    # compute the basis function a each output location and multiply by amplitude
    for ii in range(0, len(y)):

        # use only those functions that support location y[ii]
        # saves computing a lot of zeros
        id = np.where(abs(y[ii] - ym) <= 2)
        if type(id) == tuple:
            id = id[0]
        else:
            pass

        if np.size(id) > 0:
            f = bspline_basis(y[ii] - ym[id])
            z[ii] = np.dot(am[id], np.transpose(f))

            e[ii] = np.dot(np.square(aci[id]), np.transpose(np.square(f)))

            if np.size(id) > 1:
                tcon = (id < 2).any()
            else:
                tcon = id < 2

            if tcon:
                # need to include bc[0]
                abc = am[0]*bc[0] + am[1]*bc[1]
                f1 = bspline_basis(y[ii] - ym[0] + 1)
                z[ii] = z[ii] + abc*f1
                e[ii] = e[ii] + np.square(aci[0]*bc[0]) + np.square(f1)*np.square(aci[1]*bc[1])
            else:
                pass

            if np.size(id) > 1:
                tcon = (id > (M-3)).any()
            else:
                tcon = id > (M-3)

            if tcon:
                # need to include bc[-1]
                abc = am[-1] * bc[-1] + am[-2] * bc[-2]
                f2 = bspline_basis(y[ii] - ym[-1] + 1)
                z[ii] = z[ii] + abc*f2
                e[ii] = e[ii] + np.square(aci[-2] * bc[-2]) + np.square(f1)*np.square(aci[-1] * bc[-1])
            else:
                pass

        else:
            y[ii] = 0

    out = {}
    out['z'] = z
    out['e'] = e

    return out



