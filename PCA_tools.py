import numpy as np
import tools as common_tools
import EUgrid as Grid
from europe_plusgrid import europe_plus_Nodes


def load_flows(capacities, solvermode, filename=None, path='./results/LinkCapSweeps/'):
    if not filename:
        filename = ''.join(['Europe_aHE_', capacities, '_', solvermode, '_flows.npy'])

    f = np.load(path+filename)
    return f


def center(q):
    """ This function subtracts the mean of each row from itself
        so all the rows have zero mean. Returns the centered F,
        along with the column vector of the old mean.

        """
    mean = np.mean(q, axis=1)
    q_c = q - np.transpose(np.tile(mean, (q.shape[1], 1)))

    return q_c, mean


def normalize(q_c):
    """ Normalizes the centered time-dependent vector q_c,
        averaged over all time-steps it has unit length.
        Notation from Haken 1996: eq. 11.5 pp. 159:
        h(t) = (q(t) - q_mean)N. q_c = q(t) - q_mean.

        """

    ntilde = np.sqrt(q_c.shape[1]/np.sum(q_c*q_c))
    h = ntilde*q_c

    return h, ntilde


def FtoPhi(F, K=None, gamma=1.0, alpha=0.8):
    """ This function takes a L x T array (time dependent flow vector)
        and converts it into an N x T array, of the corresponding
        injection pattern.

        """

    if K==None:
        N = Grid.EU_Nodes_Gamma_Alpha_input(None, g=gamma, a=alpha, full_load=False)
        K = common_tools.AtoKh_old(N)[0]

    length_of_timeseries = F.shape[1]
    Phi = np.empty((K.shape[0], length_of_timeseries))

    for t in xrange(length_of_timeseries):
        Phi[:,t] = np.dot(K, F[:,t])

    return Phi, K


def PhitoF(Phi, K=None):
    """ This function assumes that Phi is a single N vector,
        and is intended to convert a principal component of
        and injection pattern into a principal flow component.

        """

    assert(sum(Phi)<= 1e-8), 'Phi is not a consistent flow pattern.'
    if K!=None:
        N = europe_plus_Nodes()
        K = common_tools.AtoKh_old(N)[0]
    L = np.dot(K, K.T)
    Lplus = np.linalg.pinv(L)
    KTLplus = np.dot(K.T, Lplus)
    F = np.dot(KTLplus, Phi)

    return F


def get_principal_component(h, comp_number):
    """ This function takes a centered, normalized timedependent vector
        h (notation as in Haken 1996), and returns the principal
        comp_number'th principal component, and it's corresponding
        eigenvalue.

        Important: comp_number begins from 0, 0 being the first principal
        component!.

        """
    if comp_number==-1:
        print "Warning: now accesing the least significant component!"

    R = np.cov(h, ddof=0) # eq. 11.16
    assert(np.trace(R) - 1 <= 1e-5)
    eigvals, eigvecs = np.linalg.eigh(R) # solving eq. 11.19
    sorted_eigvecs = eigvecs[:,np.argsort(eigvals)]
    sorted_eigvals = np.sort(eigvals)
    lambd = sorted_eigvals[-comp_number-1]
    princ_comp = sorted_eigvecs[:,-comp_number-1]
    assert(np.sum(princ_comp**2) - 1 <= 1e-8), "PC not unit length"

    return lambd, princ_comp


def get_xi_weight(h, comp_number):
    """ This function returns the time dependent weight of the
        comp_number'th principal component in h. See eq. 11.27.

        Important: comp_number begins from 0, 0 being the first principal
        component.

        """

    lamb, princ_comp = get_principal_component(h, comp_number)
    assert(np.sum(princ_comp**2) - 1 <= 1e-8)
    length_of_timeseries = h.shape[1]
    xi = np.empty(length_of_timeseries)
    for t in xrange(length_of_timeseries):
        xi[t] = np.dot(princ_comp, h[:,t])

    return xi


def unnormalize_uncenter(h, Ntilde, mean):
    """ This function takes a normalized, centered vector and returns
        the vector in the original coordinate system. E.g. phi in MW,
        with 0 being neither source or sink.

        """

    return h/Ntilde + mean







