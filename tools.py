#! /usr/bin/env python
#from pylab import *
#from scipy import *
import pickle
from pylab import plt
import numpy as np
from scipy.stats.mstats import mquantiles as mquant
from scipy.optimize import fmin as scifmin
from scipy.optimize import brentq
from scipy.ndimage.filters import gaussian_filter1d as gauss
import multiprocessing as mp
#from plotting import *

def close(a,b):
    return ((a < (b*1.00001 + 1e-6)) and (a > (b* 0.99999 - 1e-6))) or (a==b)

def non_zeros(s):
    v=[]
    for w in s:
        if not close(w,0):
            v.append(w)
    return v

def get_positive(x):
    """ This function returns the positive side of variable. """
    return x*(x>0.)  #Possibly it has to be x>1e-10. HAS TO BE np.array

def get_negative(x):
    """ This function returns the positive side of variable. """
    return x*(x<0.)  #Possibly it has to be x>1e-10. HAS TO BE np.array

def get_q(s,q=0.99):
    """ Looks at a cumulative histogram of a time series.It
        returns the magnitude of s that happens a fraction 'quant' of the time.
    """
    return mquant(s,q)[0]

def link_q(f,q=0.99):
    a = (1-q)/2.0
    b = q+a
    return max(-mquant(f,a)[0],mquant(f,b)[0])

def get_quant_caps(quant=0.99,F = None, filename='results/copper_flows.npy'):
    """ Looks at a cumulative histogram of the flow between two nodes, in the
        direction defined by the .npy file. It returns the magnitude of the flow
        that happens a fraction 'quant' of the time.
    """
    if F == None:
        F = np.load(filename)

    T_caps = np.zeros(len(F)*2)
    for i in range(len(F)):
        T_caps[i*2]=link_q(F[i])#get_q(F[i],quant)
        T_caps[i*2 + 1] = link_q(F[i])#-get_q(F[i],1-quant)
    return T_caps

def bal(x):
    return -sum(x*(x<0))

def ISO2LONG(x):
    table= np.load('./settings/ISET2ISO_country_codes.npy')
    for i in np.arange(len(table)):
        if table[i]["ISO"]==x: return table[i]['name']

def optimal_mix_balancing(load,wnd,sol):
    nhours = len(load)
    L, GW, GS = np.array(load,ndmin=2), np.array(wnd,ndmin=2), np.array(sol,ndmin=2)

    weighted_sum = lambda x: np.sum(x,axis=0)/np.mean(np.sum(x,axis=0))

    l = weighted_sum(L)
    w = weighted_sum(GW)
    s = weighted_sum(GS)

    #balancing_ts = lambda alpha:
    sum_bal = lambda alpha: bal((w*alpha + (1.0-alpha)*s) - l)
    alpha_opt = scifmin(sum_bal, 0.75, disp = 0, xtol =0.001, ftol = 0.001)

    return alpha_opt[0],sum_bal(alpha_opt)/nhours


def biggestpair(H):
    H0=np.zeros((len(H))/2)
    for i in range(len(H0)):
        H0[i]=max(H[2*i],H[2*i+1])
    return H0


def AtoKh_old(N,h0=None,pathadmat=None):
    if pathadmat==None:
        pathadmat=N.pathadmat
    Ad=np.genfromtxt(pathadmat,dtype='d')
    L=0
    listFlows=[]
    for j in range(len(Ad)):
        for i in range(len(Ad)):
            if i>j:
                if Ad[i,j] > 0:
                    L+=1
    K=np.zeros((len(Ad),L))
    h=np.zeros(L*2)
    h=np.append(h,np.zeros(3*len(Ad)))
    L=0
    j=0
    i=0
    for j in range(len(Ad)):
        for i in range(len(Ad)):
            if i>j:
                if Ad[i,j] > 0:
                    K[j,L]=1
                    K[i,L]=-1
                    h[2*L]=Ad[i,j]
                    h[2*L+1]=Ad[j,i]
                    listFlows.append([ str(N[j].label) , str(N[i].label) , L ])
                    L+=1
    if h0 == None:
        h0=h

    return K,h0,listFlows

def get_high_low_fft(load, tau=24):
    """Error if len(load) not even"""
    iseven = 2*(len(load)/2) == len(load)

    if not iseven:
        print "Warning! Uneven length array. Patching the end. (dfkl32k)"
        load = np.concatenate([load,[load[-1]]])

    time = np.arange(len(load));
    freq = fftfreq(len(time))[:len(time)/2+1]
    load_fft = rfft(load);
    sigma = tau*2;
    kernel = exp(-2*(pi*freq/(2*pi)*sigma)**2);
    load_fft = rfft(load);
    load_high_gauss = irfft(load_fft*(1-kernel)).real;
    load_low_gauss = load - load_high_gauss;

    if iseven:
        return load_low_gauss, load_high_gauss
    else:
        return load_low_gauss[:-1], load_high_gauss[:-1]

class Results:

    def __init__(self, filename="default", path='./results/'):
    	self.cache=[]
        try: self.load_results(filename,path)
        except: self.save_results(filename, path)

    def __getitem__(self,x):
        return self.cache[x]

    def __len__(self):
        return len(self.cache)

    def __call__(self):
        return self

    def add_instance(self,N,F):
        inst={}
        inst["alpha"]=np.round(N[0].alpha,3)
        inst["gamma"]=np.round(N[0].gamma,3)
        inst["E_B"]=sum(sum(n.balancing for n in N))
        inst["C_B"]=sum([np.max(n.balancing) for n in N])
        inst["C_T"]=sum([np.max([ max(f)  , -min(f) ]) for f in F])
        inst["Q_T"]=sum(np.max([ -get_q( f, q=0.01), get_q( f, q=0.99 )]) for f in F )
        inst["Q_B"]=sum(-get_q( n.curtailment-n.balancing, q=0.01) for n in N )
        self.cache=np.append(self.cache,inst)

    def add_instance_manually(self,inst):
        self.cache=append(self.cache,inst)

    def save_results(self,filename,path='./results/'):
        with open(path+filename,'w') as fil:
            pickle.dump(self.cache,fil)

    def load_results(self,filename,path='./results/'):
        with open(path+filename) as fil:
            self.cache=append(self.cache,pickle.load(fil))

    def look_up(self,a,g,create=False,mode="linear"):
        found = 0
        for i in self.cache:
            if np.round(i["alpha"],3)==np.round(a,3) and np.round(i["gamma"],3)==np.round(g,3):
                found = 1
                return i

        if create and not found:
            N=new_EU_Nodes()
            N.set_gammas(g)
            N.set_alphas(a)
            #N,F=solve(N,copper=1,squaremin=squaremin, c=c, nocurt=nocurt, smart=smart)
            N,F = solve(N,copper=1,verbose=0,mode=mode,msg="a="+str(a)+" g="+str(g)+", "+mode)
            self.add_instance(N,F)
            del N,F
            return self.cache[-1]#self.cache[-1]
        #return "404 -- page not found"  # :D

    def spit_them_out(self,alpha):
        R=self.cache
        for r in R:
            if alpha==np.round(r["alpha"],2) and np.round(r["gamma"],2)==0.0:
                EU_E=r["E_B"]
        g=[]
        E_B=[]
        C_B=[]
        Q_B=[]
        C_T=[]
        Q_T=[]
        for r in R:
            if alpha==np.round(r["alpha"],2):
                g.append(np.round(r["gamma"],3))
                if r["gamma"] > 1: E_B.append(r["E_B"]/8000000.0)  #TWh per year#
                else: E_B.append((r["E_B"]-(1-g[-1])*(EU_E))/8000000.0)
                C_B.append(r["C_B"]/1000.0)  #In GW
                Q_B.append(r["Q_B"]/1000.0)
                C_T.append(r["C_T"]/1000.0)  # In GW
                Q_T.append(r["Q_T"]/1000.0)
        return g, E_B, C_B, Q_B, C_T, Q_T

def linfo(pathadmat):
    path_to_lineinfo = pathadmat[0:-9] + '_lineinfo.txt'
    link_info = np.loadtxt(path_to_lineinfo, dtype='string', delimiter='\t')
    if type(link_info[0])==np.string_:
        link_info = np.array([link_info])
    return link_info

################# Cost utils
########### Annualization factors
def ann(r):
    if r==0: return 30
    return (1-(1+(r/100.0))**-30)/(r/100.0)


################### The five essential terms
def cbe(b_e,r=4.0): ### Cost of balancing energy (per annual)
    return b_e*56*ann(r)
    #return 49.0*ann(r)*1e-12
    #def cbc(b_c,r=4.0):
    ### with no CSS (900000 + 4500*ann(r)) bnv combined cycle
    ### emissions 117 lb/mmbtu --> 0.1996 tn/MWh
    ### with CSS   (2700000 +13200*ann(r)) bnv combined cycle with CCS
    ### emissions 18 lb/mmbtu --> 0.0307 tn/MWh
    ### old: return 1.4*b_c*(1370000 + 14000*ann(r))*1e-12/Energy()#return 1.0*(640000 + 120000*ann(r))*1e-12
def cbc(b_c,r=4.0): ### Cost of balancing capacity Using CCGT
    return b_c*(900000 + 4500*ann(r))
def cwc(w_c,r=4.0): ### Cost of wind capacity
    return w_c*(1500000 + 35000*ann(r))#return 1.0*(1150000 + 35000*ann(r))*1e-12
def csc(s_c,r=4.0): ### Cost of solar capacity
    return s_c*(1500000 + 8500*ann(r))#return 1.0*(800000 +  25000*ann(r))*1e-12
def ctc(t_c, pathadmat, r=4.0): ### Cost of transmission capacity (has been updated, due to line lengths)
    CTC_F = [0, 150000]
    CTC_V = [400, 1500]
    LI = linfo(pathadmat)
    F = np.zeros(len(t_c))
    for l in range(len(t_c)):
        if LI[l][1] == 'AC':
            F[l] = t_c[l]*CTC_F[0] + t_c[l]*CTC_V[0]*float(LI[l][2])
        if LI[l][1] == 'DC':
            F[l] = t_c[l]*CTC_F[1] + t_c[l]*CTC_V[1]*float(LI[l][2])
    return (sum(F))

########################### Non-essential costs
def cee(e_e, r=4.0): ### Income from selling/storing excess energy (50 EUR/MWh)
    return 0.5*e_e*100*ann(r)

def cbcCCS(b_c,r=4.0): ### Cost of balancing capacity CCGT with CCS
    return b_c*(2700000 + 13200*ann(r))

def co2(b_e,p=20.0,r=4.0): ### Taxation of CO2 using CCGT (emissions times price 'p')
    return b_e*(0.443)*p*ann(r)

def co2CCS(b_e,p=20.0,r=4.0): ### Taxation of CO2 using CCGT WITH CCS
    return b_e*(0.065)*p*ann(r)

######################## Wrapper

COST = {'B_E':cbe, 'B_C':cbc, 'B_CCCS':cbcCCS, 'W_C':cwc, 'S_C':csc, 'T_C':ctc, 'CO_2':co2, 'CO_2CCS': co2CCS, 'E_E': cee}
### for LCOE, divide everything by Energy()
def Energy(r=4.0): ### Annual energy consumed (sum of L_n(t) for all n for all t)/(years)
    return 3027140662*ann(r)
### Usage example
### LCOE = (COST['B_E'](be) + COST['B_E'](bc) + COST['B_E'](wc) + COST['B_E'](sc) + COST['B_E'](tc))/Energy()


