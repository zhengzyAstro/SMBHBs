from numpy import empty,pi,sqrt,sin,cos,var,dot,where,identity,zeros,exp,log,median,dot,log10,abs,asarray,zeros_like
from scipy.linalg import solveh_banded,cholesky
from scipy import transpose
import weave
from qso_fit_fix import qso_engine
from scipy.optimize import fmin
from sklearn.utils import check_random_state


def lombred_bootstrap(time, signal, error, f1, df, numf,ltau=2.,lvar=-2.6,do_fit=True,N_bootstraps=100,random_state=None):
    """Use a bootstrap analysis to compute Lomb-Scargle significance

    Parameters
    ----------
    """
    random_state = check_random_state(random_state)
    time = asarray(time)
    signal = asarray(signal)
    error = asarray(error) + zeros_like(signal)
    
    D = zeros(N_bootstraps)

    for i in range(N_bootstraps):
        ind = random_state.randint(0, len(signal), len(signal))
        psd,lvar,ltau,vcn = lomb(time, signal[ind], error[ind],f1,df,numf,ltau,lvar,do_fit=False)
        D[i] = psd.max()

    return D


def lomb(time, signal, error, f1, df, numf,ltau=2.,lvar=-2.6,do_fit=True):
    """
    C version of lomb_scargle

    Inputs:
        time: time vector
        signal: data vector
        error: data uncertainty vector
        df: frequency step
        numf: number of frequencies to consider

        ltau,lvar: DRW model parameters, initial guesses if do_fit=True

    Output:
        psd: power spectrum on frequency grid: f1,f1+df,...,f1+numf*df
    """
    numt = len(time)
    dt = abs(time[1:]-time[:-1]);
    dtm=log10(dt.min())
    maxt = log10(time.max()-time.min())

    wth = (1./error).astype('float64')
    s0 = dot(wth,wth)
    wth /= sqrt(s0)

    cn = (signal*wth).astype('float64')
    cn -= dot(cn,wth)*wth

    if (do_fit):
        def fit_fun(par):
            par[0] = par[0].clip(-6.,2)
            par[1] = par[1].clip(dtm-1,maxt+1)
            result = qso_engine(time, signal, error, lvar=par[0], ltau=par[1])
            chi = (result['chi2_qso/nu']+result['chi2_qso/nu_extra'])*result['nu']
            return chi
        rs = fmin(fit_fun,[lvar,ltau],disp=0)
        lvar,ltau = rs[0],rs[1]


    #print ("Noise parameters: lvar=%.3f ltau=%.3f") % (lvar,ltau)
    # sparse matrix form: ab[u + i - j, j] == a[i,j]   i<=j, (here u=1)
    T = zeros((2,numt),dtype='float64')
    arg = dt*exp(-log(10)*ltau); ri = exp(-arg); ei = 1./(1./ri-ri)
    T[0,1:] = -ei; T[1,:-1] = 1.+ri*ei; T[1,1:] += ri*ei; T[1,numt-1] += 1.
    T0 = median(T[1,:]); T /= T0

    lvar0 = log10(0.5)+lvar+ltau
    fac = exp(log(10)*lvar0)*s0/T0
    Tp = 1.*T; Tp[1,:] += wth*wth*fac

    Tpi = solveh_banded(Tp,identity(numt))

    #
    # CI[i,j] = T[1+i-k,k] Tpi[k,j]   (k>=i), k=i is diagonal
    # CI[i,j] = T[1,i] * Tpi[i,j] + T[0,i+1]*Tpi[i+1,j] + T[0,i]*Tpi[i-1,j]
    CI = empty((numt,numt),dtype='float64')
    CI[0,:] = T[1,0]*Tpi[0,:] + T[0,1]*Tpi[1,:]
    CI[numt-1,:] = T[1,numt-1]*Tpi[numt-1,:] + T[0,numt-1]*Tpi[numt-2,:]
    for i in xrange(numt-2):
        CI[i+1,:] = T[1,i+1]*Tpi[i+1,:] + T[0,i+2]*Tpi[i+2,:] + T[0,i+1]*Tpi[i,:]


    # cholesky factorization m0[i,j] (j>=i elements non-zero) dot(m0.T,m0) = CI
    CI = dot( 1./wth*identity(numt),dot(CI,wth*identity(numt)) )
    m0 = cholesky(CI,lower=False)

    #v = dot(dot(m0.T,m0),wth*wth*identity(numt))
    #print (v[:,20]/v[20,20])
    wth1 = dot(m0,wth)
    s0 = dot(wth1,wth1)
    wth1 /= sqrt(s0);
    cn = dot(m0,cn)
    cn -= dot(cn,wth1)*wth1

    tt = 2*pi*time.astype('float64')
    sinx,cosx = sin(tt*f1)*wth,cos(tt*f1)*wth
    wpi = sin(df*tt); wpr=sin(0.5*df*tt); wpr = -2.*wpr*wpr

    psd = empty(numf,dtype='float64')
    vcn = var(cn,ddof=1)

    lomb_scargle_support = """
      inline void update_sincos (int numt, double wpi[], double wpr[], double sinx[], double cosx[]) {
          double tmp;
          for (int i=0;i<numt;i++) {
              sinx[i] = (wpr[i]*(tmp=sinx[i]) + wpi[i]*cosx[i]) + sinx[i];
              cosx[i] = (wpr[i]*cosx[i] - wpi[i]*tmp) + cosx[i];
          }
      }
      inline double lomb_scargle(int numt, double cn[], double sinx[], double cosx[], double st, double ct, double cst) {
          double cs=0.,s2=0.,c2=0.,sh=0.,ch=0.,px=0.,detm;
          for (int i=0;i<numt;i++) {
              cs += cosx[i]*sinx[i];
              s2 += sinx[i]*sinx[i];
              c2 += cosx[i]*cosx[i];
              sh += sinx[i]*cn[i];
              ch += cosx[i]*cn[i];
          }
          cs -= cst; s2 -= st; c2 -= ct;
          detm = c2*s2 - cs*cs;
          if (detm>0) px = ( c2*sh*sh - 2.*cs*ch*sh + s2*ch*ch ) / detm;
          return px;
      }
      inline void calc_dotprod(int numt, double sinx[], double cosx[], double wt[], double *st, double *ct, double *cst) {
          double a=0,b=0;
          for (int i=0;i<numt;i++) {
              a += sinx[i]*wt[i];
              b += cosx[i]*wt[i];
          }
          *st = a*a; *ct = b*b; *cst =a*b;
      }
      inline void dered_sincos(int numt, double sinx[], double cosx[], double sinx1[], double cosx1[], double m0[]) {
          int i,k;
          unsigned long inumt;
          double tmpa,tmpb,tmpc,tmpc0,s1,s2;
          for (i=0;i<numt;i++) {
              tmpc0 = m0[i+i*numt];
              s1 = tmpc0*(tmpa=sinx[i]);
              s2 = tmpc0*(tmpb=cosx[i]);
              inumt = i*numt;
              for (k=i+1;k<numt;k++) {
                  tmpc=m0[k+inumt];
                  if (fabs(tmpc)<tmpc0*1.e-3) break;
                  s1 += tmpc*tmpa;
                  s2 += tmpc*tmpb;
              }
              sinx1[i] = s1; cosx1[i] = s2;
          }
      }
    """

    lomb_code = """
      double sinx1[numt],cosx1[numt],ct,st,cst;
      for (unsigned long j=0;j<numf;j++) {
          dered_sincos(numt,sinx,cosx,sinx1,cosx1,m0);
          calc_dotprod(numt,sinx1,cosx1,wth1,&st,&ct,&cst);
          psd[j] = lomb_scargle(numt,cn,sinx1,cosx1,st,ct,cst);
          update_sincos (numt, wpi, wpr, sinx, cosx);
      }
    """

    weave.inline(lomb_code, ['cn','numt','numf','psd','wpi','wpr','sinx','cosx','m0','wth1'],\
      support_code = lomb_scargle_support,force=0)


    return 0.5*psd/vcn,lvar,ltau,vcn
