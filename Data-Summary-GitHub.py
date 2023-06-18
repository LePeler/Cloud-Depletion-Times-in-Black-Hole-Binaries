import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as stats
import scipy.integrate as integ
import scipy.special as spec
import os
plt.rc('font', size=18)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
path="C:/Users/Prince LePéler/Desktop/Programs/Bachelor's Thesis/TEST"
Data=path+"/Results"
files=[f for f in os.listdir(Data) if os.path.isfile(os.path.join(Data, f))]

le=len(files)
results=np.array([[0]*11]*le,dtype=np.float64)
for n in range(le):
    results[n]=np.load(Data+"/"+files[n])

print(results)

G=6.6743e-11
c=299792459


#goodness-of-fit tests
def rms_test(sig):
    N=len(sig)
    rms=np.sqrt(np.sum(sig**2))
    def f(x):
        f=np.exp(-1/2*x**2)*x**(N-1)
        return(f)
    I=integ.quad(lambda x:f(x),rms,np.inf)[0]
    p=(1/2)**(N/2)*2/spec.gamma(N/2)*I
    return(p)

def sup_test(sig):
    N=len(sig)
    sup=np.max(np.abs(sig))
    def f(x):
        f=np.exp(-1/2*x**2)*spec.erf(x/np.sqrt(2))**(N-1)
        return(f)
    J=integ.quad(lambda x:f(x),sup,np.inf)[0]
    p=N*np.sqrt(2/np.pi)*J
    return(p)
#


def Tq(a,e,Mtot,st):
    res=results[(results[:,1]==a) & (results[:,2]==e) & (results[:,3]==Mtot) & (results[:,4]==st)]
    Q=res[:,0]
    TD=res[:,5]
    dTD=res[:,6]
    
    fig=plt.figure(figsize=(10,5.5),dpi=300)
    Ax=plt.axes()
    Ax.set_xscale("log")
    Ax.set_yscale("log")
    Ax.grid()
    Ax.set_xlabel(r"$q^{-1}$")
    Ax.set_ylabel(r"$T_\mathrm{Depletion}$ $\mathrm{in}$ $s$")
    Ax.set_title(rf"$a={a}$, $e={e}$, $M_{{\mathrm{{tot}}}}={Mtot}$, $s_t={st}$")
    
    Ax.errorbar(Q,TD,yerr=dTD,fmt="o",color="navy",label=r"$\mathrm{data}$")
    
    def f(x,a,b):
        return(a*x**b)
    
    p=opt.curve_fit(f,Q,TD,sigma=dTD)
    print(" ")
    print("T(q) fit results:")
    print("params:",p[0])
    print("errors:",np.sqrt(np.diag(p[1])))
    
    Q=np.linspace(0.95*min(Q),1.05*max(Q),2000)
    Ax.plot(Q,f(Q,p[0][0],p[0][1]),color="crimson",label=r"$\mathrm{fit}$")
    plt.legend()
    
def Ta(q,e,Mtot,st):
    res=results[(results[:,0]==q) & (results[:,2]==e) & (results[:,3]==Mtot) & (results[:,4]==st)]
    A=res[:,1]
    TD=res[:,5]
    dTD=res[:,6]
    
    fig=plt.figure(figsize=(10,5.5),dpi=300)
    Ax=plt.axes()
    Ax.set_xscale("log")
    Ax.set_yscale("log")
    Ax.grid()
    Ax.set_xlabel(r"$a$")
    Ax.set_ylabel(r"$T_\mathrm{Depletion}$ $\mathrm{in}$ $s$")
    Ax.set_title(rf"$q^{{-1}}={q}$, $e={e}$, $M_{{\mathrm{{tot}}}}={Mtot}$, $s_t={st}$")
    
    Ax.errorbar(A,TD,yerr=dTD,fmt="o",color="navy",label=r"$\mathrm{data}$")
    
    def f(x,a,b):
        return(a*x**b)
    
    p=opt.curve_fit(f,A,TD,sigma=dTD)
    print(" ")
    print("T(a) fit results:")
    print("params:",p[0])
    print("errors:",np.sqrt(np.diag(p[1])))
    
    A=np.linspace(0.95*min(A),1.05*max(A),2000)
    Ax.plot(A,f(A,p[0][0],p[0][1]),color="crimson",label=r"$\mathrm{fit}$")
    plt.legend()
    
def Te(q,a,Mtot,st):
    res=results[(results[:,0]==q) & (results[:,1]==a) & (results[:,3]==Mtot) & (results[:,4]==st)]
    E=res[:,2]
    TD=res[:,5]
    dTD=res[:,6]
    
    fig=plt.figure(figsize=(10,5.5),dpi=300)
    Ax=plt.axes()
    #Ax.set_xscale("log")
    #Ax.set_yscale("log")
    Ax.grid()
    Ax.set_xlabel(r"$e$")
    Ax.set_ylabel(r"$T_\mathrm{Depletion}$ $\mathrm{in}$ $s$")
    Ax.set_title(rf"$q^{{-1}}={q}$, $a={a}$, $M_{{\mathrm{{tot}}}}={Mtot}$, $s_t={st}$")
    
    Ax.errorbar(E,TD,yerr=dTD,fmt="o",color="navy",label=r"$\mathrm{data}$")
    
    def f(x,a,b,c):
        return(a*x**b+c)
    
    p=opt.curve_fit(f,E,TD,sigma=dTD)
    print(" ")
    print("T(e) fit results:")
    print("params:",p[0])
    print("errors:",np.sqrt(np.diag(p[1])))
    
    E=np.linspace(0.95*min(E),1.05*max(E),2000)
    Ax.plot(E,f(E,p[0][0],p[0][1],p[0][2]),color="crimson",label=r"$\mathrm{fit}$")
    plt.legend()

def Tm(q,a,e,st):
    res=results[(results[:,0]==q) & (results[:,1]==a) & (results[:,2]==e) & (results[:,4]==st)]
    Mtot=res[:,3]
    TD=res[:,5]
    dTD=res[:,6]
    
    fig=plt.figure(figsize=(10,5.5),dpi=300)
    Ax=plt.axes()
    Ax.set_xscale("log")
    Ax.set_yscale("log")
    Ax.grid()
    Ax.set_xlabel(r"$M_{\mathrm{tot}}$")
    Ax.set_ylabel(r"$T_\mathrm{Depletion}$ $\mathrm{in}$ $s$")
    Ax.set_title(rf"$q^{{-1}}={q}$, $a={a}$, $e={e}$, $s_t={st}$")
    
    Ax.errorbar(Mtot,TD,yerr=dTD,fmt="o",color="navy",label=r"$\mathrm{data}$")
    
    def f(x,a,b):
        return(a*x**b)
    
    p=opt.curve_fit(f,Mtot,TD,sigma=dTD)
    print(" ")
    print("T(Mtot) fit results:")
    print("params:",p[0])
    print("errors:",np.sqrt(np.diag(p[1])))
    
    Mtot=np.linspace(0.95*min(Mtot),1.05*max(Mtot),2000)
    Ax.plot(Mtot,f(Mtot,p[0][0],p[0][1]),color="crimson",label=r"$\mathrm{fit}$")
    plt.legend()
    
def Ts(q,a,e,Mtot):
    res=results[(results[:,0]==q) & (results[:,1]==a) & (results[:,2]==e) & (results[:,3]==Mtot)]
    st=res[:,4]
    TD=res[:,5]
    dTD=res[:,6]
    
    fig=plt.figure(figsize=(10,5.5),dpi=300)
    Ax=plt.axes()
    Ax.set_xscale("log")
    Ax.set_yscale("log")
    Ax.grid()
    Ax.set_xlabel(r"$s_t$")
    Ax.set_ylabel(r"$T_\mathrm{Depletion}$ $\mathrm{in}$ $s$")
    Ax.set_title(rf"$q^{{-1}}={q}$, $a={a}$, $e={e}$, $M_{{\mathrm{{tot}}}}={Mtot}$")
    
    Ax.errorbar(st,TD,yerr=dTD,fmt="o",color="navy",label=r"$\mathrm{data}$")
    
    def f(x,a):
        return(a+0*x)
    
    p=opt.curve_fit(f,st,TD,sigma=dTD)
    print(" ")
    print("T(st) fit results:")
    print("params:",p[0])
    print("errors:",np.sqrt(np.diag(p[1])))
    
    st=np.linspace(0.95*min(st),1.05*max(st),2000)
    Ax.plot(st,f(st,p[0][0]),color="crimson",label=r"$\mathrm{fit}$")
    plt.legend()

  

def Fit():
    Q=results[:,0]
    A=results[:,1]
    E=results[:,2]
    MTOT=results[:,3]
    ST=results[:,4]
    TD=results[:,5]
    dTD=results[:,6]
    TP=results[:,7]
    dTP=results[:,8]
    Norb=results[:,9]
    dNorb=results[:,10]
    RS=2*G*MTOT/c**2
    A=RS*A
    
    def f(x,a,b,c,d,e):
        return(a*x[0]**b*x[1]**c*x[2]**d*x[3]**e)
    
    p=opt.curve_fit(f,(Q,A,E,MTOT,ST),TD,sigma=dTD)
    print(" ")
    print("total fit results:")
    print("params:",p[0])
    print("errors:",np.sqrt(np.diag(p[1])))
    
    exp=f((Q,A,E,MTOT,ST),p[0][0],p[0][1],p[0][2],p[0][3],p[0][4])
    sigma=(TD-exp)/dTD
    print("rms-test p=",rms_test(sigma))
    print("sup-test p=",sup_test(sigma))
    

Tq(20,0,1e3,1000)
Ta(10,0,1e3,1000)
Te(10,20,1e3,1000)
Tm(10,20,0,1000)
Ts(10,20,0,1e3)
Fit()

