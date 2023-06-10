import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import scipy.optimize as opt
import scipy.interpolate as interp
plt.rc('font', size=18)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
path="C:/Users/Prince LePéler/Desktop/Programs/Bachelor's Thesis/TEST"
DData=path+"/Dynamic-Data"
Data=path+"/Depletion-Data"
Dfiles=[f for f in os.listdir(DData) if os.path.isfile(os.path.join(DData, f))]
files=[f for f in os.listdir(Data) if os.path.isfile(os.path.join(Data, f))]


def f(n):
    params=files[n].split(",")
    q=float(params[1])
    a=float(params[2])
    e=float(params[3])
    Mtot=float(params[4])
    st=float(params[5].split(".")[0])
    params=f"{q},{a},{e},{Mtot},{st}"
    
    data=np.load(Data+"/"+files[n])
    t=data[0]
    rhoa=data[1]
    
    ddata=np.load(DData+"/"+Dfiles[n])
    T=ddata[:,0]
    m=ddata[:,1]
    x=ddata[:,2:5]
    px=ddata[:,5:8]
    M=ddata[:,8]
    y=ddata[:,9:12]
    py=ddata[:,12:15]
    #His=ddata[:,15:]
    #bins=His[len(His)-1]
    #His=His[:len(His)-1]
    X=x[0][0]
    G=6.6743e-11
    c=299792458
    RS=2*G*Mtot*2e30/c**2
    Tp0=2*np.pi/np.sqrt(G*Mtot*2e30)*(RS*a)**(3/2)
    w0=2*np.pi/Tp0
    def wave(t,w):
        return(x[0][0]*np.cos(w*t))
    k=13*48
    p=opt.curve_fit(wave,T[:k],x[:k,0],p0=w0)
    w=p[0][0]
    dw=np.sqrt(p[1][0][0])
    Tp=2*np.pi/w
    dTp=Tp*dw/w
    
    
    N=15002
    le=int(len(t)/N)
    TT=np.zeros(le)
    y_mean=np.zeros(le)
    y_max=np.zeros(le)
    y_min=np.zeros(le)
    for K in range(le):
        k=N*K
        tt=t[k:k+N]
        arr=rhoa[k:k+N]
        TT[K]=np.mean(tt)
        Y=np.mean(arr)
        dY=np.std(arr)
        y_mean[K]=Y
        y_max[K]=Y+dY
        y_min[K]=Y-dY
    
    f_mean=interp.interp1d(TT,y_mean/y_mean[0],kind="cubic")
    f_max=interp.interp1d(TT,y_max/y_mean[0],kind="cubic")
    f_min=interp.interp1d(TT,y_min/y_mean[0],kind="cubic")
    x_new=np.linspace(TT[0],TT[le-1],2000)
    y_mean_new=f_mean(x_new)
    y_max_new=f_max(x_new)
    y_min_new=f_min(x_new)

    fig=plt.figure(figsize=(10,5),dpi=300)
    Ax=plt.axes()
    Ax.grid()
    Ax.set_title(rf"$q^{{-1}}={int(q)}$ $,$ $a={int(a)}$ $,$ $e={e}$ $,$ $M_{{\mathrm{{tot}}}}={int(Mtot)}$ $,$ $s_t=M_{{\mathrm{{tot}}}}/{int(st)}$")
    Ax.set_xlabel(r"$t$ $\mathrm{in}$ $s$")
    Ax.set_ylabel(r"$\mathrm{relative}$ $\mathrm{cloud}$ $\mathrm{density}$ $\mathrm{around}$ $\mathrm{BH}$ $\mathrm{X}$")
    Ax.plot(t,rhoa/y_mean[0],color="crimson",label=r"$\mathrm{data}$")
    Ax.plot(x_new,y_mean_new,color="orange",linewidth=1,label=r"$\mathrm{mean}$")
    Ax.plot(x_new,y_max_new,color="blue",linewidth=1,label=r"$\mathrm{mean}+\mathrm{std}$")
    Ax.plot(x_new,y_min_new,color="blue",linewidth=1,label=r"$\mathrm{mean}-\mathrm{std}$")
    Ax.plot([t[0],t[len(t)-1]],[1/np.e,1/np.e],color="green",label=r"$\frac{1}{e}$ $\mathrm{threshold}$")
    plt.legend()
    plt.show()
    plt.close()
    
    def F_mean(x):
        return(f_mean(x)-1/np.e)
    def F_max(x):
        return(f_max(x)-1/np.e)
    def F_min(x):
        return(f_min(x)-1/np.e)
    
    try:
        T_mean=opt.newton(F_mean,x0=TT[1],maxiter=500)
        T_max=opt.newton(F_max,x0=TT[1],maxiter=500)
        T_min=opt.newton(F_min,x0=TT[1],maxiter=500)
        dT=(T_max-T_min)/2
        n=T_mean/Tp
        dn=np.sqrt(dT**2/Tp**2+T_mean**2*dTp**2/Tp**4)
        result=np.array([q,a,e,Mtot,st,T_mean,dT,Tp,dTp,n,dn],dtype=np.float64)
        np.save(path+f"/Results/res,{params}.npy",result)
        print(f"Success for {params}")
    
    except:
        print(f"Error in {params}")
        pass
    

for n in range(len(files)):
    f(n)