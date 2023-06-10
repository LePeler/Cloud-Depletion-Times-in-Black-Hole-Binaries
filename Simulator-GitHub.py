import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib as mpl
import scipy.stats as stat
import scipy.special as spec
from tqdm import tqdm
import requests
import argparse
np.random.seed(2414)
path="C:/Users/Prince LePéler/Desktop/Programs/Bachelor's Thesis/TEST"
parser=argparse.ArgumentParser(description='This program simulates two BHs with mass ratio q, that are surrounded by a spherical cloud of particles.')
parser.add_argument('-q',type=float,help='mass ratio: M/m')
parser.add_argument('-a',type=float,help='semimajor axis in units of combined Schwarzschild-radius')
parser.add_argument('-numorb',type=int,help='number of orbits to be simulated, default=1000',default=1000,nargs='?')
parser.add_argument('-e',type=float,help='system eccentricity, default=0',default=0,nargs='?')
parser.add_argument('-N',type=int,help='number of cloud particles, default=10000',default=10000,nargs='?')
parser.add_argument('-precision',type=float,help='dt=T/precision, default=10000',default=10000,nargs='?')
parser.add_argument('-binary_mass',type=float,help='Mtot in solar masses, default=1e3',default=1e3,nargs='?')
parser.add_argument('-cloud_mass',type=float,help='st=Mtot/cloud_mass, default=1000',default=1000,nargs='?')
args=parser.parse_args()
##########
G=6.6743e-11
c=299792459
Mtot=args.binary_mass*2e30 ##########
q=args.q##########
M=q/(1+q)*Mtot
m=Mtot/(1+q)
XS=2*G*m/c**2
YS=2*G*M/c**2
a=args.a*(XS+YS)##########
e=args.e##########
st=Mtot/args.cloud_mass ##########
N=args.N##########
s=st/N
T=np.sqrt(4*np.pi**2/G/Mtot*a**3)
fps=24
spo=2
dt=T/args.precision##########
numorb=args.numorb##########
F=int(spo*fps*numorb)
spf=int(T/dt/spo/fps)
S=F*spf
##########
d=M/(M+m)*a
n=1/(1/M+1/m)
L=np.sqrt(G*M*m*n*a*(1-e**2))
E=-G*M*m/2/a
b=a*np.sqrt(1-e**2)
#
params=f"{args.q},{args.a},{args.e},{args.binary_mass},{args.cloud_mass}"

print(f"Program running, simulating {params}: \n Mtot={Mtot} \n q={q} \n m={m} \n M={M} \n XS={XS} \n YS={YS} \n a={a} \n e={e} \n st={st} \n N={N} \n s={s} \n T={T} \n fps={fps} \n spo{spo} \n dt={dt}=T/{int(args.precision)} \n numorb={numorb} \n F={F} \n spf={spf} \n S={S} \n d={d} \n n={n} \n L={L} \n E={E} \n b={b}")
#
u0=a*(1+e)
vu0=L/n/a/(1+e)
x0=np.array([M/(M+m)*u0,0,0])
y0=np.array([-m/(M+m)*u0,0,0])
px0=m*np.array([0,M/(M+m)*vu0,0])
py0=M*np.array([0,-m/(M+m)*vu0,0])
cg0=(m*x0+M*y0)/(m+M)

#cloud initialization
En=-15/24*G*Mtot*s/(2*a)
bet=-1/2/En
c1=bet**(3/2)/(np.sqrt(np.pi)/2-spec.gammainc(3/2,1))
class pE(stat.rv_continuous):
    def _pdf(self,x):
        pb=np.sqrt(x-2*En)*np.exp(-bet*x)
        return(c1*pb)
PE=pE(a=2*En,b=0)
c2=32/np.pi
class pA(stat.rv_continuous):
    def _pdf(self,x):
        pa=np.sinh(x)**2/np.cosh(x)**7
        return(c2*pa)
PA=pA(a=0)

r0=[]
pr0=[]
print("Initializing cloud...")
for k in tqdm(range(N)):
    phi=stat.uniform.rvs(0,2*np.pi)
    beta=stat.uniform.rvs(0,2*np.pi)
    theta=2*stat.anglit.rvs()
    alpha=2*stat.anglit.rvs()
    EN=PE.rvs()
    A=PA.rvs()
    R=-G*(M+m)*s/EN/np.cosh(A)**2
    p=np.sqrt(-2*s*EN*np.sinh(A)**2)
    r0.append([R*np.cos(theta)*np.cos(phi),R*np.cos(theta)*np.sin(phi),R*np.sin(theta)])
    pr0.append([p*np.cos(alpha)*np.cos(beta),p*np.cos(alpha)*np.sin(beta),p*np.sin(alpha)])

r0=np.array(r0)
pr0=np.array(pr0)
Rmax=np.sqrt(max(np.sum(r0**2,axis=1)))
#

def Fx(x,y,r):
    A=-G*M/np.sqrt(np.sum((x-y)**2))**3*(x-y)-G*s*np.sum((x-r)/((np.sum((x-r)**2,axis=1))**(3/2))[:,None],axis=0)
    return(m*A)

def Fy(x,y,r):
    A=-G*m/np.sqrt(np.sum((x-y)**2))**3*(y-x)-G*s*np.sum((y-r)/((np.sum((y-r)**2,axis=1))**(3/2))[:,None],axis=0)
    return(M*A)

def Fr(x,y,r):
    A=-G*m*(r-x)/((np.sum((x-r)**2,axis=1))**(3/2))[:,None]-G*M*(r-y)/((np.sum((y-r)**2,axis=1))**(3/2))[:,None]
    return(s*A)

x=np.array([x0]*(F+1),dtype=float)
y=np.array([y0]*(F+1),dtype=float)
px=np.array([px0]*(F+1),dtype=float)
py=np.array([py0]*(F+1),dtype=float)
cG=np.array([cg0]*(F+1),dtype=float)
mass=np.array([[m,M]]*(F+1),dtype=float)

bins=np.linspace(0,Rmax,30)
h=bins[1]-bins[0]
Rr=np.sqrt(np.sum((r0-cg0)**2,axis=1))
his=np.histogram(Rr,bins)[0]
His=np.array([his]*(F+1))

rr=np.sum((r0-cg0)**2,axis=1)
rhoa=np.array([np.histogram(rr,[0,2*a**2])[0][0]]*(S+1))

X=x0
Y=y0
R=r0
pX=px0
pY=py0
pR=pr0

d1=1/(2-2**(1/3))
d2=-2**(1/3)/(2-2**(1/3))
c1=d1/2
c2=(d1+d2)/2

ri=0
print("Doing numerical simulation...")
for j in tqdm(range(F)):
    for n in range(spf):
        #numerical method
        X=X+c1*dt*c*pX/np.sqrt(m**2*c**2+np.sum(pX**2))
        Y=Y+c1*dt*c*pY/np.sqrt(M**2*c**2+np.sum(pY**2))
        R=R+c1*dt*c*pR/np.sqrt(s**2*c**2+np.sum(pR**2,axis=1))[:,None]
        pX=pX+d1*Fx(X,Y,R)*dt
        pY=pY+d1*Fy(X,Y,R)*dt
        pR=pR+d1*Fr(X,Y,R)*dt
        
        X=X+c2*dt*c*pX/np.sqrt(m**2*c**2+np.sum(pX**2))
        Y=Y+c2*dt*c*pY/np.sqrt(M**2*c**2+np.sum(pY**2))
        R=R+c2*dt*c*pR/np.sqrt(s**2*c**2+np.sum(pR**2,axis=1))[:,None]
        pX=pX+d2*Fx(X,Y,R)*dt
        pY=pY+d2*Fy(X,Y,R)*dt
        pR=pR+d2*Fr(X,Y,R)*dt
        
        X=X+c2*dt*c*pX/np.sqrt(m**2*c**2+np.sum(pX**2))
        Y=Y+c2*dt*c*pY/np.sqrt(M**2*c**2+np.sum(pY**2))
        R=R+c2*dt*c*pR/np.sqrt(s**2*c**2+np.sum(pR**2,axis=1))[:,None]
        pX=pX+d1*Fx(X,Y,R)*dt
        pY=pY+d1*Fy(X,Y,R)*dt
        pR=pR+d1*Fr(X,Y,R)*dt
        
        X=X+c1*dt*c*pX/np.sqrt(m**2*c**2+np.sum(pX**2))
        Y=Y+c1*dt*c*pY/np.sqrt(M**2*c**2+np.sum(pY**2))
        R=R+c1*dt*c*pR/np.sqrt(s**2*c**2+np.sum(pR**2,axis=1))[:,None]
        
        #absorption mechanism
        cond1=np.where(np.sum((R-Y)**2,axis=1)<YS**2)[0]
        M=np.sqrt((np.sqrt(M**2*c**2+np.sum(pY**2))+np.sum(np.sqrt(s**2*c**2+np.sum(pR[cond1]**2,axis=1))))**2-np.sum((pY+np.sum(pR[cond1],axis=0))**2))/c
        pY=pY+np.sum(pR[cond1],axis=0)
        R=np.delete(R,cond1,axis=0)
        pR=np.delete(pR,cond1,axis=0)
        cond2=np.where(np.sum((R-X)**2,axis=1)<XS**2)[0]
        m=np.sqrt((np.sqrt(m**2*c**2+np.sum(pX**2))+np.sum(np.sqrt(s**2*c**2+np.sum(pR[cond2]**2,axis=1))))**2-np.sum((pX+np.sum(pR[cond2],axis=0))**2))/c
        pX=pX+np.sum(pR[cond2],axis=0)
        R=np.delete(R,cond2,axis=0)
        pR=np.delete(pR,cond2,axis=0)
        YS=2*G*M/c**2
        XS=2*G*m/c**2
        n=1/(1/M+1/m)
        
        #calculate cloud density
        ri=ri+1
        cg=(m*X+M*Y)/(m+M)
        rr=np.sum((R-cg)**2,axis=1)
        rhoa[ri]=np.histogram(rr,[0,2*a**2])[0][0]
        
    x[j+1]=X
    y[j+1]=Y
    px[j+1]=pX
    py[j+1]=pY
    cG[j+1]=cg
    mass[j+1]=np.array([m,M])
    
    Rr=np.sqrt(np.sum((R-cG[j+1])**2,axis=1))
    his=np.histogram(Rr,bins)[0]
    His[j+1]=his

#
print(rhoa)
Data=np.array([np.linspace(0,S,S+1)*dt,rhoa],dtype=np.float64)
np.save(path+f"/Depletion-Data/Data,{params}.npy",Data)
His=His.astype(float)
His[F]=bins[1:]
DData=np.array(np.insert(His,0,(np.linspace(0,F,F+1)*spf*dt,mass[:,0],x[:,0],x[:,1],x[:,2],px[:,0],px[:,1],px[:,2],mass[:,1],y[:,0],y[:,1],y[:,2],py[:,0],py[:,1],py[:,2]),axis=1),dtype=np.float64)
np.save(path+f"/Dynamic-Data/DData,{params}.npy",DData)
#

print("The program is finished!")

