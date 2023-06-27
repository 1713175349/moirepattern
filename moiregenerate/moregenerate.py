from hashlib import new
from turtle import color
from xmlrpc.client import NOT_WELLFORMED_ERROR
import numpy as np
import scipy.optimize as opt
from sklearn.preprocessing import scale


#逆时针旋转一个二维向量
def rotate2dv(v,theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return np.dot(R, v)

class moredata(object):
    def __init__(self) -> None:
        super().__init__()
        self.A=np.array([[1,2],[2,1]])
        self.B0=np.array([[1,2],[2,1]]) # B0 为初始的B，不转动时的格矢
        self.theta=0
        self.epsilon=0.01
        self.lepsilon=0.01
        self.maxm=10
        self.minangle=np.pi/7
        self.maxLrate=2.2
        
        self.dtheta=np.pi/180*3
    
    def changetheta(self,theta):
        self.theta=theta
        self.B=rotate2dv(self.B0.T,theta).T
    
    def getallchoose(self):
        allchoose=[]
        for i in range(-self.maxm,self.maxm):
            for j in range(-self.maxm,self.maxm):
                if i!=0 or j!=0:
                    allchoose.append([i,j])
        allchoose=np.array(allchoose,dtype=np.int32)
        allchoose1=np.dot(allchoose,self.A)
        a=np.linalg.norm(allchoose1,axis=1)
        self.lengthOfChoose=a
        #sort allchoose by a
        allchoose=allchoose[np.argsort(a)]
        self.allchoose=allchoose

    def getminmn(self):
        C=np.dot(self.A,np.linalg.inv(self.B))
        result=self.allchoose.dot(C)
        result1=np.round(result)-result

        allchoose2 = np.round(result)
        dis0=(np.dot(allchoose2,self.B)-np.dot(self.allchoose,self.A))
        dis= np.linalg.norm(dis0,axis=1)/np.linalg.norm(np.dot(allchoose2,self.B),axis=1)
        # dis=np.linalg.norm(np.dot(result1,self.A),axis=1)/np.linalg.norm(np.dot(result,self.A),axis=1)
        mns = self.allchoose[dis<self.epsilon]
        
        
        
        
        cart = np.dot(mns,self.A)

        if mns.shape[0]<2:
            return None
        newab=[mns[0]]
        areas=np.cross(cart[0],cart)
        areas=np.abs(areas)
        area0=np.abs(np.linalg.det(self.A))
        lll = np.linalg.norm(cart,axis=1)
        sortdep=areas+area0/2*lll/self.lengthOfChoose[-1]
        index=np.arange(len(areas))
        index=index[sortdep>area0*0.8]
        index=index[np.argsort(sortdep[index])]
        
        for i in range(0,len(index)):
            # newab.append(mns[index[i]])
            #print("mismatch: ",areas[index[i]],self.getmismactch(np.array([mns[0],mns[index[i]]]),self.theta))
            latticemis = self.getmismactch(np.array([mns[0],mns[index[i]]]),self.theta)
            
            if latticemis < self.lepsilon:
                latticemis0=self.getmismactch(np.array([mns[0],mns[index[i]]]),self.theta-self.dtheta)
                latticemis1=self.getmismactch(np.array([mns[0],mns[index[i]]]),self.theta+self.dtheta)
                # print("mismatch: ",latticemis0-latticemis,latticemis1)
                if latticemis0>latticemis and latticemis1>latticemis:
                    newab.append(mns[index[i]])
                    matcharea=areas[index[i]]
                    break
        return np.array(newab[:2])

    def getminmnplot(self):
        C=np.dot(self.A,np.linalg.inv(self.B))

        
        result=self.allchoose.dot(C)
        result1=np.round(result)-result

        allchoose2 = np.round(result)
        dis0=(np.dot(allchoose2,self.B)-np.dot(self.allchoose,self.A))
        dis= np.linalg.norm(dis0,axis=1)/np.linalg.norm(np.dot(allchoose2,self.B),axis=1)
        # dis=np.linalg.norm(np.dot(result1,self.A),axis=1)/np.linalg.norm(np.dot(result,self.A),axis=1)
        mns = self.allchoose[dis<self.epsilon]
        dis=dis[dis<self.epsilon]
        #mns=mns[np.argsort(dis)]
        lengths=np.linalg.norm(np.dot(mns,self.A),axis=1)
        
        # dis=dis[np.argsort(dis)]
        # # mns=mns[np.argsort((lengths/np.max(lengths)+100000000000000000000000000000*dis/np.max(dis)))]
        # #求最小的那个点附近的最短格矢量
        # minlenth=lengths[0]
        # minindex=0
       
        # for i in range(0,len(mns)):
        #     if lengths[i]<minlenth and np.abs(dis[i]-dis[0])<0.01:
        #         minlenth=lengths[i]
        #         minindex=i
        # # swap to 0
        
        
        # mns[0],mns[minindex] = mns[minindex],mns[0]
        
        mns=mns[np.argsort(lengths)]
        # outmns=[]
        # for i in mns:
        #     coline=False
        #     for j in outmns:
        #         if np.abs(np.cross(i,j)/np.linalg.norm(i)/np.linalg.norm(j))<0.001:
        #             coline=True
        #             break
        #     if not coline:
        #         outmns.append(i)
                
        # #print(len(mns)-len(outmns))
        # mns=np.array(outmns)  
        
        # print(minindex)
        cart = np.dot(mns,self.A)



        if mns.shape[0]<2:
            return None
        newab=[mns[0]]
        areas=np.cross(cart[0],cart)
        areas=np.abs(areas)
        area0=np.abs(np.linalg.det(self.A))
        lll = np.linalg.norm(cart,axis=1)
        
        angle=np.arcsin(areas/lll/lll[0])
        
        # sortdep=(areas-areas.min())/(areas.max()-areas.min())
        # sortdep[0]=sortdep.max()*100
        sortdep=areas+area0/2*lll/self.lengthOfChoose[-1]
        index=np.arange(len(areas))
        ##index=index[angle>np.pi/7]
        
        index=index[np.logical_and(angle>self.minangle,lll/lll[0]<self.maxLrate,areas>0.1)] #防止歧变晶格
        index=index[np.argsort(sortdep[index])]
        # if np.allclose(self.theta,np.pi/6,np.pi/180/2):
        #print(mns[0],mns[index].__repr__(),areas[index],sortdep[index])
        for i in range(0,len(index)):
            # newab.append(mns[index[i]])
            #print("mismatch: ",areas[index[i]],self.getmismactch(np.array([mns[0],mns[index[i]]]),self.theta))
            latticemis = self.getmismactch(np.array([mns[0],mns[index[i]]]),self.theta)
            
            if latticemis < self.lepsilon:
                latticemis0=self.getmismactch(np.array([mns[0],mns[index[i]]]),self.theta-self.dtheta)
                latticemis1=self.getmismactch(np.array([mns[0],mns[index[i]]]),self.theta+self.dtheta)
                # print("mismatch: ",latticemis0-latticemis,latticemis1)
                if latticemis0>latticemis and latticemis1>latticemis:
                    newab.append(mns[index[i]])
                    matcharea=areas[index[i]]
                    break
        if len(newab)<2:
            return None,0
        #print("matched:",newab)
        return np.array(newab[:2]),matcharea

    def plotvector(self,mn,theta):
        import matplotlib.pyplot as plt
        B=rotate2dv(self.B0.T,theta).T
        C=np.dot(self.A,np.linalg.inv(B))
        sA=np.dot(mn,self.A)
        result=mn.dot(C)
        result1=np.round(result)-result
        mn2 = np.round(result)
        sB=np.dot(mn2,B)
        sA=(sA+sB)/2
        plt.arrow(0,0,sA[0,0],sA[0,1],head_width=0.05,head_length=0.1,color='r')
        plt.arrow(0,0,sA[1,0],sA[1,1],head_width=0.05,head_length=0.1,color='r')
        plt.arrow(0,0,sB[0,0],sB[0,1],head_width=0.05,head_length=0.1,color='b')
        plt.arrow(0,0,sB[1,0],sB[1,1],head_width=0.05,head_length=0.1,color='b')
        point0=self.allchoose.dot(sA)
        point1=self.allchoose.dot(sB)
        print(point0.shape)
        plt.scatter(point0[:,0],point0[:,1],color='r',s=0.1)
        plt.scatter(point1[:,0],point1[:,1],color='b',s=0.1)
        plt.axis('equal')
        U=np.linalg.inv(sA).dot(sB)
        point3=np.dot(point0,U)
        MN=(U.T-np.eye(2))
        # MN=(MN+MN.T)/2
        print("MN norm:",MN,np.linalg.norm(MN,ord=2))
        plt.scatter(point3[:,0],point3[:,1],color='g',s=0.1)
        plt.show()

    def getmismactch(self,mn,theta):
        B=rotate2dv(self.B0.T,theta).T
        C=np.dot(self.A,np.linalg.inv(B))
        result=mn.dot(C)
        result1=np.round(result)-result

        mn2 = np.round(result)
        sA=np.dot(mn,self.A)
        sB=np.dot(mn2,B)
        sA0=sA
        sA=(sA+sB)/2
        U=np.linalg.inv(sA).dot(sB)
        MN=(U.T-np.eye(2))
        # print(sA-sB,MN)
        # MN=(MN+MN.T)/2
        # print("MN norm:",MN,np.linalg.norm(MN,ord=2))
        return np.linalg.norm(MN,ord=2)
    
    def getinformation(self,mn,theta):
        pass

    def relaxwithmn(self,mn):
        theta=self.theta
        # fun = lambda x: self.getmismactch(mn,x[0])

        # res=opt.minimize(fun,np.array([theta]),bounds=[(theta-self.dtheta,theta+self.dtheta)],method="COBYLA")
        fun = lambda x: self.getmismactch(mn,x)
        #res=opt.minimize(fun,np.array([theta]),bounds=[(theta-self.dtheta,theta+self.dtheta)],method="COBYLA")
        res =opt.minimize_scalar(fun,method="bounded",bounds=(theta-self.dtheta,theta+self.dtheta))
        return res

    def relaxwithmn2(self,mn):
        theta=self.theta
        fun = lambda x: self.getmismactch(mn,x[0])

        # res=opt.minimize(fun,np.array([theta]),bounds=[(theta-self.dtheta,theta+self.dtheta)],method="COBYLA")
        res= opt.minimize_scalar(fun,method="Golden",bounds=(0,np.pi))
        return res



if __name__=="__main__":
    #用于生成一个晶格可行的摩尔结构的可行角度
    import ase.io as aio
    b=aio.read("POSCAR")

    a=moredata()
    print(b.get_cell().angles())
    a.A=b.get_cell().array[:2,:2]
    a.B0=b.get_cell().array[:2,:2]
    # a.A=np.array([[1,0],[-np.sqrt(3)/2,1/2]])
    # a.B0=np.array([[1,0],[-np.sqrt(3)/2,1/2]])
    a.maxm=10 #最大超胞大小
    a.getallchoose() #获取所有可能的超胞
    
    a.epsilon=0.04 #超胞匹配的精度 

    thetalist=np.linspace(0,np.pi/3,1000) #搜索的角度范围
    a.dtheta=thetalist[1]-thetalist[0] #搜索的角度步长

    # for n1 in a.allchoose:
    #     for n2 in a.allchoose:
    #         if np.abs(np.cross(n1,n2)) < 1e-6:
    #             continue
    #         a.relaxwithmn2(n1)

    outs=[]
    pltdata=[]
    for theta in thetalist:
        # print(theta)
        try :
            a.changetheta(theta)
            mn,area=a.getminmnplot()
            if mn.shape==(2,2):
                res=a.relaxwithmn(mn)
                print(res.x,res.fun,res.success)
                if res.success:
                    outs.append((res.x,area,mn))
                    pltdata.append([res.x,area])
                    
        except BaseException as e:
            #print(e)
            continue
    
    import matplotlib.pyplot as plt
    pltdata=np.array(pltdata)
    #plt.scatter(pltdata[:,0],pltdata[:,1])
    plt.scatter(pltdata[:,0]*180/np.pi,pltdata[:,1],s=10)
    plt.xlabel("Theta",fontsize=25)
    plt.ylabel("Area",fontsize=25)
    # 隐藏y轴的刻度
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()
    # print(outs)
    theta=0
    a.changetheta(theta/180*np.pi)
    mn,area=a.getminmnplot()
    print(mn,area)
    print(a.getmismactch(mn,theta/180*np.pi))
    a.plotvector(mn,theta/180*np.pi)
    #print(a.getmismactch(mn,46.908/180*np.pi))
    # res=a.relaxwithmn(mn)
    # print(res.x[0]/np.pi*180)

    # kexing=[]

    # realmin=0.06
    # for t in thetalist:
    #     a.changetheta(t)
    #     mn=a.getminmn()
    #     if mn is not None:
    #         res=a.relaxwithmn(mn)
    #         if (len(kexing) == 0 or (np.abs(res.x[0]-kexing[-1][0])>0.02 or (kexing[-1][2]!=mn).any() )):
    #             kexing.append((res.x[0],res.fun,mn))
                
    # # kexing=np.array(kexing)

    # for i in kexing:
    #     print(i[0]*180/np.pi,i[1],i[2][0],i[2][1])

    # a.changetheta(np.pi/180*58)
    # a.epsilon=0.04
    # mn=a.getminmn()

    # print(mn,a.getmismactch(mn,a.theta))
    # ntheta=a.relaxwithmn(mn)
    # print(ntheta.x[0]*180/np.pi,a.getmismactch(mn,ntheta.x[0]))