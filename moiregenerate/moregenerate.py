
import numpy as np
import scipy.optimize as opt


#逆时针旋转一个二维向量
def rotate2dv(v,theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return np.dot(R, v)

class moredata(object):
    def __init__(self) -> None:
        super().__init__()
        self.A=np.array([[1,2],[2,1]]) # 下层
        self.B0=np.array([[1,2],[2,1]]) # B0 为初始的B，不转动时的格矢
        self.theta=0
        self.epsilon=0.01 # 晶格矢量允许误差
        self.lepsilon=0.01 # 晶格面积误差
        self.maxm=10
        self.max_lattice_length=3000
        self.minangle=np.pi/7
        self.maxLrate=3.2
        
        self.dtheta=np.pi/180*3
    
    def changetheta(self,theta):
        self.theta=theta
        self.B=rotate2dv(self.B0.T,theta).T # self.B	上层晶格（旋转后）
        
    
    def getallchoose(self):
        allchoose=[]
        for i in range(-self.maxm,self.maxm):
            for j in range(-self.maxm,self.maxm):
                if i!=0 or j!=0:
                    allchoose.append([i,j])
        allchoose=np.array(allchoose,dtype=np.int32)
        allchoose1=np.dot(allchoose,self.A) # 实空间可选向量
        a=np.linalg.norm(allchoose1,axis=1)
        mask=a<self.max_lattice_length
        a=a[mask]
        allchoose=allchoose[mask]
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
        
        angle=np.abs(np.arcsin(areas/lll/lll[0])) # 这个角度可能不对可能需要取绝对值
        
        # sortdep=(areas-areas.min())/(areas.max()-areas.min())
        # sortdep[0]=sortdep.max()*100
        sortdep=areas+area0/2*lll/self.lengthOfChoose[-1]
        index=np.arange(len(areas))
        ##index=index[angle>np.pi/7]
        
        #index=index[np.logical_and(angle>self.minangle,lll/lll[0]<self.maxLrate,areas>0.1)] #防止歧变晶格
        index=index[np.logical_and(angle>self.minangle,areas>0.1)] #防止歧变晶格
        index=index[np.argsort(sortdep[index])]
        # if np.allclose(self.theta,np.pi/6,np.pi/180/2):
        #print(mns[0],mns[index].__repr__(),areas[index],sortdep[index])
        
        for i in range(0,len(index)):
            print(i)
            # newab.append(mns[index[i]])
            #print("mismatch: ",areas[index[i]],self.getmismactch(np.array([mns[0],mns[index[i]]]),self.theta))
            mn=np.array([mns[0],mns[index[i]]])
            latticemis = self.getmismactch(np.array([mns[0],mns[index[i]]]),self.theta)
            fun = lambda x: self.getmismactch(mn,x[0])
       
            #res=opt.minimize(fun,np.array([theta]),bounds=[(theta-self.dtheta,theta+self.dtheta)],method="COBYLA")
            res =opt.minimize(fun,x0=self.theta)
            print(res.x[0]*180/np.pi,mn[1],res.success,res.fun)
            if latticemis < self.lepsilon:
                latticemis0=self.getmismactch(np.array([mns[0],mns[index[i]]]),self.theta-self.dtheta)
                latticemis1=self.getmismactch(np.array([mns[0],mns[index[i]]]),self.theta+self.dtheta)
                # print("mismatch: ",latticemis0-latticemis,latticemis1)
                # print("theta:",self.theta*180/np.pi,"mismatch: ",latticemis0-latticemis,latticemis1-latticemis,[mns[0],mns[index[i]]])
                if latticemis0>latticemis and latticemis1>latticemis:
                    newab.append(mns[index[i]])
                    matcharea=areas[index[i]]
                    break
        if len(newab)<2:
            return None,0
        #print("matched:",newab)
        return np.array(newab[:2]),matcharea
    


    # ======= 新增：一次性批量计算多个组合、多个角度的 mismatch =======
    def _batch_mismatch_multi(self, mn_pairs, thetas):
        """
        mn_pairs: 形状 (P, 2, 2)，每个元素是两个候选超胞基矢系数构成的 2x2 矩阵
        thetas:   长度 T 的角度数组（例如 [theta, theta-dtheta, theta+dtheta]）
        return:   形状 (P, T) 的 mismatch 最大奇异值（谱范数）
        """
        thetas = np.asarray(thetas, dtype=float)
        T = thetas.shape[0]
        P = mn_pairs.shape[0]

        # 预计算 B(θ), C(θ) = A @ inv(B(θ))，并堆叠为 (T, 2, 2)
        B_stack = np.stack([rotate2dv(self.B0.T, th).T for th in thetas], axis=0)  # (T,2,2)
        invB_stack = np.linalg.inv(B_stack)                                        # (T,2,2)
        C_stack = np.matmul(self.A, invB_stack)                                    # (T,2,2)

        # 扩展 mn_pairs 到 (P, 1, 2, 2)，使其与 (T,2,2) 广播
        mn_exp = mn_pairs[:, None, :, :]                                           # (P,1,2,2)

        # result = mn @ C(θ)  ->  (P,T,2,2)
        result = np.matmul(mn_exp, C_stack)
        mn2 = np.round(result)                                                     # (P,T,2,2)

        # sA = mn @ A（与 θ 无关），通过广播与 (P,T,2,2) 对齐
        sA = np.matmul(mn_exp, self.A)                                            # (P,1,2,2)

        # sB = mn2 @ B(θ) -> (P,T,2,2)
        sB = np.matmul(mn2, B_stack)

        # 使用平均基矢求 U
        sAavg = 0.5 * (sA + sB)                                                   # (P,T,2,2)

        # 为避免奇异，加入极小对角正则
        epsI = 1e-12 * np.eye(2)[None, None, :, :]
        inv_sAavg = np.linalg.inv(sAavg + epsI)                                   # (P,T,2,2)

        U = np.matmul(inv_sAavg, sB)                                              # (P,T,2,2)

        # MN = U^T - I
        MN = np.swapaxes(U, -1, -2) - np.eye(2)[None, None, :, :]                 # (P,T,2,2)

        # 谱范数 = 最大奇异值（一次性 SVD）
        svals = np.linalg.svd(MN, compute_uv=False)                               # (P,T,2)
        mismatch = svals[..., 0]                                                  # (P,T)
        return mismatch

    # ======= 新增：全组合 + 批量 mismatch + 局域极小筛选 + 多目标排序 =======
    def getminmnplot_v2(self):
        """
        返回： (mn_best, matcharea)；若无可行解，返回 (None, 0)
        """
        # 1) 基于当前 self.theta（外部已调用 changetheta 设置）获取候选一维超胞向量集合 mns
        C = np.dot(self.A, np.linalg.inv(self.B))
        result = self.allchoose.dot(C)                                            # (K,2)
        allchoose2 = np.round(result)                                             # (K,2)
        dis0 = (np.dot(allchoose2, self.B) - np.dot(self.allchoose, self.A))      # (K,2)
        denom = np.linalg.norm(np.dot(allchoose2, self.B), axis=1)                # (K,)
        # 防止除零
        denom = np.where(denom == 0, 1e-15, denom)
        dis = np.linalg.norm(dis0, axis=1) / denom
        mns = self.allchoose[dis < self.epsilon]                                  # (K',2)

        if mns.shape[0] < 2:
            return None, 0

        # 2) 预先计算几何量：笛卡尔矢量、长度
        cart = np.dot(mns, self.A)                                                # (K',2)
        lengths = np.linalg.norm(cart, axis=1)                                    # (K',)

        # 3) 生成所有两两组合（i<j）
        i_idx, j_idx = np.triu_indices(mns.shape[0], k=1)
        cart_i, cart_j = cart[i_idx], cart[j_idx]                                 # (P,2)
        len_i, len_j = lengths[i_idx], lengths[j_idx]                             # (P,)

        # 4) 面积、夹角与筛选（避免共线/病态组合）
        cross_ij = np.abs(cart_i[:, 0] * cart_j[:, 1] - cart_i[:, 1] * cart_j[:, 0])  # (P,)
        denom_ang = np.clip(len_i * len_j, 1e-15, None)
        sinang = np.clip(cross_ij / denom_ang, 0.0, 1.0)
        angle = np.arcsin(sinang)                                                 # (P,)
        # 长度比（较大/较小）
        ratio = np.maximum(len_i, len_j) / np.clip(np.minimum(len_i, len_j), 1e-15, None)

        # 准则：角度>最小角、面积>0.1、防止长度极端不均衡
        mask_geom = (angle > self.minangle) & (cross_ij > 0.1) & (ratio < self.maxLrate)
        if not np.any(mask_geom):
            return None, 0

        i_idx, j_idx = i_idx[mask_geom], j_idx[mask_geom]
        cart_i, cart_j = cart_i[mask_geom], cart_j[mask_geom]
        len_i, len_j = len_i[mask_geom], len_j[mask_geom]
        cross_ij = cross_ij[mask_geom]
        angle = angle[mask_geom]

        # 5) 构造所有候选 2x2 超胞矩阵（组合）
        mn_pairs = np.stack([mns[i_idx], mns[j_idx]], axis=1)                     # (P',2,2)

        # 6) 一次性计算三个角度的 mismatch，并做“局域极小”筛选
        thetas = [self.theta, self.theta - self.dtheta, self.theta + self.dtheta]
        mm = self._batch_mismatch_multi(mn_pairs, thetas)                         # (P',3)
        m0, m_minus, m_plus = mm[:, 0], mm[:, 1], mm[:, 2]

        # 条件：当前角度处 mismatch 更小（局域极小）且绝对值小于 lepsilon
        mask_local_min = (m0 < m_minus) & (m0 < m_plus) & (m0 < self.lepsilon)
        if not np.any(mask_local_min):
            return None, 0

        # 7) 在局域极小的集合中，按“面积最小 → 长度差最小 → 夹角最接近90°”排序
        area_sel = cross_ij[mask_local_min]
        ldiff_sel = np.abs(len_i[mask_local_min] - len_j[mask_local_min])
        ang90_dev = np.abs(np.pi / 2.0 - angle[mask_local_min])
        mn_sel = mn_pairs[mask_local_min]

        # lexsort 最后一个键为主键：我们要 area 最优（主），其次 ldiff，再次 ang90_dev
        order = np.lexsort((ang90_dev, ldiff_sel, area_sel))
        best = order[0]

        mn_best = mn_sel[best]                    # (2,2)
        matcharea = area_sel[best].item()

        return mn_best, matcharea

    
    def getminmn_one(self):
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
        
        
        mns=mns[np.argsort(lengths)]
        
        cart = np.dot(mns,self.A)



        if mns.shape[0]<2:
            return None
        newab=[mns[0]]
        areas=np.cross(cart[0],cart)
        areas=np.abs(areas)
        area0=np.abs(np.linalg.det(self.A))
        lll = np.linalg.norm(cart,axis=1)
        
        angle=np.abs(np.arcsin(areas/lll/lll[0]))
        
        sortdep=areas+area0/2*lll/self.lengthOfChoose[-1]
        index=np.arange(len(areas))
        ##index=index[angle>np.pi/7]
        
        index=index[np.logical_and(angle>self.minangle,lll/lll[0]<self.maxLrate,areas>0.1)] #防止歧变晶格
        index=index[np.argsort(sortdep[index])]
        newab.append(mns[index[0]])
        matcharea=areas[index[0]]
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
        # return  np.linalg.norm(MN, ord='fro')
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


def main():

    import argparse
    import os
    import ase.io as aio
    parser = argparse.ArgumentParser(description='generate a structure')
    parser.add_argument("files", type=str, default=None,nargs=2, help='file name')
    parser.add_argument('-o', '--output', type=str, default='tmpoutput', help='output dir')
    parser.add_argument("-r","--range",type=float,default=[0,180,1000],nargs=3,help="theta range")
    parser.add_argument("-e","--epsilon",type=float,default=0.04,help="epsilon 晶格矢量允许误差")
    parser.add_argument("-l","--lepsilon",type=float,default=0.04,help="lepsilon 晶格面积误差")
    parser.add_argument("-m","--maxm",type=int,default=10,help="maxmium supercell size,搜索的时候建议依次增大，可以避免找不到小原胞")
    parser.add_argument("--distance",type=float,default=3.04432,help="distance between two supercells")
    parser.add_argument("--needshift",action='store_true',help="need shift")
    args=parser.parse_args()
    if args.files is None or len(args.files)!=2:
        print("please input two files")
        exit()
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    b00=aio.read(args.files[1])
    b=aio.read(args.files[0])
    print(args.files[0],b.get_cell().lengths(), b.get_cell().angles())
    print(args.files[1],b00.get_cell().lengths(), b00.get_cell().angles())
    a=moredata()

    a.A=b.get_cell().array[:2,:2]
    a.B0=b00.get_cell().array[:2,:2]
    a.maxm=args.maxm #最大超胞大小
    a.getallchoose() #获取所有可能的超胞
    a.epsilon=args.epsilon #超胞匹配的精度 
    a.lepsilon=args.lepsilon #超胞匹配的精度
    thetalist=np.linspace(args.range[0]*np.pi/180,args.range[1]*np.pi/180,int(args.range[2])) #搜索的角度范围
    a.dtheta=thetalist[1]-thetalist[0]#max(0.01*np.pi/180,thetalist[1]-thetalist[0]) #搜索的角度步长
    print("args.range",args.range)
    kexing=[]
    outs=[]
    pltdata=[]
    import tqdm
    unikey=set()
    def getunikey(angle,mn):
        return f"{angle:.3f}_{mn[0][0]}_{mn[0,1]}_{mn[1][0]}_{mn[1][1]}"
    for theta in tqdm.tqdm(thetalist):   
        try :
            a.changetheta(theta)
            mn,area=a.getminmnplot_v2()
            if mn is None:
                continue
            if mn.shape==(2,2):
                res=a.relaxwithmn(mn)
                # key=getunikey(res.x,mn)
                # if key in unikey:
                #     continue
                # unikey.add(key)
                outs.append((res.x,area,mn))
                pltdata.append([res.x,area])
                if res.success:
                    print("success:",theta/np.pi*180,res.x/np.pi*180,mn)
                    kexing.append((res.x,res.fun,mn,area))
        except BaseException as e:
            print(e)
            # raise e
            continue     
    print([[i[0]/np.pi*180,i[2],i[3]] for i in kexing])
    
if __name__ == "__main__":
    print(__file__)
    main()