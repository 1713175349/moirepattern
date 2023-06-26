#用于构建摩尔纹结构，对输入文件要求z轴为真空层方向，需要给定层间距

from email import parser

import ase.io as aio
from ase.build import supercells,sort
from ase.visualize import view
from ase import atoms,Atoms
import numpy as np
import os
import argparse
from ase.geometry import get_distances
from .moregenerate import *

def get_thickness(a:Atoms):
    '''
    获取原子层间距
    '''
    z = a.get_positions()[:,2]
    zmax=np.max(z)
    zmin=np.min(z)
    return zmax-zmin

def get_rotation_matrix(theta):
    '''
    获取3D旋转矩阵
    '''
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    return R

def get_rotation_matrix2d(theta):
    '''
    获取2D旋转矩阵
    '''
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R


def lattice_points_in_supercell(supercell_matrix):
    """
    Returns the list of points on the original lattice contained in the
    supercell in fractional coordinates (with the supercell basis).
    e.g. [[2,0,0],[0,1,0],[0,0,1]] returns [[0,0,0],[0.5,0,0]]

    Args:
        supercell_matrix: 3x3 matrix describing the supercell

    Returns:
        numpy array of the fractional coordinates
    """
    diagonals = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    d_points = np.dot(diagonals, supercell_matrix)

    mins = np.min(d_points, axis=0)
    maxes = np.max(d_points, axis=0) + 1

    ar = np.arange(mins[0], maxes[0])[:, None] * np.array([1, 0, 0])[None, :]
    br = np.arange(mins[1], maxes[1])[:, None] * np.array([0, 1, 0])[None, :]
    cr = np.arange(mins[2], maxes[2])[:, None] * np.array([0, 0, 1])[None, :]

    all_points = ar[:, None, None] + br[None, :, None] + cr[None, None, :]
    all_points = all_points.reshape((-1, 3))

    frac_points = np.dot(all_points, np.linalg.inv(supercell_matrix))

    tvects = frac_points[np.all(frac_points < 1 - 1e-10, axis=1) & np.all(frac_points >= -1e-10, axis=1)]
    assert len(tvects) == round(abs(np.linalg.det(supercell_matrix)))
    return tvects

def build_nsupercell(a:Atoms,P):
    '''
    通过给定的矩阵P构建表面结构,可以构建一些根号超胞
    '''
    a=a.copy()
    P=np.array(P)
    a.wrap()
    cell = a.get_cell().array
    ncell=np.dot(P,cell)
    eles=a.get_atomic_numbers()
    pos=a.get_positions()
    allvec=lattice_points_in_supercell(P)
    allvec0=allvec.dot(ncell)
    npos=np.zeros((len(allvec)*len(pos),3))
    neles=np.zeros(len(allvec)*len(pos),dtype=np.int32)
    # print(allvec[1])
    for i in range(len(allvec)):
        for j in range(len(pos)):
            npos[i*len(pos)+j]=allvec0[i]+pos[j]
            neles[i*len(pos)+j]=eles[j]
    a_new=Atoms(cell=ncell,pbc=True,positions=npos,numbers=neles)
    a_new.wrap()
    return a_new

def build_nsupercel_(a:Atoms,P):
    '''
    通过给定的矩阵P构建表面结构,可以构建一些根号超胞
    '''
    a=a.copy()
    P=np.array(P)
    a.wrap()
    cell = a.get_cell().array
    ncell=np.dot(P,cell)
    scaled = a.get_scaled_positions()
    eles = a.get_atomic_numbers()
    vecrange=np.sum(np.abs(P),axis=0)+4
    vecrange=[int(i) for i in vecrange]
    # vecrange[2]=1
    #a_new=Atoms(cell=ncell,pbc=True)

    Ptr=np.linalg.inv(P)
    npos=[]
    neles=[]
    #print(np.dot(ncell[0],ncell[1]))
    for i in range(-vecrange[0],vecrange[0]+1):
        for j in range(-vecrange[1],vecrange[1]+1):
            for k in range(-vecrange[2],vecrange[2]+1):
                scaled_new = scaled + (np.array([i,j,k]))
                scaled_new = np.dot(scaled_new, Ptr)
                for pi in range(len(eles)):
                    nowpos=scaled_new[pi]
                    nowposc=nowpos.dot(ncell)
                    if np.all(np.logical_and(nowpos<=1.5,nowpos>=-0.7)):
                        # if not np.all(np.logical_and(nowpos<1,nowpos>0)):
                        #     print(nowpos)
                        if len(npos)>0:# and (np.min(np.abs(nowpos))<0.1 or np.min(np.abs(nowpos-1))<0.1):
                            _,D_l = get_distances(npos,[nowposc],cell=ncell,pbc=[True,True,True])
                            if np.min(D_l) < 0.5:
                                #print(nowpos)
                                continue
                        npos.append(nowposc)
                        neles.append(eles[pi])
                # wheres =(np.array([0,1,2],dtype=np.int32),) 
                # #wheres=np.where(np.logical_and(np.min(scaled_new,axis=1)>=-0.1,np.max(scaled_new,axis=1)<=1))
                
                # if a_new.get_number_of_atoms()==0:
                #         newwheres=wheres
                # else:
                #     newwheres=[]
                #     for i in wheres[0]:
                #         D,D_l=get_distances(a_new.positions,[ccell.cartesian_positions(scaled_new[i])],cell=ccell,pbc=a_new.get_pbc())
                #         if np.min(D_l)<0.0001:
                #             # print('Warning: the atoms are too close')
                #             # print(i)
                #             pass
                #         else:
                #             newwheres.append(i)
                #     newwheres=(np.array(newwheres,dtype=np.int32),)
                #     wheres=newwheres

                # scaled_new=scaled_new[wheres]
                # eles_new = eles[wheres]
                # a_new.extend(Atoms(cell=ncell, scaled_positions=scaled_new, numbers=eles_new, pbc=True))
    a_new=Atoms(cell=ncell,pbc=True,positions=npos,numbers=neles)
    # aa=a.get_atomic_numbers()
    # bb=a_new.get_atomic_numbers()
    # print(len(aa[aa==42])/len(aa[aa==16]),len(bb[bb==42])/len(bb[bb==16]),P)
    # cc=a_new.get_all_distances(mic=True)
    # cc+=np.eye(cc.shape[0])
    # print(np.min(cc))
    #print(a_new.get_number_of_atoms())
    a_new.wrap()
    # a_new.get_atomic_numbers()
    eles=a.get_atomic_numbers()
    statisunit=np.zeros((300,),dtype=np.int32)
    for i in range(a_new.get_number_of_atoms()):
        statisunit[a_new.get_atomic_numbers()[i]]+=1
    eles_new=a_new.get_atomic_numbers()
    statisnew=np.zeros((300,),dtype=np.int32)
    for i in range(a_new.get_number_of_atoms()):
        statisnew[eles_new[i]]+=1
    # compare there weight
    statisnew=statisnew[statisnew>0]
    statisunit=statisunit[statisunit>0]
    pro=statisnew/statisunit
    for i in range(len(pro)):
        if not np.isclose(pro[i],pro[0]):
            print("##############################################################")
            print(pro[i],pro[0])
            print(i,0)
            print("超胞生成错误")
            os._exit(0)
    return a_new
    

from ase.calculators import lammpsrun
from scipy import optimize
#set calculator
# calc=lammpsrun.LAMMPS(command="/home/zln/lammps-stable_29Sep2021_update2/build/lmp",
#     parameters={'units':'metal','pair_style':'lj/cut 11','pair_coeff':['* * 10 3.18 12']},
# )

def build_moire_patterninformation(a:Atoms,b:Atoms,theta,newvec,layer_thickness,shift=[0,0],relax_shift=True,innewB=False,vacuum_thickness=20):
    
    if np.linalg.det(newvec)<0:
        newvec=np.array([newvec[1],newvec[0]])

    shift=np.array(shift)
    cell1=a.get_cell().array
    cell2=b.get_cell().array
    A=cell1[:2,:2]
    B=cell2[:2,:2]
    R=get_rotation_matrix2d(theta)
    R3=get_rotation_matrix(theta)
    B=np.dot(R,B.T).T


    nA=np.dot(newvec,A)
    newcm=np.dot(newvec,np.dot(A,np.linalg.inv(B)))
    nB=np.dot(np.round(newcm),B)
    ## 为了减少失配度度，得到一个新的单位cell
    nD = (nA+nB)/2
    layer_thickness1 = np.max(a.positions[:,2])- np.min(a.positions[:,2])
    layer_thickness2 = np.max(b.positions[:,2])- np.min(b.positions[:,2])

    return {
        "discraption":"""
        Amn:表示下层超晶格与原胞的转换矩阵
        Bmn:表示上层超晶格与原胞的转换矩阵
        sA:表示下层的晶格矢量,每行一个
        sB:表示上层的晶格矢量,每行一个
        layer_thickness:表示下层和上层的原胞的厚度
        newlattice:表示下层和上层的超晶格
        split_height:表示下层和上层的原胞的分割高度
        """,
        'Amn':nA.tolist(),
        'Bmn':newcm.tolist(),
        'sA':nA.tolist(),
        'sB':nB.tolist(),
        'A':A.tolist(),
        'B':B.tolist(),
        'R':R.tolist(),
        'split_height':1+layer_thickness1+layer_thickness1/2,
        'layer_thickness':layer_thickness,
        'layer_thickness1':layer_thickness1,
        'layer_thickness2':layer_thickness2,
        'newlattice':nD.tolist(),
        
    }

def build_moire_pattern(a:Atoms,b:Atoms,theta,newvec,layer_thickness,shift=[0,0],relax_shift=True,innewB=False,vacuum_thickness=20):
    
    if np.linalg.det(newvec)<0:
        newvec=np.array([newvec[1],newvec[0]])

    shift=np.array(shift)
    cell1=a.get_cell().array
    cell2=b.get_cell().array
    A=cell1[:2,:2]
    B=cell2[:2,:2]
    R=get_rotation_matrix2d(theta)
    R3=get_rotation_matrix(theta)
    B=np.dot(R,B.T).T


    nA=np.dot(newvec,A)
    newcm=np.dot(newvec,np.dot(A,np.linalg.inv(B)))
    nB=np.dot(np.round(newcm),B)

    ## 为了减少失配度度，得到一个新的单位cell
    nD = (nA+nB)/2
    
    transvec1 = np.dot(np.linalg.inv(nA),nD)
    transvec2 = np.dot(np.linalg.inv(nB),nD)

    transvec1 = np.array([[transvec1[0,0],transvec1[0,1],0],[transvec1[1,0],transvec1[1,1],0],[0,0,1]])
    transvec2 = np.array([[transvec2[0,0],transvec2[0,1],0],[transvec2[1,0],transvec2[1,1],0],[0,0,1]])

    # tranvec=np.dot(np.linalg.inv(nB),nA) # 将转动后B胞中的坐标变到A胞中（用于处理晶格失配）
    # tranvec=np.array([[tranvec[0,0],tranvec[0,1],0],[tranvec[1,0],tranvec[1,1],0],[0,0,1]])

    layer_thickness1 = np.max(a.positions[:,2])- np.min(a.positions[:,2])
    layer_thickness2 = np.max(b.positions[:,2])- np.min(b.positions[:,2])
    newzlength = layer_thickness1+layer_thickness2+layer_thickness+vacuum_thickness #20位

    moiremodel=Atoms(cell=[[nD[0][0],nD[0][1],0],[nD[1][0],nD[1][1],0],[0,0,newzlength]],pbc=True)

    sa=build_nsupercell(a,[[newvec[0][0],newvec[0][1],0],[newvec[1][0],newvec[1][1],0],[0,0,1]])
    newcm=np.round(newcm)
    sb=build_nsupercell(b,[[newcm[0][0],newcm[0][1],0],[newcm[1][0],newcm[1][1],0],[0,0,1]])
    
    if (sa.get_global_number_of_atoms()!=sb.get_global_number_of_atoms()):
        print('Warning: the number of atoms in two supercells are not equal',sa.get_chemical_formula(),sb.get_chemical_formula())
    sb.positions=np.dot(R3,sb.positions.T).T
    ncell2=np.dot(R3,sb.get_cell().array.T).T
    sb.set_cell(ncell2)
    sazmin=np.min(sa.get_positions()[:,2])
    sa.positions[:,2]+=(1-sazmin) # 将sa移动到高1埃的位置
    sa.positions=np.dot(sa.positions,transvec1) # 将转动后A胞中的坐标变到moire胞中
    sazmax=np.max(sa.get_positions()[:,2])
    
    moiremodel.extend(sa)

    
    if relax_shift:

        """优化层间相对位置，采用简单的lj势能"""
        def shift_energy(shift1):
            #print(shift1)
            sb1=sb.copy()
            moiremodeltmp=moiremodel.copy()
            shift1=np.dot(shift1,nB)
            sb1.positions=np.dot(sb1.positions,transvec2)
            sb1zmin=np.min(sb1.get_positions()[:,2])
            shift1=np.array([shift1[0],shift1[1],-sb1zmin+sazmax+layer_thickness])
            sb1.positions+=np.broadcast_to(shift1,sb1.positions.shape)
            moiremodeltmp.extend(sb1)
            moiremodeltmp.wrap()
            moiremodeltmp=supercells.make_supercell(moiremodeltmp,np.array([[3,0,0],[0,3,0],[0,0,1]])) #扩胞，防止错误
            moiremodeltmp=sort(moiremodeltmp)
            moiremodeltmp.set_calculator(calc)
            return moiremodeltmp.get_potential_energy()

        rs=optimize.minimize(shift_energy,np.array([0.2,0.3]),bounds=[(0,1),(0,1)],method='Nelder-Mead')
        print(rs.x,rs)
        shift=rs.x
    if innewB or relax_shift:
        shift=np.dot(shift,nB)
    else:
        shift=np.dot(shift,B)
    sb.positions=np.dot(sb.positions,transvec2)
    sbzmin=np.min(sb.get_positions()[:,2])
    shift=np.array([shift[0],shift[1],-sbzmin+sazmax+layer_thickness])
    sb.positions+=np.broadcast_to(shift,sb.positions.shape)
    moiremodel.extend(sb)


    moiremodel=sort(moiremodel)
    #print(moiremodel.get_chemical_symbols())
    return moiremodel


def main():

    import argparse
    parser = argparse.ArgumentParser(description='generate a structure')
    parser.add_argument("files", type=str, default=None,nargs=2, help='file name')
    parser.add_argument('-o', '--output', type=str, default='tmpoutput', help='output dir')
    parser.add_argument("-r","--range",type=float,default=[0,180,1000],nargs=3,help="theta range")
    parser.add_argument("-e","--epsilon",type=float,default=0.04,help="epsilon")
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

    thetalist=np.linspace(args.range[0]*np.pi/180,args.range[1]*np.pi/180,int(args.range[2])) #搜索的角度范围
    a.dtheta=thetalist[1]-thetalist[0] #搜索的角度步长

    kexing=[]
    outs=[]
    pltdata=[]
    for theta in thetalist:   
        try :
            a.changetheta(theta)
            mn,area=a.getminmnplot()
            if mn.shape==(2,2):
                res=a.relaxwithmn(mn)
                outs.append((res.x,area,mn))
                pltdata.append([res.x,area])
                if res.success:
                    print(theta)
                    kexing.append((res.x,res.fun,mn))
        except BaseException as e:
            #print(e)
            continue     
    print(kexing)
    kk=0
    informations={}
    for k in kexing:
        print(kk)
        kk+=1
        try :
            m=build_moire_pattern(b,b00,k[0],k[2],args.distance,relax_shift=args.needshift)
            filename="POSCAR-{k1:.3f}-{rrr:.2f}.vasp".format(k1=k[0]*180/np.pi,rrr=k[1]*100)
            informations0=build_moire_patterninformation(b,b00,k[0],k[2],args.distance,relax_shift=args.needshift)
            informations0['theta']=k[0]
            informations0['epsilon']=k[1]
            informations[filename]=informations0
            

            m.write(os.path.join(args.output,filename),format="vasp",direct=True,wrap=True)
        except BaseException as e:
            print(e)
            continue

    import json

    if os.path.exists(os.path.join(args.output,"informations.json")):
        with open(os.path.join(args.output,"informations.json"),'r') as f:
            informationsold=json.load(f)
            informations1=informationsold.update(informations)
            informations=informationsold
    with open(os.path.join(args.output,"informations.json"),'w') as f:
        json.dump(informations,f)
    # aa=(build_nsupercell(a,np.array([[2,1,0],[0,1,0],[0,0,1]])))
    # aa.write("POSCAR.vasp")

if __name__ == '__main__':
    main()
    