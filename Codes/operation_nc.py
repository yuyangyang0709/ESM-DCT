import sys, getopt, os 
import netCDF4 as nc
import numpy as np 
import math

np.set_printoptions(precision=15)

def read_cam_files(input_file):
    
    is_SE=True

    ncfile = nc.Dataset(input_file,"r")
    
    #获取参数lev
    input_dims = ncfile.dimensions
    nlev=len(input_dims["lev"])
    ncol=len(input_dims["ncol"])
    
    
    #获取2d，3d变量
    input_vars = ncfile.variables
    num_vars = len(input_vars)   

    d2_var_names = []
    d3_var_names = []

    excludevarlist=['EMISCLD','LANDFRAC','PHIS','SOLIN','AEROD_v','AODDUST1','AODDUST3','AODVIS','BURDEN1','BURDEN2','BURDEN3','BURDENBC','BURDENDUST','BURDENPOM','BURDENSEASALT','BURDENSO4','BURDENSOA','LWCF','SWCF','FLDS','FLNS','FLNSC','FLNT','FLNTC','FLUT','FLUTC','FSDS','FSDSC','FSNS','FSNSC','FSNT','FSNTC','FSNTOA','FSNTOAC','FSUTOA','QRL','QRS']
    
    #面积权重
    area_wgt = np.zeros(ncol, dtype=np.float64)
    area=input_vars["area"]
    area_wgt[:] = area[:]
    total = np.sum(area_wgt)
    area_wgt[:] /= total
    
    for k,v in input_vars.items():  
        var = k
        vd = v.dimensions # all the variable's dimensions (names)
        vr = len(v.dimensions) # num dimension
        vs = v.shape # dim values
        is_2d = False
        is_3d = False

        if ((vr == 2) and (vs[1] == ncol)):  
           is_2d = True 
        elif ((vr == 3) and (vs[2] == ncol and vs[1] == nlev )):  
           is_3d = True 
                   
        if ((is_3d == True) and (k not in excludevarlist)) :
            d3_var_names.append(k)
        elif  ((is_2d == True) and (k not in excludevarlist)):    
            d2_var_names.append(k)
    

    
    d2_var_names.sort()
    d3_var_names.sort()
    
    #面积权重平均    
 
    var_vec=[]
    for vcount,vname in enumerate(d3_var_names):    
        data=ncfile.variables[vname]            
        
        area_ave_temp=[]
        for k in range(nlev):
            area_ave1= np.average(data[0,k,:], weights=area_wgt)   #(time,nlev,ncol)
            area_ave_temp.append(area_ave1)
        area_ave=np.average(area_ave_temp)
        var_vec.append(area_ave)
            
    for vcount,vname in enumerate(d2_var_names):    
        data=ncfile.variables[vname]
        area_ave= np.average(data[0,:], weights=area_wgt)   #(time,ncol)      
        var_vec.append(area_ave)
        
    var_vec=np.array(var_vec)

    ncfile.close()
           
    return var_vec
    
def read_pop_files(input_file):
    
    is_SE=True

    ncfile = nc.Dataset(input_file,"r")
    
    #获取参数lev
    input_dims = ncfile.dimensions
    nlev=len(input_dims["z_t"])
    nlon=len(input_dims["nlon"])
    nlat=len(input_dims["nlat"])
    
    
    #面积权重
    input_vars = ncfile.variables
    area_wgt = np.zeros((nlat,nlon), dtype=np.float64)
    area=input_vars["TAREA"]
    area_wgt[:,:] = area[:,:]
    total = np.sum(area_wgt)
    area_wgt[:,:] /= total
    
    t_temp = ncfile.variables["TEMP"]
    s_temp = ncfile.variables["SALT"]
    u_temp = ncfile.variables["UVEL"]
    v_temp = ncfile.variables["VVEL"]
        
    t=np.ma.masked_values(t_temp[0],t_temp._FillValue)        
    s=np.ma.masked_values(s_temp[0],s_temp._FillValue)
    u=np.ma.masked_values(u_temp[0],u_temp._FillValue)
    v=np.ma.masked_values(v_temp[0],v_temp._FillValue)
    
    #面积权重平均    
    
    var_vec=[]
    area_t_temp=[]
    area_s_temp=[]
    area_u_temp=[]
    area_v_temp=[]
    
    for k in range(nlev):
        area_t= np.average(t[k,:,:], weights=area_wgt)   #(time,z_t,nlat,nlon)
        area_t_temp.append(area_t)
        
        area_s= np.average(s[k,:,:], weights=area_wgt)   #(time,z_t,nlat,nlon)
        area_s_temp.append(area_s)
        
        area_u= np.average(u[k,:,:], weights=area_wgt)   #(time,z_t,nlat,nlon)
        area_u_temp.append(area_u)
        
        area_v= np.average(v[k,:,:], weights=area_wgt)   #(time,z_t,nlat,nlon)
        area_v_temp.append(area_v)
        
    area_ave=np.average(area_t_temp)
    var_vec.append(area_ave)
    area_ave=np.average(area_s_temp)
    var_vec.append(area_ave)
    area_ave=np.average(area_u_temp)
    var_vec.append(area_ave)
    area_ave=np.average(area_v_temp)
    var_vec.append(area_ave)
 
    var_vec=np.array(var_vec)

    ncfile.close()
           
    return var_vec
    
def main(argv):
    
    #input_dir_cam='/home/haida_zhangshaoqing/yuyy/MPE_test/cam/6'
    #in_files_temp = os.listdir(input_dir_cam)
    
    #input_dir_pop='/home/haida_zhangshaoqing/yuyy/MPE_test/pop/6'
    #in_files_temp = os.listdir(input_dir_pop)
    
    #input_dir_cam='/home/haida_zhangshaoqing/yuyy/CPE_data/zm_4_rliq/cam'
    #in_files_temp = os.listdir(input_dir_cam)
    
    input_dir_pop='/home/haida_zhangshaoqing/yuyy/CPE_data/zm_4_rliq/pop'
    in_files_temp = os.listdir(input_dir_pop)
    
    in_files=sorted(in_files_temp)
    
    for i in range (0,len(in_files)):
        #input_file_cam = input_dir_cam+'/'+in_files[i]   
        #files= read_cam_files(input_file_cam)     
        
        input_file_pop = input_dir_pop+'/'+in_files[i]    
        files= read_pop_files(input_file_pop)
        
        #with open("/home/haida_zhangshaoqing/yuyy/BiGRU_autoEncoder/data/14/cam/"+str(i+1), "w") as f:
        #with open("/home/haida_zhangshaoqing/yuyy/BiGRU_autoEncoder/data/14/pop/"+str(i+1), "w") as f:
        
        #with open("/home/haida_zhangshaoqing/yuyy/BiGRU_autoEncoder/data_test/zm_4_rliq/cam/"+str(i+1), "w") as f:
        with open("/home/haida_zhangshaoqing/yuyy/BiGRU_autoEncoder/data_test/zm_4_rliq/pop/"+str(i+1), "w") as f:
            
            f.write(str(files)+"\n")
            f.close()
    
if __name__ == "__main__":
    main(sys.argv[1:])