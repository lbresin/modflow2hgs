#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:42:40 2017

@author: Lysander Bresinsky, University of Göttingen (Applied Geology)
"""
import flopy
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import label
import scipy as sp
import os
import pandas as pd
import time

def load_modflowmodel(modelprefix):
    '''
    Loads all modflow files via flopy
    '''
    
    # Assign all Modflow packages to variables
    ml = flopy.modflow.Modflow() 
    dis = flopy.modflow.mfdis.ModflowDis.load(modelprefix+'.dis', ml) 
    bas = flopy.modflow.mfbas.ModflowBas.load(modelprefix+'.ba6', ml)
    lpf = flopy.modflow.mflpf.ModflowLpf.load(modelprefix+'.lpf', ml)
    rch = flopy.modflow.mfrch.ModflowRch.load(modelprefix+'.rch',ml)
    wel = flopy.modflow.mfwel.ModflowWel.load(modelprefix+'.wel',ml)
    drn = flopy.modflow.mfdrn.ModflowDrn.load(modelprefix+'.drn',ml)
    
    return ml, dis, bas, lpf, rch, wel, drn

def model_dimensions(dis):
    '''
    Get the model dimensions from the dis package
    '''
    
    nrow=dis.nrow #number of rows (block centered)
    ncol=dis.ncol #number of columns
    ncolc=ncol+1 #number of columns (node centered)
    nrowc=nrow+1 #number of rows
    nlay=dis.nlay
         
    return nrow, ncol, nrowc, ncolc, nlay

def units(ml, dis):
    '''
    Get the model units used in Modflow
    '''
    
    time=dis.itmuni_dict[dis.itmuni][0:-1]
    weight='kilogram'
    length_dict={0 : 'undefined', 1 : 'feet', 2 : 'metre', 3 : 'centimetre'}
    length=length_dict[dis.lenuni]
    
    return weight, length, time
    
def _dir_check(): 
    '''
    Check for export folder "hgs_files"
    '''

    if os.path.isdir('./hgs_files')==1:
        pass
    elif os.path.isdir('./hgs_files')==0:
        os.mkdir('./hgs_files')
    if os.path.isdir('./hgs_files/rch')==1:
        pass
    elif os.path.isdir('./hgs_files/rch')==0:
        os.mkdir('./hgs_files/rch')
        
def labelPix(pix):
    lbl=[]
    for i in range(0,len(pix)):
        height, width, _ = np.shape(pix[i])
        pixRows = np.reshape(pix[i], (height * width, 4))
        unique, counts = np.unique(pixRows, return_counts = True, axis = 0)

        unique = [list(elem) for elem in unique]

        labeledPix = np.zeros((height, width), dtype = int)
        offset = 0
        for index, zoneArray in enumerate(unique):
            index += offset
            zone = list(zoneArray)
            zoneArea = (pix[i] == zone).all(-1)
            elementsArray, numElements = label(zoneArea)
            
            elementsArray[elementsArray!=0] += offset
            
            labeledPix[elementsArray!=0] = elementsArray[elementsArray!=0]
            
            offset += numElements
        
        if i ==0:
            lbl.append(labeledPix)
        elif i>0:
            lbl.append(labeledPix+np.max(lbl[i-1]))
    lblmax=np.max(lbl)
    
    return lbl, lblmax

# exporting single packages - functions
def export_grid(dis):
    '''
    This creates a file '2d_layer_grid.in' of the horizontal 2d grid-slice and a 'zX.in' file of each layer(incl. baselayer). The z-file consists of a list height values for each node

    This file can be loaded in HGS via:
        generate variable rectangles
            include data/2d_layer_grid.in
        
        generate layers interactive
            zone by layer

            base elevation
                elevation from gms file
                    data/z1.in 
            end ! base elevation

         new layer
            layer name
                layer_1
            minimum layer thickness
                0.0 
            elevation from gms file
                data/z2.in
        end ! new layer
        end ! layers interactive
        end ! grid generation
    '''

    #model geometry size
    [nrow, ncol, nrowc, ncolc, nlay]=model_dimensions(dis)

    #xy-coordinates (block-centered)
    y=np.array(dis.get_node_coordinates()[0]) # ([0]: y-coordinates; [1]: x-coordinates]])
    x=np.array(dis.get_node_coordinates()[1])
    # Create coorindates of all nodes (block-centered)
    XYgrid=np.array([np.meshgrid(x,y)[0].ravel(), np.meshgrid(x,y)[1].ravel()]).T # meshgrid creates the coordinates of all existing nodes
    
    #xy-coordinates (node centered)
    xc=flopy.utils.reference.SpatialReference.get_xedge_array(dis)
    yc=flopy.utils.reference.SpatialReference.get_yedge_array(dis)
    # Create coorindates of all nodes (block-centered)
    XYCgrid=np.array([np.meshgrid(xc,yc)[0].ravel(), np.meshgrid(xc,yc)[1].ravel()]).T # meshgrid creates the coordinates of all existing nodes
    
    #z-coordinates (block-centered)
    z=[]
    for i in reversed(range(0,nlay)):        #flip layer numbering 0(lowest) to n+1 (highest)
        z.append(dis.getbotm()[i])           #z-value of layer bottom
    z.append(dis.gettop())                #z-value of layer top
    
    # filter out unusual high or low botm layers
    for i in range(0,len(z)-1):
        z[i][np.where(z[i]>z[3])]=z[3][np.where(z[i]>z[3])]
        #z[i+1][np.where(z[i+1]<z[i])]=z[i][np.where(z[i+1]<z[i])]

    
    #z-coordninates (node-centered)
    zc=[]
    for i in range(0,nlay):
        zc.append(griddata(XYgrid,z[i].ravel(),XYCgrid, method='cubic',fill_value=666).reshape(nrowc, ncolc)) # calculated via cubic interpolation
    zc.append(griddata(XYgrid,z[-1].ravel(),XYCgrid, method='cubic',fill_value=666).reshape(nrowc, ncolc))

    #flip along x-axis and ravel
    zcr=[] # ravel height
    for i in range(0,nlay+1):
        zc[i]=np.flipud(zc[i])
        zcr.append(zc[i].ravel())
    
    #sort yc xc ascending
    xc=np.sort(xc)
    yc=np.sort(yc)
    
   
    #Write Model Information to files
    _dir_check()
    file=open('hgs_files/2d_layer_grid.in', 'w')
    file.write(str(ncolc)+'  '+'!xi coordinates \n')
    for i in range(1,len(xc)+1):
        if i%3==0:
            file.write(str(xc[i-1])+'\n')
        else:
            file.write(str(xc[i-1])+' ')
    file.write('\n'+str(nrowc)+'  '+'!yi coordinates \n')
    for i in range(1,len(yc)+1):
        if i%3==0:
            file.write(str(yc[i-1])+'\n')
        else:
            file.write(str(yc[i-1])+' ')
    file.close
    #z-values(height)
    for i in range(0, len(zcr)):
        file=open('hgs_files/z'+str(i+1)+'.in', 'w')
        file.write('DATASET \n')
        file.write('ND '+ str(nrowc*ncolc)+'\n')
        file.write('TS 0.0 \n')
        for q in range(0,ncolc*nrowc):
            file.write(str(zcr[i][q])+'\n')
        file.close
        
    print('-----------------')
    print('\nSucessfully exported Geometry Files!\n')
    print('Total layers:'+str(nlay)+'\n')

    
def export_ibound(dis, bas, nsublayer):
    '''
    This creates a file 'ibound.in'. 
    'nsublayer' contains the desired sublayers of each existing layer as an array (if none type 1; eg. [1 1 1] for a three layer system). 
    The exported ibound file consists of a list of element numbers, one entry per line.:
        N (number of inactive cell)
        ..

    This file can be loaded in HGS via:
        Choose elements list
            fname.in
        make elements inactive
    '''
    
    # Model geometry
    [nrow, ncol, nrowc, ncolc, nlay]=model_dimensions(dis)
    
    # Get ibound array
    ibound=bas.ibound.array 
           
    # Flip layer numbering 
    ibound=np.flipud(ibound)    # layer numbering 0(lowest) to n+1 (highest)
    
    # Flip along y-axis
    for i in range(0,len(ibound)):    
        ibound[i]=np.flipud(ibound[i])
    
    # Gather element number of inactive cell
    iboundnel=[]  # ibound element number acc. to HGS
    previous_elements=0     # total element number of the subjacent layers
    for i in range(0,len(ibound)):
        ibound_temp=np.where(ibound[i]==0) # m n number of matrix
        for q in range(nsublayer[i]):
            ibound_temp2=(ibound_temp[0]*ncol)+(ibound_temp[1]+1)+previous_elements
            iboundnel.append(ibound_temp2)
            previous_elements+=nrow*ncol
            
    # Write Model Information to files
    _dir_check()
    file=open('hgs_files/ibound.in', 'w')
    for q in range(0,len(iboundnel)):
        for i in iboundnel[q]:  
            file.write(str(i)+'\n')
    file.close
    
    # Print out Information
    print('-----------------')
    print('\nSucessfully exported Ibound File!\n')
    print('Total layers:'+str(nlay)+'\n')
    print('Total sublayers:'+str(np.sum(nsublayer))+'\n')
    print('Total elements: '+str(previous_elements)+'\n')
    
def export_mprops(dis, lpf):
    '''
    This creates a file for the (1) hydraulic conductivity, (2) specific storage and (3) specific yield of each layer. The file consists of a list of element numbers, one entry per line.:
        (1) hydraulic conductivity
            N kxx kyy kzz
            ..
        
        (2) specific storage
            N ss
            ..
        
        (3) specific yield
            N sy
            ..

    This file can be loaded in HGS via:
        read elemental k from file
            data/hydraulic_conductivity1.in
        
        read elemental specific storage from file
            data/specific_storage1.in
            
        read elemental porosity from file
            data/specific_yield1.in
    '''
    #-----------
    # Model Information
    #-----------   
    #model geometry size
    [nrow, ncol, nrowc, ncolc, nlay]=model_dimensions(dis)
    
    #model properties
    hk=np.flipud(lpf.hk.array) # horizontal hydraulic conductivity
    vk=np.flipud(lpf.vka.array) # vertical hydraulic conductivity
    ss=np.flipud(lpf.ss.array) # specific storage
    sy=np.flipud(lpf.sy.array) # specific yield
    

    #-----------
    # transform 'hk, vk, ss, sy'
    #----------- 
    #flip along x-axis
    mprops=[]
    lbl=[]
    for i in range(0,len(hk)): #flip
        hk[i]=np.flipud(hk[i])
        vk[i]=np.flipud(vk[i])
        ss[i]=np.flipud(ss[i])
        sy[i]=np.flipud(sy[i])
    
    #ravel property array
    hkr=[]
    vkr=[]
    ssr=[]
    syr=[]
    for i in range(0,len(hk)):
        hkr.append(hk[i].ravel())
        vkr.append(vk[i].ravel())
        ssr.append(ss[i].ravel())
        syr.append(sy[i].ravel())

    # assign zone labels (of same mprops)   
    for i in range(0,len(hk)): #label
        mprops.append(np.zeros((np.shape(hk)[1],np.shape(hk)[2], 4))) 
        mprops[i][:,:,0]=hk[i]
        mprops[i][:,:,1]=vk[i]
        mprops[i][:,:,2]=ss[i]
        mprops[i][:,:,3]=sy[i]
    [lbl, lblmax]=labelPix(mprops)
    
    # ravel labels
    lblr=[]
    for i in range(0, len(hk)):
        lblr.append(lbl[i].ravel())
        
    #-----------
    #Write Model Information to files
    #-----------
    _dir_check()
    
    # hydraulic conductivity (n kxx kyy kzz)
    file=open('hgs_files/hydraulic_conductivity.in', 'w')
    for i in range(0, len(hkr)):
        for q in range(0,len(hkr[i])):
            file.write(str((q+1)+(i*nrow*ncol))+' '+
                           str(hkr[i][q])+' '+
                           str(hkr[i][q])+' '+
                           str(vkr[i][q])+'\n')
    file.close

    # specific storage (n ssr)
    file=open('hgs_files/specific_storage.in', 'w')
    for i in range(0, len(hkr)):    
        for q in range(0,len(hkr[i])):
            file.write(str((q+1)+(i*nrow*ncol))+' '+
                           str(ssr[i][q])+'\n')
    file.close    
    
    # specific yield (n syr)
    file=open('hgs_files/specific_yield.in', 'w')
    for i in range(0, len(hkr)):
        for q in range(0,len(hkr[i])):
            file.write(str((q+1)+(i*nrow*ncol))+' '+
                           str(syr[i][q])+'\n')
    file.close
    
    # zone file
    file=open('hgs_files/zones.in', 'w')
    for i in range(0, len(hkr)):    
        for q in range(0,len(hkr[i])):
            file.write(str((q+1)+(i*nrow*ncol))+' '+
                           str(lblr[i][q])+'\n')
    file.close  
    return lbl, lblmax

def export_recharge(dis, rch):
    '''
    This creates a recharge file for every stress period. The file consists of a list of recharge flux values for every node, one entry per line.:
        q(N(i))
        q(N(i+1))
        ..

    This file can be loaded in HGS via:
        ????
        boundary condition type flux

        face 
            set top
        
        time file table
            bc_time(i), bc_file(i)
        
        end ! new specified head


        ????
    '''
    #-----------
    # Model Information
    #-----------   
    #model geometry size
    [nrow, ncol, nrowc, ncolc, nlay]=model_dimensions(dis)

    # model properties    
    rc=[]
    for i in range(0, len(rch.rech.array)):
        rc.append(rch.rech.array[i]) # recharge per stress period per cell
    
    #flip
    for i in range(0, len(rc)):
        rc[i]=np.flipud(rc[i])
    
    #ravel 
    rcr=[]
    for i in range(0, len(rc)):
        rcr.append(rc[i].ravel(i))
    
    # calcuate absolute time of each step
    t=[]
    perlen=dis.perlen.array
    for i in range(0,len(perlen)):
        t.append(np.sum(perlen[0:i]))

    #-----------
    #Write Model Information to files
    #-----------
    _dir_check()
    #write general rch input file
    file=open('hgs_files/recharge.in', 'w')
    for i in range(0,len(t)):
        file.write(str(t[i])+' '+'hgs_files/rch/recharge_t'+str(i+1)+'.in \n')
    file.close
    #write recharge data files for each stress period
    for i in range(0,len(t)):
        file=open('hgs_files/rch/recharge_t'+str(i+1)+'.in', 'w')
        file.write(str(len(rcr[i]))+'\n')
        for q in range(0,len(rcr[i])):
            file.write(str(rcr[i][q])+'\n')
        file.close
        
def export_well(dis, wel):
    #-----------
    # Model Information
    #-----------   
    [nrow, ncol, nrowc, ncolc, nlay]=model_dimensions(dis)
    
    #xy-coordinates (node-centered)
    x=flopy.utils.reference.SpatialReference.get_xedge_array(dis)
    y=np.sort(flopy.utils.reference.SpatialReference.get_yedge_array(dis))
    z=[] #z-coordinates (block-centered)
    for i in reversed(range(0,nlay)):        #flip layer numbering 0(lowest) to n+1 (highest)
        z.append(np.flipud(dis.getbotm()[i]))           #z-value of layer bottom
    z.append(np.flipud(dis.gettop()))                #z-value of layer top
    # filter out unusual high or low botm layers
    for i in range(0,len(z)-1):
        z[i][np.where(z[i]>z[3])]=z[3][np.where(z[i]>z[3])]
    
    #Time data
    t=[] # time
    perlen=dis.perlen.array
    for i in range(0,len(perlen)):
        t.append(np.sum(perlen[0:i]))
    
    # WEL-file data
    qfact=wel.stress_period_data.array['qfact'] # Qfact—is the factor used to calculate well recharge rate from the parameter value. The recharge rate is the product of Qfact and the parameter value.
    flux=wel.stress_period_data.array['flux'] # The Well package is used to simulate a specified flux to individual cells and specified in units of length^3/time.
     
    # flip layers
    for i in range(0, len(qfact)):
        qfact[i]=np.flipud(qfact[i])
        flux[i]=np.flipud(flux[i])
        for q in range(0, len(qfact[i])):
            qfact[i][q]=np.flipud(qfact[i][q])
            flux[i][q]=np.flipud(flux[i][q])
            
    # Coordinates of well (in node number)
    well_coordinates=np.where(qfact==1)[-3:]
    well_coordinates=pd.DataFrame({'nlay':well_coordinates[0], 'y':well_coordinates[1], 'x':well_coordinates[2]}) #drop duplicates with pandas
    well_coordinates=well_coordinates.drop_duplicates()
    well_coordinates=well_coordinates.values.T

    # calulate HGS Well node number
    #well_nodes=well_coordinates[0,:]*nrowc*ncolc*2+well_coordinates[2,:]*ncolc+well_coordinates[1,:]
    
    # Coordinates of well (in [m, km])
    x_well=[]
    y_well=[]
    z_well_bttm=[]
    z_well_top=[]
    for i in range(0, len(well_coordinates[2,:])):
        x_well.append(x[well_coordinates[1,i]])
        y_well.append(y[well_coordinates[2,i]])
        z_well_bttm.append(z[well_coordinates[0,i]][well_coordinates[2,i],well_coordinates[1,i]])
    well_node_coordinates=[x_well, y_well, z_well_bttm]
    
    # Flux per stress period
    well_flux=np.ones([np.shape(well_coordinates)[1], len(qfact)])
    for q in range(0,len(qfact)):     
        for i in range(0,np.shape(well_coordinates)[1]):
            well_flux[i,q]=flux[q,well_coordinates[0,i],well_coordinates[2,i],well_coordinates[1,i]]
    
    #-----------
    #Write Model Information to files
    #-----------
    _dir_check()
    file=open('hgs_files/well_bc.in', 'w')        
    for i in range(0,np.shape(well_node_coordinates)[1]):    
        file.write('!---- well ' + str(i+1)+'\n')
        file.write('clear chosen nodes \n') 
        file.write('choose node \n')
        file.write('  '+str(well_node_coordinates[0][i]))
        file.write('  '+str(well_node_coordinates[1][i]))
        file.write('  '+str(well_node_coordinates[2][i])+'\n\n')
        file.write('create node set \n')
        file.write('  well'+str(i+1)+'\n \n')
        file.write('boundary condition \n')
        file.write('  type\n')
        file.write('  flux nodal\n\n')
        file.write('  node set \n'+'  well'+str(i+1)+'\n\n')
        file.write('  time value table \n')
        for q in range(0, len(well_flux[i,:])):
            file.write('    '+str(t[q])+' '+str(well_flux[9,q])+'\n')
        file.write('  end \n')
        file.write('end \n')
    file.close  

def export_drain(dis, drn):      
    #-----------
    # Model Information
    #-----------   
    [nrow, ncol, nrowc, ncolc, nlay]=model_dimensions(dis)

    #xy-coordinates (node-centered)
    x=flopy.utils.reference.SpatialReference.get_xedge_array(dis)
    y=np.sort(flopy.utils.reference.SpatialReference.get_yedge_array(dis))
    z=[] #z-coordinates (block-centered)
    for i in reversed(range(0,nlay)):        #flip layer numbering 0(lowest) to n+1 (highest)
        z.append(np.flipud(dis.getbotm()[i]))           #z-value of layer bottom
    z.append(np.flipud(dis.gettop()))                #z-value of layer top
    # filter out unusual high or low botm layers
    for i in range(0,len(z)-1):
        z[i][np.where(z[i]>z[3])]=z[3][np.where(z[i]>z[3])]
    
        #Time data
    t=[] # time
    perlen=dis.perlen.array
    for i in range(0,len(perlen)):
        t.append(np.sum(perlen[0:i]))
    
    #drain data
    drain_data=drn.stress_period_data.get_dataframe()
    
    #flip layer and rows
    drain_data['k']=nlay-1-drain_data['k']
    drain_data['i']=nrow-drain_data['i']-1

    #xy coordinates [m,km]
    x_drn=[]
    y_drn=[]
    z_drn=[]
    for i in range(len(drain_data['j'])):
        x_drn.append(x[drain_data['j'][i]])
        y_drn.append(y[drain_data['i'][i]])
        z_drn.append(z[drain_data['k'][i]][drain_data['i'][i],drain_data['j'][i]])
    drain_data['x']=x_drn
    drain_data['y']=y_drn
    drain_data['z']=z_drn
    
    #-----------
    #Write Model Information to files
    #-----------
    _dir_check()
    file=open('hgs_files/drain_bc.in', 'w')        
    for i in range(len(drain_data)):    
        file.write('!---- drain ' + str(i+1)+'\n')
        file.write('clear chosen nodes \n') 
        file.write('choose node \n')
        file.write('  '+str(drain_data['x'][i]))
        file.write('  '+str(drain_data['y'][i]))
        file.write('  '+str(drain_data['z'][i])+'\n\n')
        file.write('create node set \n')
        file.write('  drain'+str(i+1)+'\n \n')
        file.write('boundary condition \n')
        file.write('  type\n')
        file.write('  simple drain\n\n')
        file.write('  node set \n'+'  drain'+str(i+1)+'\n\n')
        file.write('  time value table \n')
        
        q=0
        hand=0
        while hand!=1:
            if hasattr(drain_data, str('cond'+str(q))):
                file.write('    '+str(t[q])+' '\
                           +str(drain_data['elev'+str(q)][i])+' '\
                           +str(drain_data['cond'+str(q)][i])+'\n')
                q+=1
            else:
                hand=1
        
        file.write('  end \n')
        file.write('end \n\n')
    file.close  
    
    
def export_all(modelprefix):
    #PROGRESS BAR
       
    #load allm
    [ml, dis, bas, lpf, rch, wel, drn]=load_modflowmodel(modelprefix)
    [nrow, ncol, nrowc, ncolc, nlay]=model_dimensions(dis)
    #export all
    export_grid(dis)
    export_ibound(dis,bas)
    export_initialhead(dis,bas)
    lbl, lblmax=export_mprops(dis,lpf)
    export_well(dis, wel, lblmax)
    export_recharge(dis, rch)
    


'''
# unfinished projects

def export_evapotranspiration_segments(modelprefix):
    ml=flopy.modflow.Modflow()
    flopy.modflow.ModflowDis.load(modelprefix+'.dis', ml)
    bcf=flopy.modflow.ModflowBcf.load(modelprefix+'.bcf',ml)
#   ....   

def export_stress(modelprefix):
#   ....

# drain obersavation (.drob file)    
# head observation (.hob file)
'''
