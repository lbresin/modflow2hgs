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
import os
import pandas as pd
import matplotlib
from numpy import ma
from matplotlib import colors, ticker, cm

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
    y=np.array(dis.get_node_coordinates()[0]) 
    x=np.array(dis.get_node_coordinates()[1])
    XYgrid=np.array([np.meshgrid(x,y)[0].ravel(), np.meshgrid(x,y)[1].ravel()]).T # meshgrid creates the coordinates of all existing nodes
    

    #xy-coordinates (node centered)
    xc=flopy.utils.reference.SpatialReference.get_xedge_array(dis)
    yc=flopy.utils.reference.SpatialReference.get_yedge_array(dis)
    XYCgrid=np.array([np.meshgrid(xc,yc)[0].ravel(), np.meshgrid(xc,yc)[1].ravel()]).T 
    

    #z-coordinates (block-centered)
    z=[]
    for i in reversed(range(0,nlay)):        #flip layer numbering 0(lowest) to n+1 (highest)
        z.append(dis.getbotm()[i])           #z-value of layer bottom
    z.append(dis.gettop())                #z-value of layer top
    

    # filter out unusual high or low botm layers
    for i in range(0,len(z)-1):
        z[i][np.where(z[i]>z[3])]=z[3][np.where(z[i]>z[3])]

    
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
    
    # round to 2 digits
    xc=np.round(xc,0)
    yc=np.round(yc,0)
    for i in range(0,nlay+1):
        zcr[i]=np.round(zcr[i],0)
        
    
    #Write Model Information to files
    _dir_check()
    file=open('hgs_files/2d_layer_grid.in', 'w')
    file.write('!xi coordinates \n')
    file.write(str(ncolc)+'\n')
    for i in range(1,len(xc)+1):
        if i%8==0:
            file.write('{0:.0f}'.format(xc[i-1])+'\n')
        else:
            file.write('{0:.0f}'.format(xc[i-1])+' ')
    file.write('\n\n!yi coordinates \n')
    file.write(str(nrowc)+'\n')
    for i in range(1,len(yc)+1):
        if i%8==0:
            file.write('{0:.0f}'.format(yc[i-1])+'\n')
        else:
            file.write('{0:.0f}'.format(yc[i-1])+' ')
    file.close
    for i in range(0, len(zcr)):  #z-values(height)
        file=open('hgs_files/z'+str(i+1)+'.in', 'w')
        file.write('DATASET \n')
        file.write('ND '+ str(nrowc*ncolc)+'\n')
        file.write('TS 0.0 \n')
        for q in range(0,ncolc*nrowc):
            file.write('{0:.0f}'.format(zcr[i][q])+'\n')
        file.close
    
   
    # print result   
    print('-----------------')
    print('\nSucessfully exported Geometry Files!\n')
    print('Total layers:'+str(nlay)+'\n')

    return XYgrid,XYCgrid

def export_new_gridsize(xll=0, yll=0):
    return
def export_ZoneRaster(lbl):
    
    # Model geometry
    [nrow, ncol, nrowc, ncolc, nlay]=model_dimensions(dis)
    if nsublayer==0:
        nsublayer=np.ones([1,nlay])
        
    # xy-coordinates (block-centered)
    y=np.array(dis.get_node_coordinates()[0]) 
    x=np.array(dis.get_node_coordinates()[1])
    XYgrid=np.array([np.meshgrid(x,y)[0].ravel(), np.meshgrid(x,y)[1].ravel()]).T # meshgrid creates the coordinates of all existing nodes
    
    # xy-coordinates (cell-centered)
    yc=np.linspace(np.max(y),0,nrow*3)
    xc=np.linspace(0,np.max(x),ncol*3)
    XYCgrid=np.array([np.meshgrid(xc,yc)[0].ravel(), np.meshgrid(xc,yc)[1].ravel()]).T # meshgrid creates the coordinates of all existing nodes
    
    # flip lbl along y-achis 
    lbl=lbl.reshape(nlay, nrow, ncol)
    lblf=[]
    for i in range(len(lbl)):
        lblf.append(np.flipud(lbl[i]))
    lblf
    
    # transform to fine mesh
    label_fine=[]
    for i in range(len(ibound)):
        label_fine.append(griddata(XYgrid,lblf[i].ravel(),XYCgrid, method='nearest',fill_value=666).reshape(nrow*3, ncol*3)) # calculated via cubic interpolation
    
    # raster data
    ncols=ncol*3
    nrows=nrow*3
    xllcorner=np.min(xc)
    yllcorner=np.min(yc)
    nodata_value=-9999
    cellsize=(exit)*(abs(np.min(yc)-np.max(yc))/nrows)
    
def export_ibound(dis, bas, nsublayer=0):
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
    if nsublayer==0:
        nsublayer=np.ones([1,nlay])
    
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
    
    
    # total ibound cells
    tot_ibound=0
    for i in range(len(iboundnel)): 
        tot_ibound+=np.shape(iboundnel[i])[0]
    
        
    # Write Model Information to files
    _dir_check()
    
    for i in range(nlay):
        if i == 0:
            start=0
            stop=nsublayer[i]
        else:
            start+=nsublayer[i-1]
            stop+=nsublayer[i]
            
        file=open('hgs_files/ibound'+str(i+1)+'.in', 'w')
        for q in range(start,stop):  
            for u in iboundnel[q]:
                file.write(str(u)+'\n')
        file.close
         
    
    # Print out Information
    print('-----------------')
    print('Sucessfully exported Ibound File!\n')
    print('Total layers:'+str(nlay)+'\n')
    print('Total sublayers:'+str(np.sum(nsublayer))+'\n')
    print('Total elements: '+str(tot_ibound)+'\n')
    return ibound

def export_mprops(dis, lpf, ibound, nsublayer=0):
    '''
    This creates a file for the (1) hydraulic conductivity, (2) specific storage and (3) specific yield of each layer. The file consists of a list of element numbers, one entry per line.:
        (1) hydraulic conductivity
            N kxx kyy kzz
            ..
        
        (2) specific storage
            N ss
            ..cd 
        
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
    
    def labelPix(pix,n_val):
        '''
        Pix: Input Array of Mprops
        n_val: Number of Mprops per cell
        
        Returns unique Zone numbers for cells of same material properties
        '''
        height, width, _ = pix.shape
        pixRows = np.reshape(pix, (height * width, n_val))
        unique, count = np.unique(pixRows, return_counts = True, axis = 0)

        lbl = np.zeros((height, width), dtype = int)
    
        index=1
        for zoneArray in unique:
            pos=np.where((pix[:,:,0]==zoneArray[0])&(pix[:,:,1]==zoneArray[1])&(pix[:,:,2]==zoneArray[2]) & (pix[:,:,3]==zoneArray[3]))
            lbl[pos]=index
            index+=1
    
        lblmax=index-1
    
        return lbl,lblmax, unique
    
    #model geometry
    [nrow, ncol, nrowc, ncolc, nlay]=model_dimensions(dis)
    if nsublayer==0:
        nsublayer=np.ones([1,nlay])
    
    #model properties
    hk=lpf.hk.array # horizontal hydraulic conductivity
    vk=lpf.vka.array # vertical hydraulic conductivity
    ss=lpf.ss.array # specific storage
    sy=lpf.sy.array # specific yield
    

    # flip layer numbering
    hk=np.flipud(hk)
    vk=np.flipud(vk)
    ss=np.flipud(ss)
    sy=np.flipud(sy)
    
    
    # flip along x-axis
    for i in range(0,len(hk)): #flip
        hk[i]=np.flipud(hk[i])
        vk[i]=np.flipud(vk[i])
        ss[i]=np.flipud(ss[i])
        sy[i]=np.flipud(sy[i])

    # delete unneccessary information via ibound
    for i in range(0,len(hk)):
        hk[i] = hk[i] * ibound[i]
        vk[i] = vk[i] * ibound[i]
        ss[i] = ss[i] * ibound[i]
        sy[i] = sy[i] * ibound[i]
    
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
    mprops=np.zeros((nlay,nrow*ncol,4))
    for i in range(nlay):
        mprops[i][:,0]=hkr[i]
        mprops[i][:,1]=vkr[i]
        mprops[i][:,2]=ssr[i]
        mprops[i][:,3]=syr[i]
    [lbl, lblmax, zone_mprops]=labelPix(mprops,4)

      
    # Write Model Information to files
    _dir_check()
    previous_elements=0     # total element number of the subjacent layers
    file=open('hgs_files/zones.in', 'w')
    for i in range(len(hkr)):
         for u in range(nsublayer[i]):
            for q in range(len(hkr[i])):
                file.write(str((q+1)+previous_elements)+' '+str(lbl[i][q])+'\n')
            previous_elements+=nrow*ncol
    file.close  
    
    file=open('hgs_files/porous_media.mprops', 'w')
    for i in range(len(zone_mprops)):
        file.write('!------------------------------------------\n')
        file.write('material_zone_'+str(i+1))
        file.write('\n\nk anisotropic\n')
        file.write('  '+'{0:.5e}'.format(zone_mprops[i,1]).replace('e','d')+'  '\
                   +'{0:.5e}'.format(zone_mprops[i,1]).replace('e','d')+'  '\
                   +'{0:.5e}'.format(zone_mprops[i,0]).replace('e','d')+'\n\n')
        file.write('Specific storage\n')
        file.write('  '+'{0:.5e}'.format(zone_mprops[i,2]).replace('e','d')+'\n\n')
        file.write('Porosity\n')
        file.write('  '+'{0:.5e}'.format(zone_mprops[i,3])+'\n\n')
        file.write('end material\n')
    file.close

    file=open('hgs_files/mprops.in', 'w')
    for i in range(len(zone_mprops)):
        file.write('!--------------------------zone'+str(i+1)+'\n\n')
        file.write('clear chosen zones\n')
        file.write('choose zone number\n')
        file.write('  '+str(i+1)+'\n\n')
        file.write('read properties\n')
        file.write('material_zone_'+str(i+1)+'\n')
    file.close
    
    # print result   
    print('-----------------')
    print('Sucessfully exported mprops files!\n')
    print('Total Zones: '+str(lblmax)+'\n')
    
    # plot mprop zones
    y=np.array(dis.get_node_coordinates()[0]) 
    x=np.array(dis.get_node_coordinates()[1])
    X, Y = np.meshgrid(x, y)
    zone_lbl=[]
    for i in range(nlay):
        zone_lbl.append(lbl[i,:].reshape((nrow,ncol)))
        
    for i in range(nlay):
        fig, ax = matplotlib.pyplot.subplots()
        bounds = np.array(range(1,lblmax))
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=250)

        plt = matplotlib.pyplot.pcolor(X, Y, np.flipud(zone_lbl[i]), cmap=cm.jet, norm=norm) 
        cbar = fig.colorbar(plt)
               
        
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

   
    # Write Model Information to files
    _dir_check()
    
    file=open('hgs_files/recharge.in', 'w')
    for i in range(0,len(t)):
        file.write(str(t[i])+' '+'hgs_files/rch/recharge_t'+str(i+1)+'.in \n')
    file.close
    
    for i in range(0,len(t)):
        file=open('hgs_files/rch/recharge_t'+str(i+1)+'.in', 'w')
        file.write(str(len(rcr[i]))+'\n')
        for q in range(0,len(rcr[i])):
            file.write(str(rcr[i][q])+'\n')
        file.close
    

    # print result   
    print('-----------------')
    print('Sucessfully exported recharge files!\n')
    
    
def export_well(dis, wel):
    #-----------
    # Model Information
    #-----------   
    [nrow, ncol, nrowc, ncolc, nlay]=model_dimensions(dis)
    
    #xy-coordinates (node-centered)
    x=flopy.utils.reference.SpatialReference.get_xedge_array(dis)
    y=np.sort(flopy.utils.reference.SpatialReference.get_yedge_array(dis))
    
    #z-coordinates (block-centered)
    z=[] 
    for i in reversed(range(0,nlay)):        #flip layer
        z.append(dis.getbotm()[i])           #z-value of layer bottom
    z.append(dis.gettop())                   #z-value of layer top
    
    # flip z-layer
    for i in range(len(z)):
        z[i]=np.flipud(z[i])
    
    # filter out unusual high or low botm layers
    for i in range(0,len(z)-1):
        z[i][np.where(z[i]>z[3])]=z[3][np.where(z[i]>z[3])]
    
    #Time data
    t=[] # time
    perlen=dis.perlen.array
    for i in range(0,len(perlen)):
        t.append(np.sum(perlen[0:i]))
    
    # WEL-file data
    well_data=wel.stress_period_data.get_dataframe()

    # flip layer and rows
    well_data['k']=nlay-1-well_data['k']
    well_data['i']=nrow-well_data['i']-1
    
    #xy coordinates [m,km]
    x_wel=[]
    y_wel=[]
    z_wel=[]
    for i in range(len(well_data['j'])):
        x_wel.append(x[well_data['j'][i]])
        y_wel.append(y[well_data['i'][i]])
        z_wel.append(z[well_data['k'][i]][well_data['i'][i],well_data['j'][i]])
    well_data['x']=x_wel
    well_data['y']=y_wel
    well_data['z']=z_wel
    
    
    #Write Model Information to files
    _dir_check()
    file=open('hgs_files/well_bc.in', 'w')        
    for i in range(len(well_data)):    
        file.write('!---- well ' + str(i+1)+'\n')
        file.write('clear chosen nodes \n') 
        file.write('choose node \n')
        file.write('  '+str(well_data['x'][i]))
        file.write('  '+str(well_data['y'][i]))
        file.write('  '+str(well_data['z'][i])+'\n\n')
        file.write('create node set \n')
        file.write('  well'+str(i+1)+'\n \n')
        file.write('boundary condition \n')
        file.write('  type\n')
        file.write('  flux nodal\n\n')
        file.write('  node set \n'+'  well'+str(i+1)+'\n\n')
        file.write('  time value table \n')
        Q_temp=0
        q_now=0
        for q in range(len(t)):
            if hasattr(well_data, str('flux'+str(q))):
                q_prev = q_now
                q_now = well_data['flux'+str(q)][i]
                if q_now != q_prev:
                    file.write('  '+str(t[q])+' '\
                           +str(well_data['flux'+str(q)][i])+'\n')
                else:
                    pass
            else:
                pass
        file.write('  end \n')
        file.write('end \n\n')
    file.close 
    
    # print result   
    print('-----------------')
    print('Sucessfully exported well data file!\n')
    print('Total Wells: '+str(np.shape(well_data)[0])+'\n')

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
    
    
    #Write Model Information to files
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
                file.write('  '+str(t[q])+' '\
                           +str(drain_data['elev'+str(q)][i])+' '\
                           +str(drain_data['cond'+str(q)][i])+'\n')
                q+=1
            else:
                hand=1
        
        file.write('  end \n')
        file.write('end \n\n')
    file.close  
    
 
    # print result   
    print('-----------------')
    print('Sucessfully exported drain data file!\n')
    print('Total Drains: '+str(np.shape(drain_data)[0])+'\n')

def export_timestep(dis):
    #Time data
    t=[] # time
    perlen=dis.perlen.array
    for i in range(0,len(perlen)):
        t.append(np.sum(perlen[0:i]))
    
    # Write Model Information
    file=open('hgs_files/timestep.in', 'w')
    file.write('initial time \n')
    file.write('  '+str(np.min(t))+'\n')
    file.write('output times \n')
    for i in range(len(t)):
        file.write('  '+str(t[i])+'\n')
    file.write('end ! output times \n')
    file.close
    
def export_all(modelprefix, nsublayer=0):
      
    #load modflow
    [ml, dis, bas, lpf, rch, wel, drn]=load_modflowmodel(modelprefix)
    [nrow, ncol, nrowc, ncolc, nlay]=model_dimensions(dis)
    if nsublayer==0:
        nsublayer=np.ones([1,nlay])
    
    #export all
    XYgrid,XYCgrid=export_grid(dis)
    ibound=export_ibound(dis,bas, nsublayer)
    lbl, lblmax=export_mprops(dis,lpf,ibound, nsublayer)
    export_well(dis, wel)
    export_drain(dis, drn)
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
