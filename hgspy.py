#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:42:40 2017

@author: Lysander Bresinsky, University of Göttingen (Applied Geology)
"""
import flopy
import numpy as np
from scipy.interpolate import griddata
import os
import matplotlib


def load_modflow_packages(modelprefix):
    '''
    Loads all modflow files via flopy
    
    Input:
        - modelprefix: Prefix of the Model files (*.nam, *.in, *.dis, etc.)
        
    Output:
        - ml: General Modelvariable
        - dis: Discretization Package
        - bas: Basic Package
        - lpf: Layer-Property Flow Package
        - rch: Recharge Package
        - wel: Well Package
        - drn: Drain Package
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
    
    Input:
        - dis: Discretization Package
        
    Output:
        - nrow: Number of rows (block-centered)
        - ncol: Number of columns (block-centered)
        - nrowc: Number of rows (node-centered)
        - ncolc: Number of columns (node-centered)
        - nlay: Number of layer
    '''
    nrow=dis.nrow #number of rows (block centered)
    ncol=dis.ncol #number of columns
    ncolc=ncol+1 #number of columns (node centered)
    nrowc=nrow+1 #number of rows
    nlay=dis.nlay
    
    #xy-coordinates (block-centered)
    y=np.sort(np.array(dis.get_node_coordinates()[0])) 
    x=np.array(dis.get_node_coordinates()[1])
    XYgrid=np.array([np.meshgrid(x,y)[0].ravel(), np.meshgrid(x,y)[1].ravel()]).T # meshgrid creates the coordinates of all existing nodes
    
    #xy-coordinates (node centered)
    xc=flopy.utils.reference.SpatialReference.get_xedge_array(dis)
    yc=np.sort(flopy.utils.reference.SpatialReference.get_yedge_array(dis))
    XYCgrid=np.array([np.meshgrid(xc,yc)[0].ravel(), np.meshgrid(xc,yc)[1].ravel()]).T 
      
    return nrow, ncol, nrowc, ncolc, nlay, x, y, XYgrid, xc, yc, XYCgrid

def units(ml, dis):
    '''
    Get the model units used in Modflow
    
        Input:
        - ml: General Modelvariable
        - dis: Discretization Package
        
    Output:
        - weight: Weight unit
        - length: Lenght unit
        - time: Time unit
    '''
    
    time=dis.itmuni_dict[dis.itmuni][0:-1]
    weight='kilogram' #not yet implemented
    length_dict={0 : 'undefined', 1 : 'feet', 2 : 'metre', 3 : 'centimetre'}
    length=length_dict[dis.lenuni]
    
    return weight, length, time



# exporting single packages - functions  
def export_grid(dis, bas,
                raster_step=80, interpolation_method='nearest',
                nodata_value=-99999,
                nrow=0,  ncol=0, nrowc=0, ncolc=0, nlay=0, 
                x=0, y=0, XYgrid=0, xc=0, yc=0, XYCgrid=0):
    
    # Model geometry
    if (nrow    == 0  or  ncol   == 0  or 
        nrowc   == 0  or  ncolc  == 0  or 
        nlay    == 0  or  x      == 0  or
        y       == 0  or  XYgrid == 0  or 
        xc      == 0  or  yc     == 0  or
        XYCgrid == 0):
        [nrow,  ncol, nrowc, ncolc, nlay,x, y, XYgrid, xc, yc, XYCgrid]=model_dimensions(dis)

    # Get ibound array
    ibound=_get_ibound_array(dis,bas)   
       
    # get bottom array
    btom=_change_order_to_hydrogeosphere(dis.botm.array)
    
    # get top array
    top=_change_order_to_hydrogeosphere(dis.top.array)
    
#    # reduce height when it exceds higher layer
#    btom[-1][np.where(btom[-3]>top)]=top[np.where(btom[-3]>top)]
#    for i in reversed(range(len(btom[0:-1]))):
#        btom[i][np.where(btom[i]>btom[i+1])]=btom[i+1][np.where(btom[i]>btom[i+1])]
    
    # Interpolate Arrays
    [x_step, y_step, 
     x_range, y_range, 
    x_nsteps, y_nsteps, 
    XLLCORNER, YLLCORNER, 
    XYRgrid] = _calculate_raster_parameter(raster_step, xc, yc)
    
    btom_interpolated=_interpolate(btom, XYgrid, XYRgrid, method=interpolation_method, nodata_value=nodata_value)
    top_interpolated=_interpolate(top, XYgrid, XYRgrid, method=interpolation_method, nodata_value=nodata_value)
    ibound_interpolated=_interpolate(ibound, XYgrid, XYRgrid, method=interpolation_method, nodata_value=nodata_value)
    ibound_interpolated_joined=_create_ibound_joined(ibound_interpolated)
    
    # reduce height when it exceds higher layer
    for xi in range(np.shape(btom_interpolated)[2]):
        for yi in range(np.shape(btom_interpolated)[1]):
            if btom_interpolated[-1][yi,xi]>top_interpolated[0][yi,xi]:
                btom_interpolated[-1][yi,xi]=top_interpolated[0][yi,xi]
            else:
               pass
    for zi in reversed(range(np.shape(btom_interpolated)[0]-1)):
        for xi in range(np.shape(btom_interpolated)[2]):
            for yi in range(np.shape(btom_interpolated)[1]):
                if btom_interpolated[zi][yi,xi]>btom_interpolated[zi+1][yi,xi]:
                    btom_interpolated[zi][yi,xi]=btom_interpolated[zi+1][yi,xi]
                else:
                    pass
                
    # increase hight to above when ibound above is zero
    for xi in range(np.shape(btom_interpolated)[2]):
        for yi in range(np.shape(btom_interpolated)[1]):
            if ((ibound_interpolated[-1][yi,xi]==0) &
            (ibound_interpolated_joined[yi,xi]==1)):
                btom_interpolated[-1][yi,xi]=top_interpolated[0][yi,xi]
            else:
               pass    
    for zi in reversed(range(np.shape(btom_interpolated)[0]-1)):
        for xi in range(np.shape(btom_interpolated)[2]):
            for yi in range(np.shape(btom_interpolated)[1]):
                if ((ibound_interpolated[zi][yi,xi]==0) &
            (ibound_interpolated_joined[yi,xi]==1)):
                    btom_interpolated[zi][yi,xi]=btom_interpolated[zi+1][yi,xi]
                else:
                    pass
                
    #increase height for very small cell volumes
    for zi in reversed(range(np.shape(btom_interpolated)[0]-1)):
        for xi in range(np.shape(btom_interpolated)[2]):
            for yi in range(np.shape(btom_interpolated)[1]):
                diff=abs(btom_interpolated[zi][yi,xi]-btom_interpolated[zi+1][yi,xi]) # heigt of the lowest layer
                if ((diff<20)&(ibound_interpolated[zi][yi,xi]==1)):
                    btom_interpolated[zi][yi,xi]=btom_interpolated[zi+1][yi,xi]-20
                else:
                    pass

                
    # merge top and btom
    z_interpolated=btom_interpolated[:]
    z_interpolated.append(top_interpolated[0][:])
  
    # create raster file
    _write_raster('ibound', ibound_interpolated, x_nsteps, y_nsteps, XLLCORNER, YLLCORNER, raster_step=raster_step, nodata_value=-99999, number_format='{0:.0f}')
    _write_raster('z', z_interpolated, x_nsteps, y_nsteps, XLLCORNER, YLLCORNER, raster_step=raster_step, nodata_value=-99999, number_format='{0:.0f}')
    return ibound

def export_mprops(dis, lpf, bas,
                  nrow=0,  ncol=0, nrowc=0, ncolc=0, nlay=0, 
                  x=0, y=0, XYgrid=0, xc=0, yc=0, XYCgrid=0):
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
    

    
    #model geometry
    if (nrow    == 0  or  ncol   == 0  or 
        nrowc   == 0  or  ncolc  == 0  or 
        nlay    == 0  or  x      == 0  or
        y       == 0  or  XYgrid == 0  or 
        xc      == 0  or  yc     == 0  or
        XYCgrid == 0):
        [nrow,  ncol, nrowc, ncolc, nlay,x, y, XYgrid, xc, yc, XYCgrid]=model_dimensions(dis)

    
    #model properties
    hk=_change_order_to_hydrogeosphere(lpf.hk.array) # horizontal hydraulic conductivity
    vk=_change_order_to_hydrogeosphere(lpf.vka.array) # vertical hydraulic conductivity
    ss=_change_order_to_hydrogeosphere(lpf.ss.array) # specific storage
    sy=_change_order_to_hydrogeosphere(lpf.sy.array) # specific yield
 
    # ibound - zero cell values
    ibound=_get_ibound_array(dis, bas)
    
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
    [lbl, lblmax, zone_mprops]=_labelPix(mprops,4)

     # Write Model Information to files
    _dir_check()
    
    file=open('hgs_files/porous_media.mprops', 'w')
    for i in range(len(zone_mprops)):
        file.write('!------------------------------------------\n')
        file.write('material_zone_'+str(i+1))
        file.write('\n\nk anisotropic\n')
        file.write('  '+'{0:.5e}'.format(zone_mprops[i,0]).replace('e','d')+'  '\
                   +'{0:.5e}'.format(zone_mprops[i,0]).replace('e','d')+'  '\
                   +'{0:.5e}'.format(zone_mprops[i,1]).replace('e','d')+'\n\n')
        file.write('Specific storage\n')
        file.write('  '+'{0:.5e}'.format(zone_mprops[i,2]).replace('e','d')+'\n\n')
        #file.write('Porosity\n')
        #file.write('  '+'{0:.5e}'.format(zone_mprops[i,3])+'\n\n')
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
    
    return lbl.reshape([nlay,nrow,ncol]), lblmax, zone_mprops, hk, vk, ss, sy
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
               
        
    return lbl.reshape([nlay,nrow,ncol]), lblmax

def export_zone_raster(lbl, dis, raster_step=40,
                              nrow=0,  ncol=0, nrowc=0, ncolc=0, nlay=0, 
                              x=0, y=0, XYgrid=0, xc=0, yc=0, XYCgrid=0):
    # Model geometry
    if (nrow    == 0  or  ncol   == 0  or 
        nrowc   == 0  or  ncolc  == 0  or 
        nlay    == 0  or  x      == 0  or
        y       == 0  or  XYgrid == 0  or 
        xc      == 0  or  yc     == 0  or
        XYCgrid == 0):
        [nrow,  ncol, nrowc, ncolc, nlay,x, y, XYgrid, xc, yc, XYCgrid]=model_dimensions(dis)
    
    
    
    #write raste
    _write_raster('zones',lbl,xc,yc, XYgrid,raster_step=raster_step, method='nearest', nodata_value=1, number_format='{0:.0f}')
    
def export_recharge_raster(dis, rch, bas,
                           raster_step=80, interpolation_method='nearest',
                           nodata_value=-99999,
                           nrow=0,  ncol=0, nrowc=0, ncolc=0, nlay=0,
                           x=0, y=0, XYgrid=0, xc=0, yc=0, XYCgrid=0):
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
    if (nrow    == 0  or  ncol   == 0  or 
        nrowc   == 0  or  ncolc  == 0  or 
        nlay    == 0  or  x      == 0  or
        y       == 0  or  XYgrid == 0  or 
        xc      == 0  or  yc     == 0  or
        XYCgrid == 0):
        [nrow,  ncol, nrowc, ncolc, nlay,x, y, XYgrid, xc, yc, XYCgrid]=model_dimensions(dis)

    # model properties    
    rc=[]
    for i in range(0, len(rch.rech.array)):
        rc.append(rch.rech.array[i]) # recharge per stress period per cell
    
    # Get ibound array
    ibound=_get_ibound_array(dis,bas)  
    
    #flip
    for i in range(0, len(rc)):
        rc[i]=np.flipud(rc[i])
    return rc
    # calcuate absolute time of each step
    t=[]
    perlen=dis.perlen.array
    for i in range(0,len(perlen)):
        t.append(np.sum(perlen[0:i]))

    # Average RECHAGRE
    a=[]
    for i in range(len(rc)):
        a.append(np.mean(rc[i][0][np.where((ibound[0]+ibound[1]+ibound[2])==1)]))
    rch_avg=np.mean(a)
   
    # Get rid of layers within stress perios
    rcc=[]
    for i in range(len(rc)):
        rcc.append(rc[i][0])
        
    # Create stationary recharge variable
    rcc_stat=np.sum(rcc, 0)/np.shape(rcc)[0]/10
    #_write_raster('rch/recharge_steady',rcc_stat,xc,yc, XYgrid,raster_step=40)

    # Write Model Information to files
    [x_step, y_step, 
     x_range, y_range, 
    x_nsteps, y_nsteps, 
    XLLCORNER, YLLCORNER, 
    XYRgrid] = _calculate_raster_parameter(raster_step, xc, yc)
    ##interpolate    
    rcc_i=_interpolate(rcc, XYgrid, XYRgrid, method=interpolation_method, nodata_value=nodata_value)
    rcc_stat_i=_interpolate(rcc_stat, XYgrid, XYRgrid, method=interpolation_method, nodata_value=nodata_value)
    ##write raster
    _write_raster('rch/recharge_stationary',rcc_stat_i,x_nsteps, y_nsteps, XLLCORNER, YLLCORNER, raster_step=raster_step, nodata_value=nodata_value, number_format='{0:.8f}')
    _write_raster('rch/recharge_t',rcc_i,x_nsteps, y_nsteps, XLLCORNER, YLLCORNER, raster_step=raster_step, nodata_value=nodata_value, number_format='{0:.8f}')
    
    
    file=open('hgs_files/recharge_raster.in', 'w')
    for i in range(0,len(t)):
        file.write(str(t[i])+' '+'hgs_files/rch/recharge_t'+str(i+1)+'.asc \n')
    file.close

    # print result   
    print('-----------------')
    print('Sucessfully exported recharge files!\n')
    print('Average Rcharge: '+'{0:.2e}'.format(rch_avg))
    
    
def export_well(dis, wel, 
                nrow=0,  ncol=0, nrowc=0, ncolc=0, nlay=0, 
                x=0, y=0, XYgrid=0, xc=0, yc=0, XYCgrid=0):
    #-----------
    # Model Information
    #-----------   
    if (nrow    == 0  or  ncol   == 0  or 
        nrowc   == 0  or  ncolc  == 0  or 
        nlay    == 0  or  x      == 0  or
        y       == 0  or  XYgrid == 0  or 
        xc      == 0  or  yc     == 0  or
        XYCgrid == 0):
        [nrow,  ncol, nrowc, ncolc, nlay,x, y, XYgrid, xc, yc, XYCgrid]=model_dimensions(dis)

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
        z_bot=z[well_data['k'][i]][well_data['i'][i],well_data['j'][i]]
        z_top=z[well_data['k'][i]+1][well_data['i'][i],well_data['j'][i]]
        z_well_bot=z_bot+((z_top-z_bot)/2)
        z_wel.append(z_well_bot)
    well_data['x']=x_wel
    well_data['y']=y_wel
    well_data['z']=z_wel
    
    
    #Write Model Information to files
    _dir_check()
    
   
    for i in range(len(well_data)):  
        file=open('./hgs_files/well_db/well_'+str(i)+'.in', 'w')              
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
        q_tot=0
        q_count=0
        for q in range(len(t)):
            if hasattr(well_data, str('flux'+str(q))):
                q_tot+= well_data['flux'+str(q)][i]
                q_count+=1
                q_prev = q_now
                q_now = well_data['flux'+str(q)][i]
                if q_now != q_prev:
                    file.write('  '+str(t[q])+' '\
                           +str(well_data['flux'+str(q)][i])+'\n')
                else:
                    pass
            else:
                pass
        q_avg=q_tot/q_count
        file.write('  end \n')
        file.write('  \n')
        file.write('  tecplot output\n\n')
        file.write('end \n\n')
        file.close()
 

        # Steady-State Extraction     
        file=open('./hgs_files/well_db/well_stat_'+str(i)+'.in', 'w')              
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
        file.write('   0 '+str(q_avg)+'\n')  
        file.write('  end \n')
        file.write('  \n')
        file.write('  tecplot output\n\n')
        file.write('end \n\n')
        file.close
    
    file=open('./hgs_files/well_bc.in', 'w') 
    for i in range(len(well_data)):
        file.write('include '+'./well_db/well_'+str(i)+'.in \n')
    file.close
    
    file=open('./hgs_files/well_bc_stat.in', 'w') 
    for i in range(len(well_data)):
        file.write('include '+'./well_db/well_stat_'+str(i)+'.in \n')
    file.close
    
    # print result   
    print('-----------------')
    print('Sucessfully exported well data file!\n')
    print('Total Wells: '+str(np.shape(well_data)[0])+'\n')
    
    return well_data
 

def export_drain(dis, drn,
                 nrow=0,  ncol=0, nrowc=0, ncolc=0, nlay=0, 
                 x=0, y=0, XYgrid=0, xc=0, yc=0, XYCgrid=0):      
    # Model Information
    if (nrow    == 0  or  ncol   == 0  or 
        nrowc   == 0  or  ncolc  == 0  or 
        nlay    == 0  or  x      == 0  or
        y       == 0  or  XYgrid == 0  or 
        xc      == 0  or  yc     == 0  or
        XYCgrid == 0):
        [nrow,  ncol, nrowc, ncolc, nlay,x, y, XYgrid, xc, yc, XYCgrid]=model_dimensions(dis)

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
        z_bot=z[drain_data['k'][i]][drain_data['i'][i],drain_data['j'][i]]
        z_top=z[drain_data['k'][i]+1][drain_data['i'][i],drain_data['j'][i]]
        z_drain_bot=z_bot+((z_top-z_bot)/2)
        z_drn.append(z_drain_bot)
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

    return drain_data
 
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

def _polish_grid(z, ibound, nrow, ncol, nlay):
    '''
    Input:
        z - Array of heights in the following shape (nlay, nrow, ncol)
        ibound - Array of inactive/active elements (0 = inactive, 1 = active)
        nrow - number of rows
        ncol - number of columns
        nlay -  number of layers
        
    Output:
        z_polished - polished array of z
    
    This function imnproves grid quality for HGS by removing height information
    of the inactive cells and by reducing the height in cases of layers exceeding
    the higher layer
    '''
    
    # get ibound array and add safety margin 
#    if len(np.shape(ibound))==3:
#        ibound_sum=np.sum(ibound, axis=0)
#        ibound_joined=np.squeeze(np.array([ibound_sum>0], dtype=int))
#    else:
#        ibound_joined=ibound
    
    pos_ones=np.where(ibound_joined==1)
    for i in [0,1,-1,]:
        for q in [0,1,-1]:
            ibound_joined[pos_ones[0]+i,pos_ones[1]+q]=1
    
    #Filter out height information outside of model domain
#    for i in range(nlay+1):
#        z[i]=z[i]*ibound_joined
    
    # increase height
#    for i in range(nlay):
#        cond2=abs(z[i]-z[i+1])<10
#        cond3=ibound[i]!=0
#        print(np.shape(np.where(cond2))[1])
    for i in range(nlay):
        cond2=abs(z[i]-z[i+1])<10
        cond1=abs(z[i]-z[i+1])==0
        print(np.shape(np.where(cond1&cond2))[1])
        pos=np.where(cond1&cond2)
        for q in range(np.shape(pos)[1]):
            z[i][pos[0][q],pos[1][q]]=z[i+1][pos[0][q],pos[1][q]]

    # Filter out unusua al high or low botm layers
    for q in range(4):
        for i in range(0,len(z)-1):
            z[i][np.where(z[i]>z[i+1])]=z[i+1][np.where(z[i]>z[i+1])]


    return z

# Internal functions    
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
    if os.path.isdir('./hgs_files/well_db')==1:
        pass
    elif os.path.isdir('./hgs_files/well_db')==0:
        os.mkdir('./hgs_files/well_db')
        
        
def _labelPix(pix,n_val):
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
        
def _interpolate(array, XYgrid, XYgrid_interpolated, method='cubic', nodata_value=-99999):
    array_interpolated=[]
    shape=[len(np.unique(XYgrid_interpolated[:,1])),
           len(np.unique(XYgrid_interpolated[:,0]))]
    
    if len(np.shape(array))==3:    
        for i in range(0,len(array)):
            array_interpolated.append(griddata(XYgrid,array[i].ravel(),XYgrid_interpolated, method=method,fill_value=nodata_value).reshape(shape)) # calculated via cubic interpolation
    elif len(np.shape(array))==2:
        array_interpolated.append(griddata(XYgrid,array.ravel(),XYgrid_interpolated, method=method,fill_value=nodata_value).reshape(shape))
    else:
        raise IndexError
        
    return array_interpolated

def _write_raster(file_name, array, x_nsteps, y_nsteps, XLLCORNER, YLLCORNER, raster_step=40, nodata_value=-99999, number_format='{0:.2f}'):
      
    # Export to raster
    if len(np.shape(array))==3:
        iterations=len(array)
    elif len(np.shape(array))==2:
        iterations=1  
  
    for i in range(iterations):
        file=open('hgs_files/'+file_name+str(i+1)+'.asc','w')
        file.write('NCOLS '+str(x_nsteps)+'\n')
        file.write('NROWS '+str(y_nsteps)+'\n')
        file.write('XLLCORNER '+'{0:.2f}'.format(XLLCORNER)+'\n')
        file.write('YLLCORNER '+'{0:.2f}'.format(YLLCORNER)+'\n')
        file.write('CELLSIZE '+'{0:.2f}'.format(raster_step)+'\n')
        file.write('NODATA_VALUE '+number_format.format(nodata_value)+'\n')
        for q in reversed(range(y_nsteps)):
            for element in array[i][q,:]:
                file.write(number_format.format(element).replace('e','d')+' ')
            file.write('\n')
        file.close
        
def _change_order_to_hydrogeosphere(array):
    # change layer numbering
    array=np.flipud(array)
    
    # flip along x-achis
    if len(np.shape(array))==2:
        pass
    else:
        iterations=np.shape(array)[0]
        for i in range(iterations):
            array[i]=np.flipud(array[i])
    
    return array

def _create_ibound_joined(ibound):
    if len(np.shape(ibound))==3:
        ibound_sum=np.sum(ibound, axis=0)
        ibound_joined=np.squeeze(np.array([ibound_sum>0], dtype=int))
    else:
        ibound_joined=ibound
        
    return ibound_joined

def _get_ibound_array(dis, bas):
    ibound=_change_order_to_hydrogeosphere(bas.ibound.array)    
    ibound=_get_rid_zero_cells_ibound(dis, ibound) # get rid of zero volume cells
    return ibound

def _get_rid_zero_cells_ibound(dis, ibound):
    #get cell volume
    cell_volume=dis.get_cell_volumes()
    
    #change layer numbering
    cell_volume=np.flipud(cell_volume)
    
    #flip along axis
    for i in range(len(cell_volume)):
        cell_volume[i]=np.flipud(cell_volume[i])
        
    # get rid of all cells with an zero volume
    for i in range(len(ibound)):
        ibound[i][np.where((cell_volume[i]==0)&(ibound[i]==1))]=0
        
    return ibound

def _calculate_raster_parameter(raster_step, xc, yc):
    # raster parameter
    x_step=raster_step
    y_step=raster_step
    x_range=range(int(np.min(xc)-x_step*1), int(np.max(xc)+x_step*1), x_step)
    y_range=range(int(np.min(yc)-y_step*1), int(np.max(yc)+y_step*1), y_step)
    x_nsteps=len(x_range)
    y_nsteps=len(y_range)
    XLLCORNER=np.min(x_range)
    YLLCORNER=np.min(y_range)
    XYRgrid=np.array([np.meshgrid(x_range,y_range)[0].ravel(), np.meshgrid(x_range,y_range)[1].ravel()]).T 
    
    return x_step, y_step, x_range, y_range, x_nsteps, y_nsteps, XLLCORNER, YLLCORNER, XYRgrid
 
def _repair_wrong_ibound_val(lbl):
    pos=np.where((lbl[2]!=1) & (lbl[0]!=1) & (lbl[1]==1))
    lbl[1][pos]=lbl[0][pos]
    return lbl
