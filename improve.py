#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 20:20:09 2023

@author: flavio
"""
import numpy as np
import plotly.graph_objects as go


def okada(x,y,z, xs, ys, zs, azimuth, dip, length, width, strikeslip=0, dipslip=0, opening=0, nu=0.25):
    
    ux = []
    uy = []
    uz = []
    
    
    for x_, y_,z_ in zip(x,y,z):
       ux_, uy_, uz_ = _okada(x_,y_,z_, xs, ys, zs, azimuth, dip, length, width, strikeslip, dipslip, opening, nu=nu)
       
       ux.append(ux_)
       uy.append(uy_)
       uz.append(uz_)
        
    ux = np.array(ux) 
    uy = np.array(uy) 
    uz = np.array(uz) 
     
    return ux, uy, uz

def _okada(x,y,z, xs, ys, zs, azimuth, dip, length, width, strikeslip=0, dipslip=0, opening=0, nu=0.25):
    # % S.ParameterNames = {'E','N','U','Azim','DIP','Len','Wid','Stk-s','Dip-s','Tens'};
        
    x0 = (x - xs)/1000
    y0 = (y-ys)/1000
    z0 = (z-zs)/1000
    
    XROT = np.array([[np.cos(azimuth*np.pi/180), -np.sin(azimuth*np.pi/180), 0],
        [np.sin(azimuth*np.pi/180), np.cos(azimuth*np.pi/180), 0],
        [0, 0, 1]])
    
    newx = np.dot(XROT, np.array([x0,y0,z0]))
    
    H =  newx[2]    
    XX = newx[0]
    YY = newx[1]
    FA = 0
   
    
    ELAS = (1-2*nu)
        
    US = np.array([opening, strikeslip, -dipslip])
    TA = dip*np.pi/180
    CT=np.cos(TA)
    ST=np.sin(TA)
    CTT=CT/ST
    CF=np.cos(FA)
    SF=np.sin(FA)
    W=width/1000;
    RA=length/2/1000
    D=H+W*ST
    XPA=-H*CF*CTT
    YPA=H*SF*CTT
    
    tmpX,tmpY,AZ = FAULT(XX,YY,XPA,YPA,RA,W,D,1,SF,CF,CT,ST,CTT,ELAS,US)
    AX,AY,tmpZ =  FAULT(XX,YY,XPA,YPA,RA,W,D,2,SF,CF,CT,ST,CTT,ELAS,US)
    
    
    u = np.dot(XROT.T, np.array([AX, AY, AZ]))
    
    return u[0], u[1], u[2]



def  FAULT(XX,YY,XP,YP,AL,W,D,IND,SF,CF,CT,ST,CTT,ELAS,US):

    X1= (XX-XP)*SF + (YY-YP)*CF
    Y1=-(XX-XP)*CF + (YY-YP)*SF
    
    X2= X1
    Y2= Y1 + D*CTT
    
    QS1= X2+AL
    QS2= X2-AL
    P1= Y2*CT +D*ST
    P2= P1 - W
    Q= Y2*ST - D*CT
    
    UX1,UY1,UZ1 = DISLOK (QS1,P1,IND,CT,ST,CTT,ELAS,US,Q)
    UX2,UY2,UZ2 = DISLOK (QS1,P2,IND,CT,ST,CTT,ELAS,US,Q)
    UX3,UY3,UZ3 = DISLOK (QS2,P1,IND,CT,ST,CTT,ELAS,US,Q)
    UX4,UY4,UZ4 = DISLOK (QS2,P2,IND,CT,ST,CTT,ELAS,US,Q)
    
    A2 = UX1-UX2-UX3+UX4
    B2 = UY1-UY2-UY3+UY4
    C2 = UZ1-UZ2-UZ3+UZ4
    
    AX=(A2*SF - B2*CF)/2./np.pi
    AY=(A2*CF + B2*SF)/2./np.pi
    AZ=C2/2./np.pi
    
    return AX, AY, AZ

def DISLOK(QSI,ETA,IND,CT,ST,CTT,ELAS,US,Q):

    BY=ETA*CT+Q*ST
    BD=ETA*ST-Q*CT
    R=np.sqrt(QSI**2+ETA**2+Q**2)
    XX=np.sqrt(QSI**2+Q**2)
    
    RE=R+ETA
    RQ=R+QSI
    QRE=Q/R/RE
    QRQ=Q/R/RQ
    ATQE=np.arctan(QSI*ETA/Q/R)
    
    AT= ETA*(XX+Q*CT)+XX*(R+XX)*ST
    BT= QSI*(R+XX)*CT
    
    FUX=0
    FUY=0
    FUZ=0
    
    AI4= ELAS*(np.log(R+BD)-ST*np.log(R+ETA))/CT
    AI5= 2*ELAS*np.arctan(AT/BT)/CT
    
    if (IND==1):
    
        if (US[0]!=0):
            FUZ=(BY*QRQ+CT*QSI*QRE-CT*ATQE-AI5*ST*ST)*US[0]

        if(US[1]!=0):
            FUZ=FUZ-(BD*QRE+Q*ST/RE+AI4*ST)*US[1]
    
        if (US[2]!=0):
            FUZ=FUZ+(-BD*QRQ-ST*ATQE+AI5*ST*CT)*US[2]
    
    else:
        AI3=ELAS*(BY/CT/(R+BD)-np.log(R+ETA))+AI4/CTT
        AI1= -ELAS*QSI/CT/(R+BD)-AI5/CTT
        AI2= -ELAS*np.log(R+ETA)-AI3
    
        if (US[0]!=0):
            FUX=(Q*QRE - AI3*ST*ST)*US[0]
            FUY=-(BD*QRQ+ST*QRE*QSI-ST*ATQE+AI1*ST*ST)*US[0]
        
        if(US[1]!=0):
            FUX=FUX-(QSI*QRE+ATQE+AI1*ST)*US[1]
            FUY=FUY-(BY*QRE+Q*CT/RE+AI2*ST)*US[1]
        
    
        if (US[2]!=0):
            FUX=FUX+(-Q/R+AI3*ST*CT)*US[2]
            FUY=FUY-(BY*QRQ+CT*ATQE-AI1*ST*CT)*US[2]
        
    
    return FUX,FUY,FUZ



def plotOkada(xs, ys, zs, azimuth, dip, length, width):
    cx = xs
    cy = ys
    cz = zs
    
    AZIM = azimuth*np.pi/180
    DIP = dip*np.pi/180
    LEN2 = length/2;
    WID = width
    
    x = np.array([0]*5)
    y = np.array([0]*5)
    z = np.array([0]*5)
    
    x[0] = 0;
    y[0] = LEN2;
    z[0] = 0;
    
    x[1] = 0;
    y[1] = -LEN2;
    z[1] = 0;
    
    x[2] = WID*np.cos(DIP)*np.sign(DIP);
    y[2] = y[1];
    z[2] = 0 - WID*np.sin(DIP)*np.sign(DIP);
    
    x[3] = x[2];
    y[3] = y[0];
    z[3] = z[2];
    
    x[4] = x[0];
    y[4] = y[0];
    z[4] = z[0];
    
    sX = np.linspace(np.min(x),np.max(x),20);
    sY = np.linspace(np.min(y),np.max(y),20);
    
    [XX,YY] = np.meshgrid(sX,sY);
    
    Z = z[0] + (z[2]-z[0])*XX/x[2];
    
    YY = YY;
    ZZ = Z;
    
    X = XX*np.cos(AZIM) + YY*np.sin(AZIM);
    Y = -XX*np.sin(AZIM) + YY*np.cos(AZIM);
    Z = ZZ;
    
    X = X + cx;
    Y = Y + cy;
    
    magnificvert = 2
    Z = Z*magnificvert + cz;
    
    return X, Y, Z
    


def plot3Dfield(ux, uy, uz, eastings, northings, elevations = None,
                scalehorizfactor = 80000, scalevertfactor = 10000, arrow_tip_ratio = 0.2,
                arrow_starting_ratio = 0.90, color='blue', fig=None):
    
    # Create figure
    if fig is None:
        fig = go.Figure()
    
    if elevations is None:
        elevations = np.zeros(eastings.shape)


    # Add the lines and cones to the figure
    for x0,y0,z0,dx,dy,dz in zip(eastings.reshape(-1), northings.reshape(-1), elevations.reshape(-1), 
                          ux.reshape(-1), uy.reshape(-1), uz.reshape(-1)):

        # magnification factor
        z0 *= 1
        
        fig.add_trace(go.Scatter3d(
            x=[x0, x0 + dx*scalehorizfactor],
            y=[y0, y0 + dy*scalehorizfactor],
            z=[z0, z0 + dz*scalevertfactor],
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False
        ))

        fig.add_trace(go.Cone(
            x=[x0 + arrow_starting_ratio*dx*scalehorizfactor],
            y=[y0 + arrow_starting_ratio*dy*scalehorizfactor],
            z=[z0 + arrow_starting_ratio*dz*scalevertfactor],
            u=[arrow_tip_ratio*dx*scalehorizfactor],
            v=[arrow_tip_ratio*dy*scalehorizfactor],
            w=[arrow_tip_ratio*dz*scalevertfactor],
            showlegend=False,
            showscale=False,
            colorscale=[[0, color], [1, color]]
            ))
        
        
    return fig

