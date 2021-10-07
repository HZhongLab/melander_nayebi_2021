import numpy as np
import matplotlib.pyplot as plt
import os.path as path

def save_figure(name,user='josh_dr',figure_num=1,dpi=80,ext='.eps'):
    if user == 'josh_dr':
        basepath = '/home/melander/REPOS/synvivo/figures/'
    elif user == 'aran_laptop':
        basepath = '/Users/anayebi/Desktop/synvivo/figures/'
    
    basepath = path.join(basepath,'figure_' + str(figure_num))
    basepath = path.join(basepath,name+ext)
    
    plt.savefig(basepath,dpi=dpi, bbox_inches='tight')
    
    
    