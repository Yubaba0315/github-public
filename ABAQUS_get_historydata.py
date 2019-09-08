from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
from odbAccess import*
import os

def findodb(path, odblist):
    filelist = os.listdir(path)
    for filename in filelist:
        if os.path.isfile(filename):
            if filename.endswith('.odb'):
                odblist.append(os.path.splitext(filename)[0])

root = '.'
print('Root path: '+ os.path.abspath(root))
odblist = []
findodb(root, odblist)
print('ODB files below will be processed: ')
print(odblist)
for odbpath in odblist:
    odb = openOdb(path=str(odbpath)+'.odb')
    file = open(str(odbpath) + '.txt', 'w')
    file.write('U' + ',' + 'RF' + '\n')
    file.write('rad' + ',' + 'kN' + '\n')
    file.write(',' + str(odbpath) + '\n')
    for stepname in odb.steps.keys():
        step=odb.steps[stepname]
        region=step.historyRegions['Node ASSEMBLY.3']
        RF_DataList = region.historyOutputs['RF3'].data
        U_DataList=region.historyOutputs['U3'].data
        # print(RF_DataList, U_DataList)
        for i in range(0, len(U_DataList)):
            # print(str(U_DataList[i][1]) + ','+ str(RF_DataList[i][1])+ '\n')
            file.write(str(U_DataList[i][1]/1500) + ','+ str(RF_DataList[i][1]/1000)+ '\n')
    print('Data output completed: '+str(odbpath) + '.txt')
    file.close()
print('All data output completed!')