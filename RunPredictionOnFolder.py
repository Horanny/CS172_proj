import os
import torch
import cv2
import numpy as np
import FCN_NetModel as FCN 

category={}
category['Vessel']=-1
category['V Label']=-1
category['V Cork']=-1
category['V Parts GENERAL']=-1
category['Ignore']=-1
category['Liquid GENERAL']=-1
category['Liquid Suspension']=-1
category['Foam']=-1
category['Gel']=-1
category['Solid GENERAL']=-1
category['Granular']=-1
category['Powder']=-1
category['Solid Bulk']=-1
category['Vapor']=-1
category['Other Material']=-1
category['Filled']=-1


InputDir="/home/yyin/CV/proj/InputImages" 
OutDir="OutputImages\\" 
Trained_model_path ="/home/yyin/CV/proj/TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch"

UseGPU=False 
FreezeBatchNormStatistics = False 
OutEnding="" 
if (os.path.exists(OutDir) != True): 
    os.makedirs(OutDir) 


Net=FCN.Net(category) 
if UseGPU:
    Net.load_state_dict(torch.load(Trained_model_path))
    print("USING GPU")
else:
    Net.load_state_dict(torch.load(Trained_model_path, map_location=torch.device('cpu')))
    print("USING CPU")
#--------------------------------------------------------------------------------------------------------------------------
# Net.half()
for name in os.listdir(InputDir): # Read and change the images' size
    print(name)
    InPath=InputDir+"/"+name
    Im=cv2.imread(InPath)
    h,w,d=Im.shape
    r=np.max([h,w])

    Imgs=np.expand_dims(Im,axis=0)
    if not (type(Im) is np.ndarray): continue
#................................Make Prediction.............................................................................................................
    with torch.autograd.no_grad():
          OutProbDict,OutLbDict=Net.forward(Images=Imgs,TrainMode=False,UseGPU=UseGPU, FreezeBatchNormStatistics=FreezeBatchNormStatistics) # Run net inference and get prediction
#...............................Save prediction on fil
    for nm in OutLbDict:
        Lb=OutLbDict[nm].data.cpu().numpy()[0].astype(np.uint8)
        if Lb.mean()<0.001: continue
        if nm=='Ignore': continue
        ImOverlay1 = Im.copy()
        ImOverlay1[:, :, 0][Lb==1] = 255
        ImOverlay1[:, :, 1][Lb==1] = 0
        ImOverlay1[:, :, 2][Lb==1] = 255
        FinIm=np.concatenate([Im,ImOverlay1],axis=0)

        if nm != "Liquid GENERAL": continue
        OutPath = OutDir + "//" 

        if not os.path.exists(OutPath): os.makedirs(OutPath)
        OutName=OutPath+name[:-4]+OutEnding+".png"
        cv2.imwrite(OutName,FinIm)
    print("Saving output to: " + OutDir)







