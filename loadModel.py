import torch
from collections import OrderedDict


def loadModel(model, pathToModel, dataParallelModel=False):

    try:
        #LOAD TRAINED MODEL INTO GPU
        if(torch.cuda.is_available() and dataParallelModel==False):
            model = torch.load(pathToModel)
            print("\n--------model restored--------\n")
            return model
        elif(torch.cuda.is_available() and dataParallelModel==True):
            state_dict = torch.load(pathToModel)
            print(state_dict.keys())
            model.load_state_dict(state_dict)
            print("\n--------DataParallel GPU model restored--------\n")
            return model
        #LOAD MODEL TRAINED ON GPU INTO CPU
        elif(torch.cuda.is_available()==False and dataParallelModel==True):
            state_dict = torch.load(pathToModel, map_location=lambda storage, loc: storage)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                if k[0] == 'm':
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            print("\n--------GPU data parallel model restored--------\n") 
            return model       
        else:
            storage = {"cuda":"cpu"}
            model = torch.load(pathToModel, map_location=lambda storage, loc: storage)
            print("\n--------GPU model restored--------\n")
            return model
    except:
        print("\n--------no saved model found--------\n")


def loadStateDict(model, pathToStateDict):
    try:
        if(torch.cuda.is_available()):
            state_dict = torch.load(pathToStateDict)
            model.load_state_dict(state_dict)
            print("\n--------GPU state dict restored and loaded into GPU--------\n")
            return model

        else:
            state_dict = torch.load(pathToStateDict, map_location=lambda storage, loc: storage)
            #print(state_dict.keys())
            model.load_state_dict(state_dict)
            print("\n--------GPU state dict restored, loaded into CPU--------\n")
            return model
    except:
        print("\n--------no saved model found--------\n")