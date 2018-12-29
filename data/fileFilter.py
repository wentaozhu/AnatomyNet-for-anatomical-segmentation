import os
pathList = ["./HNPETCTclean/", "./HNCetuximabclean/", "./pddca18/"]
for path in pathList:
    for subPath in os.listdir(path):
        for f in os.listdir(path+subPath):
            if f.endswith("_crp.npy"):
                os.remove(path+subPath+"/"+f)
        for f in os.listdir(path+subPath+"/structures/"):
            if f.endswith("_crp.npy"):
                os.remove(path+subPath+"/structures/"+f)