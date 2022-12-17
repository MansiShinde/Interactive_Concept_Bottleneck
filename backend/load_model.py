import torch 
import pickle



def define_con_model():

    con_model = torch.jit.load('../backend/models/model_scripted.pt')
    con_model.eval()

    return con_model

def define_mlp_model():
    svm_model =pickle.load(open("../backend/models/model_scripted_svm1.sav", 'rb'))
    
    return svm_model