# question_4_vanishing_gradient.py
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
INPUT_SIZE=16
HIDDEN_SIZE=32
SEQUENCE_LENGTHS=[10,25,50,100]
TRIALS=10
DEVICE=torch.device("cpu")

def measure_gradient(cell_type,seq_len):
    values=[]
    for seed in range(TRIALS):
        torch.manual_seed(seed)
        if cell_type=="RNN":cell=nn.RNN(INPUT_SIZE,HIDDEN_SIZE,batch_first=True)
        elif cell_type=="LSTM":cell=nn.LSTM(INPUT_SIZE,HIDDEN_SIZE,batch_first=True)
        else:cell=nn.GRU(INPUT_SIZE,HIDDEN_SIZE,batch_first=True)
        cell=cell.to(DEVICE)
        x=torch.randn(1,seq_len,INPUT_SIZE,device=DEVICE,requires_grad=True)
        output,state=cell(x)
        final_output=output[:,-1,:]
        loss=final_output.square().mean()
        cell.zero_grad()
        loss.backward()
        first_step_grad=x.grad[:,0,:].norm().item()
        values.append(first_step_grad)
    return float(np.mean(values))

print(f"{'Sequence length':<18}{'Vanilla RNN':<18}{'LSTM':<18}{'GRU':<18}")
print("-"*72)
for length in SEQUENCE_LENGTHS:
    rnn_grad=measure_gradient("RNN",length)
    lstm_grad=measure_gradient("LSTM",length)
    gru_grad=measure_gradient("GRU",length)
    print(f"{length:<18}{rnn_grad:<18.8e}{lstm_grad:<18.8e}{gru_grad:<18.8e}")
print("\nInterpretation: compare how the gradient reaching the first input changes as sequence length increases.")
print("Smaller gradients indicate weaker backward signal to the earlier input.")
print("LSTM and GRU can help gradient flow, but the result depends on initialization and sequence length.")