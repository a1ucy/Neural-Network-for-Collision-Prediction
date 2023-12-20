import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.fc1 = nn.Linear(6,64)
        self.fc2 = nn.Linear(64,128)
        self.fc3 = nn.Linear(128,256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,64)
        self.output = nn.Linear(64,1)
        self.af = nn.ReLU()
        # self.af2 = nn.Sigmoid
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        x = self.dropout(self.af(self.fc1(input)))
        x = self.dropout(self.af(self.fc2(x)))
        x = self.dropout(self.af(self.fc3(x)))
        x = self.dropout(self.af(self.fc4(x)))
        x = self.dropout(self.af(self.fc5(x)))
        output = self.af(self.output(x))
        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        model.eval()
        sum_loss = 0
        with torch.no_grad():
            for i in test_loader:
                x, y = i['input'], i['label']
                output = model(x)
                loss = loss_function(output, y.unsqueeze(1))
                sum_loss += loss.item()
        return sum_loss/len(test_loader)

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
