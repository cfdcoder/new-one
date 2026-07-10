##Day 2: Structure n Intepretation of Deep Neural Networks
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

x_data = torch.linspace(-2,2,101).reshape(-1,1)
y_data= 2*x_data - x_data**3

# working on y= 2x- x^3

class SimpleNetwork(nn.Module):
    def __init__(self, use_xavier = False):
        super().__init__()
        
        self.layer1= nn.Linear(1,10)
        self.layer2 = nn.Linear(10,20)
        self.layer3 = nn.Linear(20,10)
        self.layer4 = nn.Linear(10,1)

        if  use_xavier:
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            print("using xavier initilization")
        else:
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                nn.init.normal_(layer.weight, mean=0, std= 1.0)
                nn.init.zeros_(layer.bias)
            print("Using Random Initilization")

    def forward(self,x):
        x= torch.tanh(self.layer1(x))
        x= torch.tanh(self.layer2(x))
        x= torch.tanh(self.layer3(x))
        x= self.layer4(x)
        return x


## Training 

def train(model, epochs=500, lr=0.01):
    optimizer= optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses=[]

    for epoch in range(epochs):
        #Forward pass
        prediction = model(x_data)
        loss= loss_fn(prediction, y_data)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #record loss
        losses.append(loss.item())

        #progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch{epoch+1}: loss={loss.item():.6f}")

    return losses

#train both models

model_random = SimpleNetwork(use_xavier=False)
losses_random = train(model_random, epochs= 500)

model_xavier= SimpleNetwork(use_xavier=True)
losses_xavier = train(model_xavier, epochs= 500)

fig, axes = plt.subplots(1,3,figsize=(14,4))

axes[0].plot(losses_random,'r-',label='Random Init', alpha =0.7)
axes[0].plot(losses_xavier,'b-',label='Xavier Init', alpha =0.7)
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Losses')
axes[0].set_title('Training Loss Comparison')
axes[0].legend()
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

with torch.no_grad():
    pred_random= model_random(x_data).numpy()
axes[1].plot(x_data.numpy(), y_data.numpy(), 'b-', linewidth=2, label='Exact')
axes[1].plot(x_data.numpy(), pred_random, 'r--', linewidth=2, label='Predicted')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Random Init Result')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

with torch.no_grad():
    pred_xavier = model_xavier(x_data).numpy()
axes[2].plot(x_data.numpy(), y_data.numpy(), 'b-', linewidth=2, label='Exact')
axes[2].plot(x_data.numpy(), pred_xavier, 'g--', linewidth=2, label='Predicted')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_title('Xavier Init Result')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
print(f"Random Init - Final Loss: {losses_random[-1]:.6f}")
print(f"Xavier Init - Final Loss: {losses_xavier[-1]:.6f}")
print(f"\nXavier initialization typically trains better!")
print("Plot saved as 'exercise1_results.png'")
                
                
