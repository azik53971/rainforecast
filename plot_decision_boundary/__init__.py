import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, X, y, colors=('red', 'blue')):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.01),
                            torch.arange(y_min, y_max, 0.01),
                            indexing='ij')
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
    
    with torch.no_grad():
        Z = model(grid)
    Z = Z.argmax(dim=1).reshape(xx.shape)
    
    cmap = ListedColormap(colors)
    
    plt.contourf(xx.numpy(), yy.numpy(), Z.numpy(), alpha=0.8, cmap=cmap)
    plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=[colors[label] for label in y.numpy()], edgecolors='k', marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Decision Boundary')
    plt.show()

# plot_decision_boundary(model, X, Y, colors=('green', 'orange'))
