import matplotlib.pyplot as plt 

def plot_points():
    print("hello")

def visualize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"Layer: {name} | Shape: {param.shape}")
            print(f"# filters {param.shape[0]}, # depth of filters {param.shape[1]}, filter size {param.shape[2]}x{param.shape[3]}")

            for filter_number in range(param.shape[0]):
                plt.figure()
                plt.title(f"{name} filter {filter_number}")
                plt.imshow(param[filter_number].detach().cpu().numpy().squeeze(), cmap='Purples')
                plt.colorbar()
                plt.show()


if __name__ == "__main__":
    print("Hello, world!")