from model_utils import load_data, train_model
import matplotlib.pyplot as plt

# Load and train on red wine
X_red, y_red = load_data("data/winequality-red.csv")
_, _, red_mse, red_rmse = train_model(X_red, y_red, learning_rate=0.001, epochs=100, label="Red Wine")

# Load and train on white wine
X_white, y_white = load_data("data/winequality-white.csv")
_, _, white_mse, white_rmse = train_model(X_white, y_white, learning_rate=0.001, epochs=100, label="White Wine")

# Plot side-by-side: RMSE and MSE
epochs = list(range(1, 101))
plt.figure(figsize=(12, 6))

# RMSE Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, red_rmse, label='Red Wine RMSE', color='red')
plt.plot(epochs, white_rmse, label='White Wine RMSE', color='blue')
plt.title("RMSE over Epochs")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.grid(True)
plt.legend()

# MSE Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, red_mse, label='Red Wine MSE', color='red')
plt.plot(epochs, white_mse, label='White Wine MSE', color='blue')
plt.title("MSE over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show(block=False)

# Print final RMSE values
print("\n📊 Final RMSE Comparison:")
print(f"🍷 Red Wine  RMSE: {red_rmse[-1]:.4f}")
print(f"🥂 White Wine RMSE: {white_rmse[-1]:.4f}")

# Ask user if they want to save the chart
save = input("\n💾 Do you want to save this chart as 'wine_comparison.png'? (yes/no): ").strip().lower()
if save == "yes":
    plt.savefig("wine_comparison.png", dpi=300)
    print("✅ Chart saved as 'wine_comparison.png'.")

# Show plot (blocking until you close it manually)
# Show the plot without blocking the script
plt.show(block=False)

# Give a message and let the script end naturally
print("\n🖼️ Chart is open — feel free to inspect it. Close the chart window when you're done.")
