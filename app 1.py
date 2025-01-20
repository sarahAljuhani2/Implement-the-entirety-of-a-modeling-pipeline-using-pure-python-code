import random
import streamlit as st
import plotly.graph_objects as go


# Step 1: Generate synthetic data
def generate_data(num_samples, slope, intercept, noise_std):
    x = [random.uniform(0, 10) for _ in range(num_samples)]
    y = [slope * xi + intercept + random.uniform(-noise_std, noise_std) for xi in x]
    return x, y


# Step 2: Linear regression formula
def linear_regression(x, w, b):
    return [w * xi + b for xi in x]


# Step 3: Loss function (Mean Squared Error)
def compute_loss(y_true, y_pred):
    n = len(y_true)
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n


# Step 4: Optimization function (Gradient Descent)
def optimize(x, y_true, w, b, learning_rate):
    n = len(x)
    dw = -2 / n * sum(x[i] * (y_true[i] - (w * x[i] + b)) for i in range(n))
    db = -2 / n * sum(y_true[i] - (w * x[i] + b) for i in range(n))
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b


# Step 5: Training function
def train(x, y, epochs, learning_rate):
    w, b = 0, 0
    for epoch in range(epochs):
        y_pred = linear_regression(x, w, b)
        loss = compute_loss(y, y_pred)
        w, b = optimize(x, y, w, b, learning_rate)
    return w, b


# Streamlit application
def main():
    st.title("Linear Regression Predictor with Plotly Visualization")

    # Generate synthetic data
    st.write("Training the model on synthetic data...")
    x, y = generate_data(100, 2, 5, 1)

    # Train the model (using default settings)
    w, b = train(x, y, epochs=1000, learning_rate=0.01)

    # Create a scatter plot for the synthetic data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Synthetic Data', marker=dict(color='blue', size=8)))

    # Create a line plot for the model predictions
    x_line = [i for i in range(0, 11)]  # Plotting line from 0 to 10
    y_line = linear_regression(x_line, w, b)
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Model Prediction', line=dict(color='red')))

    st.plotly_chart(fig)

    # Input for user-provided model parameters
    st.write("You can now input your custom values for w and b:")
    w_input = st.number_input("Enter w (slope)", value=2.0, step=0.1)
    b_input = st.number_input("Enter b (intercept)", value=5.0, step=0.1)

    # Input for prediction
    st.write("Enter x-values (comma-separated) to predict y-values:")
    user_input = st.text_input("Input x-values", "1, 2, 3, 4, 5")
    if user_input:
        try:
            x_values = list(map(float, user_input.split(',')))
            predictions = linear_regression(x_values, w_input, b_input)
            st.write(f"Predicted y-values: {predictions}")

            # Show the prediction as a plot (for user inputs)
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=x_values, y=predictions, mode='markers', name='Predictions',
                                          marker=dict(color='green', size=8)))
            st.plotly_chart(fig_pred)
        except ValueError:
            st.error("Please enter valid comma-separated numbers.")


if __name__ == "__main__":
    main()
