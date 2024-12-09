using CairoMakie
using Random

# Step 1: Generate a synthetic dataset
function generate_data(n_points::Int)
    x1 = 10 .* randn(n_points)  # Feature 1
    x2 = 10 .* randn(n_points)  # Feature 2
    noise = randn(n_points)     # Noise
    y = 3.0 .* x1 .+ 2.0 .* x2 .+ noise  # Linear relationship with noise
    X = hcat(x1, x2)  # Combine features into a matrix
    return X, y
end

n_points = 500
X, y = generate_data(n_points)

# Step 2: Compute distances from the origin
distances = [sqrt(x[1]^2 + x[2]^2) for x in eachrow(X)]

# Observables
sampled_indices = Observable(Int[])
predictions = Observable(Float32[])
line_x = Observable(1:1)  # Observable for x-values
line_y = Observable(Float32[])  # Observable for predictions

# Create plots
fig = Figure(resolution=(800, 600))
ax1 = Axis(fig[1, 1], title="Sampling Outward with Removal", xlabel="x₁", ylabel="x₂")
ax2 = Axis(fig[1, 2], title="Prediction Variance for f(y₀)", xlabel="Sample Iterations", ylabel="f(y₀)")

# Scatter plot for sampled data
scatter_positions = @lift(hcat(X[$sampled_indices, 1], X[$sampled_indices, 2]))
scatter!(ax1, scatter_positions, color=:blue, markersize=8, label="Sampled Points")
scatter!(ax1, X[:, 1], X[:, 2], color=:gray, markersize=4, transparency=0.5, label="Full Dataset")
axislegend(ax1)

# Line plot for f(y₀) variance
lines!(ax2, line_x, line_y, color=:orange, label="Predicted f(y₀)")
hlines!(ax2, [true_y0], linestyle=:dash, color=:red, label="True f(y₀)")
axislegend(ax2)

# Step 3: Compute coefficients for sampled points
function compute_coefficients(sample_indices)
    X_sample = X[sample_indices, :]
    y_sample = y[sample_indices]
    return X_sample \ y_sample  # Solve for β (least squares)
end

# Step 4: Define test point
x0 = [1.0, 1.0]  # Test point
true_coefficients = [3.0, 2.0]
true_y0 = true_coefficients' * x0

# Step 5: Animate sampling outward with removal
current_radius = 0.0
record(fig, "outward_sampling_with_removal.gif", 1:100; framerate=20) do frame
    # Define the current radius
    max_radius = 20.0
    prev_radius = ((frame - 10) / 100.0) * max_radius
    prev_radius = max(prev_radius, 0.0)
    current_radius = (frame / 100.0) * max_radius
    
    p = (frame / 100.0) * max_radius

    # Filter indices for points within the current radius
    current_indices = findall((distances .<= current_radius) .& (distances .> prev_radius))    # Update sampled points
    sampled_indices[] = current_indices

    # Compute coefficients and predict f(y₀)
    coefficients = compute_coefficients(sampled_indices[])
    current_y0 = coefficients' * x0

    # Update predictions for the line plot
    # line_x[] = 1:frame
    push!(line_y[], current_y0)
end
