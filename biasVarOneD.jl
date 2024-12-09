import Pkg
Pkg.add(["GLMakie", "FileIO"])

using GLMakie
using FileIO
using LinearAlgebra
# Sample data parameters
num_points = 100                 # Total number of data points
range_x = 20.0                   # Range for x₁ and x₂
m_start = 2.0                    # Initial slope when zoomed in
m_end = 5.0                      # Final slope when zoomed out
noise_level = 2.0 
# Sample data (replace with actual data)
function generate_y(x1, slope, noise_std)
    return slope * x1 + randn(length(x1)) * noise_std
end
X = rand(100, 2) * 20  # 100 points in 2D
current_slope = m_start
y = generate_y(X[:, 1], current_slope, noise_level)
sampled_indices = findall(1:10 .== 1)  # Initial sampled indices
x0 = [10.0, 10.0]
true_coefficients = [3.0, 2.0]
true_y0 = dot(true_coefficients, x0)  # Now dot is defined

# Create a single Observable for sampled points positions
O_sampled_pos = Observable(hcat(X[sampled_indices, 1], zeros(length(sampled_indices))))

# Create a figure with two subplots
fig = Figure(resolution = (800, 400))
ax1 = Axis(fig[1, 1], title = "Sampled Points", xlabel = "X", ylabel = "Y")
ax2 = Axis(fig[1, 2], title = "Full Dataset", xlabel = "X", ylabel = "Y")

# Initialize scatter plots
scatter_sampled = scatter!(ax1, O_sampled_pos,
                          color = :blue, markersize = 8, label = "Sampled Points")
scatter_full_ax1 = scatter!(ax1, X[:, 1], zeros(size(X, 1)),
                            color = :gray, markersize = 4, alpha = 0.5, label = "Full Dataset")
axislegend(ax1)

scatter_full_ax2 = scatter!(ax2, X[:, 1], y,
                            color = :gray, markersize = 4, alpha = 0.3, label = "Full Dataset")
axislegend(ax2)

# Add Test Point x0
scatter!(ax1, x0[1], 0.0, color = :red, markersize = 10, label = "Test Point x₀")

# Add True y0 line

# Set initial axis limits for ax1
initial_radius = 5.0
ylim_start = true_y0 - 5.0
ylim_end = true_y0 + 5.0
ylims!(ax2, minimum(y) - 5, maximum(y) + 5)  # Fixed y-limits for ax2
# Initialize the Learnt Line in ax2 with placeholder data to prevent errors
# Initialize the Learnt Line in ax2 with separate x and y vectors
# Initialize the Learnt Line in ax2 with Observables
learnt_line_x = Observable([x0[1] - initial_radius, x0[1] + initial_radius])
learnt_line_y = Observable([0.0, 0.0])
learnt_line = lines!(ax2, learnt_line_x, learnt_line_y,
                     color = :orange, linewidth = 2, label = "Learnt Line")

# Create an Observable for the current sampling radius
current_radius_obs = Observable(initial_radius)
function compute_regression(x, y)
    if length(x) < 2
        # Not enough points to compute regression
        return NaN, NaN
    end
    X_matrix = hcat(ones(length(x)), x)
    coeffs = X_matrix \ y  # Least squares solution
    intercept, slope = coeffs
    return slope, intercept
end
# Define a callback to update ax2's x-axis limits whenever current_radius_obs changes
on(current_radius_obs) do radius
    xlims!(ax2, x0[1] - radius, x0[1] + radius)
end
# Initialize the Learnt Line in ax2
# Function to update the Observable with new sampled points
function update_sampled_points!(frame)
    max_radius = 20.0
    current_radius = (frame / 100.0) * max_radius

    # Update sampled indices based on current radius
    sampled = abs.(X[:, 1] .- x0[1]) .<= current_radius
    sampled_indices = findall(sampled)

    # Update the Observable with new positions
    new_positions = hcat(X[sampled_indices, 1], zeros(length(sampled_indices)))
    O_sampled_pos[] = new_positions  # Assign new positions to the Observable
   
    
    ###
   
    current_radius_obs[] = current_radius

    if length(sampled_indices) >= 2
        sampled_x = X[sampled_indices, 1]
        sampled_y = y[sampled_indices]
        slope, intercept = compute_regression(sampled_x, sampled_y)

        if !isnan(slope) && !isnan(intercept)
            # Ensure x0[1] and current_radius are scalars
            x_center = x0[1]

            # Create a range of length 100
            regression_x = range(x_center - current_radius, stop=x_center + current_radius, length=100)

            # Compute the corresponding y values
            regression_y = slope * regression_x .+ intercept

            # Update the Observables so that both have the same length arrays
            learnt_line_x[] = regression_x # This is now a Vector{Float64} of length 100
            learnt_line_y[] = regression_y           # Also Vector{Float64} of length 100
        end
    else
        # Not enough points to compute regression; reset the learnt line to placeholder
        learnt_line_x[] = [x0[1] - current_radius, x0[1] + current_radius]
        learnt_line_y[] = [0.0, 0.0]
    end
end

# Animate sampling outward with removal and record to GIF
record(fig, "outward_sampling_with_removal_1D.gif", 1:100; framerate = 20) do frame
    update_sampled_points!(frame)
end

# Display the figure (optional if running interactively)
display(fig)