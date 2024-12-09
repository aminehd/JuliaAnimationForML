ENV["DISPLAY"] = ":0"
import Pkg; Pkg.add("GLMkie")
using GLMakie
using Statistics  # For mean and std
import Pkg; Pkg.add("GeometryBasics")
using GeometryBasics  # Import Point2f0

using PyCall
# Import NumPy
np = pyimport("numpy")

println(ENV["DISPLAY"])
println("amineh")

# Generate synthetic 2D data for linear regression
function generate_data(n_points::Int)
    std_x_1 = 3.0
    std_x_2 = 1.0

    x1 = std_x_1 .* np.random.randn(n_points)  + -3 * np.ones(n_points)# Feature 1 with std = 3
    x2 = std_x_2 * np.random.randn(n_points)   +  3 * np.ones(n_points)    # Feature 2 with std = 1
    noise = 0.5 .* np.random.randn(n_points)  # Noise
    y = 3.0 .* x1 .+ 2.0 .* x2 .+ noise  # Response variable
    X = hcat(x1, x2)  # Combine features into a matrix
    return X, y
end


function cost_function(b, X, y)
    preds = X * b  # Predictions from the model
    mse = np.mean((preds .- y).^2)  # Mean Squared Error using NumPy
    return mse
end

function normalize_features(X)
    return (X .- mean(X, dims=1)) ./ std(X, dims=1)
end

# Create a grid for the cost function
function create_contour_grid(X, y; n_points=100)
    b1_range = np.linspace(-1000, 1000, n_points)  # Range of β1 using NumPy
    b2_range = np.linspace(-1000, 1000, n_points)  # Range of β2 using NumPy

    # Evaluate the cost function for all combinations of b1 and b2
    Z = np.array([
        cost_function([b1, b2], X, y) for b1 in b1_range, b2 in b2_range
    ])
    return b1_range, b2_range, Z
end

# Generate initial data
n_points = 100
X, y = generate_data(n_points)
X_copy = np.copy(X)
X_norm = normalize_features(X_copy)  # Precompute normalized X

# Initialize Observables for dynamic updates
X_obs = Observable(X)  # Observable for feature matrix
y_obs = Observable(y)  # Observable for response variable

# Create a dynamic grid for the cost function
function update_contour!(X, y, Z_obs)
    _, _, Z = create_contour_grid(X, y)
    Z_obs[] = Z
end

# Create a grid for the cost function
b1_range, b2_range, initial_Z = create_contour_grid(X, y)
Z_obs = Observable(initial_Z)  # Observable for the cost function grid
scatter_positions = Observable(hcat(X_obs[][:, 1], X_obs[][:, 2]))


fig = Figure(resolution=(1200, 600))
ax1 = Axis(fig[1, 1], title="Scatter Plot of X", xlabel="x₁", ylabel="x₂")
ax2 = Axis(fig[1, 2], title="Cost Function Contour", xlabel="β1", ylabel="β2")
xlims!(ax1, -10, 10)  # Set x-axis range
ylims!(ax1, -10, 10)  # Set y-axis range


# Use the Observable in the scatter plot
scatter!(ax1, scatter_positions; color=:blue, markersize=8, label="X Points")
axislegend(ax1)


function custom_purplish_viridis_colormap()
    return cgrad([:black, :purple, "#440154", "#3b528b", "#21908d", "#5dc863", "#fde725"], 
        0.6, categorical=false)
end

contourf!(ax2, b1_range, b2_range, Z_obs, levels=30, colormap=custom_purplish_viridis_colormap()
)
function ease(x, R)
    return sin(π * (x / R))
end
# Animate changes in data
record(fig, "linear_regression_contour_normiiing_cusvirdis.gif", 1:100; framerate=20) do frame
    # Interpolate X from non-normalized to normalized
    t = ease(frame, 100.0)  # Scale t from 0 to 1
    println(t)
    println(X_obs[][1:1])
    println(X_copy[1:1])
    X_obs[] .= (1 - t) .* X_copy .+ t .* X_norm  # Blend non-normalized and normalized X

    scatter_positions[] = hcat(X_obs[][:, 1], X_obs[][:, 2])
    update_contour!(X_obs[], y_obs[], Z_obs)
end

