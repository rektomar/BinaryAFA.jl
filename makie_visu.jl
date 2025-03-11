using Makie
using GLMakie

using bmnist_pred

using NPZ
using StaticBitSets


function load_data()
    x_trn = npzread("data/bmnist_trn_x.npy") |> permutedims;
    y_trn = npzread("data/bmnist_trn_y.npy");

    x_tst = npzread("data/bmnist_tst_x.npy") |> permutedims;        
    y_tst = npzread("data/bmnist_tst_y.npy");
    x_trn, x_tst, y_trn, y_tst
end

n_classes = 10
P = 0.01  # kernel width (probability of error)
x_trn, x_tst, y_trn, y_tst = load_data()
N = ceil(Int64, size(x_trn, 1) / 64)
model = EmpiricalModel(x_trn, y_trn, n_classes, P);

reshape_img(img_flat, shape=(28, 28)) = reshape(img_flat, shape)'
pixel_position(x, y; width=28, height=28) = width*(y-1) + x
function feature2pixel(f; width=28, height=28) 
    fdw= f ÷ width
    y= fdw+1
    x = f-fdw*width
    x,y
end


function create_mnist_grid(train_x, train_y)
    # Initialize empty image with white background
    grid_width = 10 * 28 + 9 * 5   # 10 digits + 9 padding spaces
    grid_height = 3 * 28 + 2 * 5   # 3 samples + 2 padding spaces
    grid_image = ones(grid_height, grid_width)
    
    for digit in 0:9
        class_samples = findall(x -> x == digit, train_y)
        selected_samples = class_samples[1:3]
        
        # Calculate the starting column for this digit
        col_start = digit * (28 + 5) + 1
        
        for (i, sample_idx) in enumerate(selected_samples)
            # Calculate row position (top to bottom)
            row_start = (i-1) * (28 + 5) + 1
            
            # Copy the digit image into the grid
            grid_image[row_start:row_start+27, col_start:col_start+27] = 
                reshape_img(train_x[:,sample_idx])
        end
    end
    return grid_image
end

function create_interactive_mnist_viz()
    # Load MNIST data
    x_trn, x_tst, y_trn, y_tst = load_data()
    test_image = reshape_img(x_tst[:,1])  # Get first test image, properly oriented

    # Create figure with layout
    fig = Figure(resolution=(1000, 800))
    
    # Top part: MNIST samples (3 samples for each class 0-9)
    mnist_grid = create_mnist_grid(x_trn, y_trn)
    ax_mnist = Axis(fig[1:3, 1:10],
                aspect=DataAspect(), yreversed=true,
                title="MNIST Samples (3 per digit)")
    image!(ax_mnist, mnist_grid')
    hidedecorations!(ax_mnist)


    # Bottom left: Interactive 28x28 grid
    ax_grid = Axis(fig[4:7, 1:5],
                   aspect=1, yreversed=true,
                   title="Interactive Grid")
    deregister_interaction!(ax_grid, :rectanglezoom)

    # Create empty 28x28 grid
    grid_data = 0.5*ones(28,28) #test_image; #zeros(28, 28)
    grid_image = Observable(grid_data)
    hm=image!(ax_grid, grid_image, colormap=:grays, interpolate=false)
    m = SBitSet{N, UInt64}()    # initialize empty mask
    xtest = SBitSet{N, UInt64}(findall(x_tst[:,1] .== 1))   # take the first test set observation


    # Add click interaction
    on(events(fig).mousebutton) do event
        if event.button == Mouse.left
            # Get mouse position in axis coordinates
            pos = mouseposition(ax_grid)
            if !isnothing(pos)
                x, y = pos
                # Convert to grid coordinates (1-28)
                grid_x = round(Int, x)
                grid_y = round(Int, y)
                if 1 ≤ grid_x ≤ 28 && 1 ≤ grid_y ≤ 28
                    add_new(grid_y,grid_x)
                end
            end
        end
        return Consume(false)
    end

    function add_new(grid_x,grid_y)
        grid_data[grid_x, grid_y] = test_image[grid_x, grid_y]
        # Update the heatmap
        grid_image[]=grid_data'
        println("Clicked cell: ($grid_x, $grid_y)")

        feature_id = pixel_position(grid_x, grid_y)
        m = push(m, feature_id)         # add feature index to mask
        pp,yp = predict(model,xtest, m)
        prob_x[] = pp;
    end

    bu_auto = Button(fig[4, 6:7], label="Auto")
    bu_clr = Button(fig[4,9:10], label="Clear")
    bu_sho = Button(fig[4,8], label="Show")

    # Connect the button to your function
    on(bu_auto.clicks) do n
        feature_id = afa_step(model, xtest, m)
        grid_x,grid_y= feature2pixel(feature_id);
        add_new(grid_x,grid_y)

        println("Proposed cell: ($grid_x, $grid_y)")

        # add_new(1,1)
    end

    on(bu_clr.clicks) do n
        m = SBitSet{N, UInt64}()    # initialize empty mask
        grid_data = 0.5*ones(28,28) #test_image; #zeros(28, 28)
        println("Clicked clear")
        grid_image[]=grid_data
        prob_x[] = 0.1*ones(10);

        # add_new(1,1)
    end

    on(bu_sho.clicks) do n
        grid_data = test_image
        grid_image[]=grid_data'
        # add_new(1,1)
    end
    

    # Bottom right: Stem plot of probabilities
    ax_stem = Axis(fig[5:7, 6:10],
                   title="Class Probabilities",
                   xlabel="Class",
                   ylabel="Probability")
    
    # Generate random probabilities (uniform distribution)
    probs = ones(10)
    probs = probs ./ sum(probs)  # Normalize to sum to 1
    
    prob_x = Observable(0.1*ones(10))
    # Create stem plot
    stem!(ax_stem, 0:9, prob_x)
    
    # Set axis limits
    xlims!(ax_stem, -0.5, 9.5)
    ylims!(ax_stem, 0, 1.0)
    
    return fig
end

# Create and display the visualization
fig = create_interactive_mnist_viz()
display(fig)
