import numpy as np
import matplotlib.pyplot as plt
import pymannkendall as mk

# The Mann-Kendall test combined with Sen's slope can be employed to detect trends for Temperature vs Time.
# If the trend subsides (becomes non-significant) in the latter part of the time series, 
# it's a sign that the system might be approaching or is in equilibration.

# Here's the methodology:

# Divide the data into segments (e.g., consecutive windows of data).
# For each segment, apply the Mann-Kendall test. 
# If a segment shows no significant trend and Sen's slope is near zero, 
# the system can be considered as approaching equilibration from that point forward.
# Determine the equilibrated temperature as the average of temperatures beyond this point.

# Function to generate random temperature vs. time data
# This function creates a set of synthetic temperature data points for demonstration purposes.
# The data simulates temperature readings over a given number of time points.
def generate_random_data(num_points=1000):
    # Create an array of time points
    time = np.linspace(0, num_points - 1, num_points)
    
    # Generate temperature data with random fluctuations
    # Starting at 20 degrees and adding cumulative noise
    temp = 20 + np.random.normal(0, 0.05, num_points).cumsum()
    return time, temp

# Function to detect equilibration in the data using the Mann-Kendall test and Sen's slope
# It analyzes segments of the temperature data for trends and identifies points where
# the system starts equilibrating.
def detect_equilibration(time, temp, window_size=20):
    equilibration_points = []  # To store potential equilibration points
    slopes = []  # To store Sen's slope values for each segment

    # Iterate through temperature data in segments
    for i in range(0, len(temp) - window_size):
        subset = temp[i:i+window_size]
        
        # Applying the Mann-Kendall trend test to each segment
        mk_result = mk.original_test(subset)
        
        # Computing Sen's slope for the segment
        sen_result = mk.sens_slope(subset)
        slopes.append(sen_result.slope)
        
        # If trend is not significant and Sen's slope is near zero, mark as equilibration point
        if mk_result.p > 0.05 and abs(sen_result.slope) < 0.001:
            equilibration_points.append(i + window_size // 2)
    
    return equilibration_points, slopes

# Main script execution
# Generate random temperature data
time, temp = generate_random_data()

# Detect equilibration points in the generated data
equilibration_points, slopes = detect_equilibration(time, temp)

# Visualization of results
plt.figure(figsize=(15, 7))

# Plotting the temperature data
plt.plot(time, temp, label="Temperature", color="blue")

# Highlighting identified equilibration points on the plot
for point in equilibration_points:
    plt.axvline(x=time[point], color='red', linestyle='--')

# Plotting Sen's slope to visualize its trend over time
plt.plot(time[:-20], slopes, label="Sen's Slope", color="green")  # Exclude last points due to window size

# Setting plot labels and title
plt.xlabel("Time")
plt.ylabel("Temperature / Slope")
plt.title("Temperature vs. Time with Equilibration Points and Sen's Slope")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
