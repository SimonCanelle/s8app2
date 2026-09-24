import numpy as np


def transmit(Data):

    # Initialize budget
    budget_data = 0

    # Loop through all cells
    for i, data in enumerate(Data, start=1):

        # Empty MATLAB cells
        if data is None:
            continue

        data = np.asarray(data)

        # Check if the data is below the dynamic range
        if np.min(data) < 0:
            return -i

        # Check if the data is above the dynamic range
        if np.max(data) > 2**i - 1:
            return -i

        # Calculate the amount of transmitted data
        budget_data += data.size * i

    return budget_data