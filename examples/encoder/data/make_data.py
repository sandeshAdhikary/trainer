import numpy as np
import os
import itertools



def build_grid_dataset(config):
        size = config['size']
        # data: (x,y) positions on the grid
        data = np.array(list(itertools.product(range(size), range(size))))
        np.save(os.path.join(os.path.dirname(__file__), 'data.npy'), data)

if __name__ == "__main__":
    config = {
        'size': 20
    }

    data = build_grid_dataset(config)