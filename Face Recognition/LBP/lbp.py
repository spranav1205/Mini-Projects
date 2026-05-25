import numpy as np

neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1),
        (1, 1), (1, 0), (1, -1),
        (0, -1)
    ]

class LBP:

    def __init__(self, neighbors=8, window_size=24, pieces=2, num_bins=64
    ):
        self.neighbors = neighbors
        self.window_size = window_size
        self.pieces = pieces
        self.num_bins = num_bins

    def extract_feature(self, image, i, j):

        lbp_image = np.zeros((self.window_size-2, self.window_size-2), dtype=np.uint8)

        piece_h = (self.window_size - 2) // self.pieces
        piece_w = (self.window_size - 2) // self.pieces

        for x in range(1, self.window_size - 1):
            for y in range(1, self.window_size - 1):
                center = image[i + x, j + y]
                value = 0
                for bit, (dx, dy) in enumerate(neighbors):
                    if image[i + x + dx, j + y + dy] >= center:
                        value |= (1 << bit) # Set the bit at position 'bit' to 1
                lbp_image[x-1, y-1] = value

        histograms = []
        for px in range(self.pieces):
            for py in range(self.pieces):

                piece = lbp_image[
                    px * piece_h:(px + 1) * piece_h,
                    py * piece_w:(py + 1) * piece_w
                ]

                hist, _ = np.histogram(
                    piece,
                    bins=self.num_bins,
                    range=(0, self.num_bins)
                )

                histograms.append(hist)

        return np.array(histograms).flatten().astype(np.float32)