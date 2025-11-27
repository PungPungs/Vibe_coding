import segyio
import numpy as np
import traceback

class SegyReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.coords = None

    def read_coords(self):
        """
        Reads Source X and Source Y coordinates from the SEG-Y file.
        Returns a numpy array of shape (N, 2) where N is the number of traces.
        """
        # Try different endianness
        endians = ['big', 'little']
        
        for endian in endians:
            try:
                print(f"Trying to open {self.filepath} with endian='{endian}'...")
                with segyio.open(self.filepath, ignore_geometry=True, endian=endian) as f:
                    # Extract Source X (byte 73) and Source Y (byte 77)
                    # segyio.TraceField.SourceX is 73
                    # segyio.TraceField.SourceY is 77
                    
                    source_x = f.attributes(segyio.TraceField.SourceX)[:]
                    source_y = f.attributes(segyio.TraceField.SourceY)[:]
                    
                    # Handle scalar (byte 71)
                    scalars = f.attributes(segyio.TraceField.ElevationScalar)[:]
                    
                    # Create a float array for coordinates
                    x = source_x.astype(np.float64)
                    y = source_y.astype(np.float64)
                    
                    # Apply scalar
                    if len(scalars) > 0:
                        # Vectorized scalar application
                        # Positive scalars: multiplier
                        pos_mask = scalars > 0
                        x[pos_mask] *= scalars[pos_mask]
                        y[pos_mask] *= scalars[pos_mask]
                        
                        # Negative scalars: divisor
                        neg_mask = scalars < 0
                        x[neg_mask] /= np.abs(scalars[neg_mask])
                        y[neg_mask] /= np.abs(scalars[neg_mask])

                    self.coords = np.column_stack((x, y))
                    print(f"Successfully read {len(self.coords)} traces.")
                    return self.coords
            except Exception as e:
                print(f"Failed with endian='{endian}': {e}")
                # traceback.print_exc()
                continue
        
        print(f"Could not read {self.filepath} with any standard configuration.")
        return None
