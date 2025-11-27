import segyio
import numpy as np
import traceback
import os

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

    def split_file(self, interval_meters, output_dir):
        """
        Splits the SEG-Y file into multiple files based on distance interval.
        """
        if self.coords is None:
            self.read_coords()
            
        if self.coords is None:
            return False, "Could not read coordinates."

        try:
            # Determine endianness again (or store it from read_coords)
            # For simplicity, we'll try to detect it again or just try 'big' then 'little'
            # Ideally we should store the successful endianness in self.endian
            
            # Let's just try to open it again to get the spec
            endian = 'big'
            try:
                with segyio.open(self.filepath, ignore_geometry=True, endian='big') as f:
                    f.tracecount
            except:
                endian = 'little'
                
            with segyio.open(self.filepath, ignore_geometry=True, endian=endian) as src:
                # Calculate cumulative distance
                # coords is (N, 2)
                # dists[i] is distance between point i and i+1
                dists = np.sqrt(np.sum(np.diff(self.coords, axis=0)**2, axis=1))
                cum_dists = np.cumsum(dists)
                cum_dists = np.insert(cum_dists, 0, 0) # Insert 0 at start
                
                # Identify split points
                # We want to split when cum_dist crosses k * interval
                
                split_indices = [0]
                current_dist_threshold = interval_meters
                
                for i, d in enumerate(cum_dists):
                    if d >= current_dist_threshold:
                        split_indices.append(i) # i is the index of the trace AFTER the cut?
                        # If d is exactly threshold, we include i. 
                        # If d > threshold, we include i.
                        # We want the chunk to be roughly interval_meters.
                        # So we cut at i.
                        current_dist_threshold += interval_meters
                
                split_indices.append(len(self.coords))
                
                base_name = os.path.splitext(os.path.basename(self.filepath))[0]
                created_files = []
                
                for i in range(len(split_indices) - 1):
                    start_idx = split_indices[i]
                    end_idx = split_indices[i+1]
                    
                    if start_idx >= end_idx:
                        continue
                        
                    part_name = f"{base_name}_part{i+1}.sgy"
                    out_path = os.path.join(output_dir, part_name)
                    
                    # Create new file
                    # We need to copy spec
                    spec = segyio.tools.metadata(src)
                    spec.tracecount = end_idx - start_idx
                    
                    print(f"Creating {out_path} with {spec.tracecount} traces...")
                    
                    with segyio.create(out_path, spec) as dst:
                        # Copy text header
                        dst.text[0] = src.text[0]
                        
                        # Copy binary header
                        dst.bin = src.bin
                        
                        # Copy traces and headers
                        # This might be slow for large files, can be optimized
                        dst.trace[:] = src.trace[start_idx:end_idx]
                        dst.header[:] = src.header[start_idx:end_idx]
                        
                    created_files.append(out_path)
                    
                return True, f"Split into {len(created_files)} files."
                
        except Exception as e:
            traceback.print_exc()
            return False, str(e)

    def extract_segment(self, start_idx, end_idx, output_path):
        """
        Extracts a segment of traces from start_idx to end_idx (inclusive) to a new file.
        """
        try:
            # Determine endianness
            endian = 'big'
            try:
                with segyio.open(self.filepath, ignore_geometry=True, endian='big') as f:
                    f.tracecount
            except:
                endian = 'little'
                
            with segyio.open(self.filepath, ignore_geometry=True, endian=endian) as src:
                spec = segyio.tools.metadata(src)
                spec.tracecount = end_idx - start_idx + 1
                
                print(f"Extracting segment {start_idx}-{end_idx} to {output_path}...")
                
                with segyio.create(output_path, spec) as dst:
                    # Copy text header
                    dst.text[0] = src.text[0]
                    
                    # Copy binary header
                    dst.bin = src.bin
                    
                    # Copy traces and headers
                    # +1 because end_idx is inclusive in our logic but slice is exclusive at end
                    dst.trace[:] = src.trace[start_idx : end_idx + 1]
                    dst.header[:] = src.header[start_idx : end_idx + 1]
                    
            return True, "Segment extracted successfully."
            
        except Exception as e:
            traceback.print_exc()
            return False, str(e)
