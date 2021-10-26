import os
import glob
import numpy as np
import tifffile

class AutoVerticalStitchFunctions:
    def __init__(self, parameters):
        self.lvl0 = os.path.abspath(parameters["projections_input_dir"])
        self.parameters = parameters
        self.z_dirs = []
        self.ct_dirs = []

    def run_vertical_auto_stitch(self):
        """
        Main function that calls all other functions
        """

        self.print_parameters()

        # Check input directory and find structure
        print("--> Finding Z-Directories")
        self.find_z_dirs()
        print(self.z_dirs)

        # Determine CT-directories from list of z-directories
        print("--> Finding CT-Directories")
        self.find_ct_dirs()
        print(self.ct_dirs)

        print("--> Finding Stitch Index")
        self.find_stitch_pixel()

    def find_z_dirs(self):
        """
        Walks directories rooted at "Input Directory" location
        Appends their absolute path to ct-dir if they contain a directory with same name as "tomo" entry in GUI
        :return: Sets a list of z-directory paths in class member self.z_dirs
        """
        for root, dirs, files in os.walk(self.lvl0):
            for name in dirs:
                if name == "tomo":
                    self.z_dirs.append(root)
        self.z_dirs = sorted(list(set(self.z_dirs)))

    def find_ct_dirs(self):
        """
        Gets absolute path to parent directories containing z-subdirectories.
        :return: Sets a list of ct-directory paths in class member self.ct_dirs
        """
        temp_ct_dirs = []
        for z_path in self.z_dirs:
            ct_dir_path, zdir = os.path.split(z_path)
            temp_ct_dirs.append(ct_dir_path)
        self.ct_dirs = sorted(list(set(temp_ct_dirs)))

    def find_stitch_pixel(self):
        """
        Looks at each ct-directory, finds the midpoint z-directory and it's successor
        We then use images from the "tomo" directory to determine the point of overlap
        """
        for ct_dir in self.ct_dirs:
            z_glob_path = os.path.join(ct_dir, 'z??')
            z_list = sorted(glob.glob(z_glob_path))

            # Get list of z-directories within each ct directory
            midpoint_zdir = z_list[int(len(z_list) / 2)]
            one_after_midpoint_zdir = z_list[int(len(z_list) / 2) + 1]
            print(midpoint_zdir)
            print(one_after_midpoint_zdir)

            # Get the 'middle' z-directories
            midpoint_zdir_tomo = os.path.join(midpoint_zdir, "tomo")
            one_after_midpoint_zdir_tomo = os.path.join(one_after_midpoint_zdir, "tomo")
            # Get the list of images in these 'middle' z-directories
            midpoint_image_list = sorted(glob.glob(os.path.join(midpoint_zdir_tomo, '*.tif')))
            one_after_midpoint_image_list = sorted(glob.glob(os.path.join(one_after_midpoint_zdir_tomo, '*.tif')))

            midpoint_first_image_path = midpoint_image_list[0]
            one_after_midpoint_first_image_path = one_after_midpoint_image_list[0]

            self.compute_stitch_pixel(midpoint_first_image_path, one_after_midpoint_first_image_path)

    def compute_stitch_pixel(self, upper_image, lower_image):
        """
        Takes two pairs of images with vertical overlap, determines the point at which to stitch the images
        :return:
        """

        # Read in the images to numpy array
        first = self.read_image(upper_image, False)
        second = self.read_image(lower_image, False)

        # Do flat field correction using flats/darks directory in same ctdir as input images
        tomo_path, filename = os.path.split(upper_image)
        zdir_path, tomo_name = os.path.split(tomo_path)
        flats_path = os.path.join(zdir_path, "flats")
        darks_path = os.path.join(zdir_path, "darks")
        flat_files = self.get_filtered_filenames(flats_path)
        dark_files = self.get_filtered_filenames(darks_path)

        flats = np.array([tifffile.TiffFile(x).asarray().astype(np.float) for x in flat_files])
        darks = np.array([tifffile.TiffFile(x).asarray().astype(np.float) for x in dark_files])
        dark = np.mean(darks, axis=0)
        flat = np.mean(flats, axis=0) - dark
        first = (first - dark) / flat
        second = (second - dark) / flat

        # We must flip the second image to that it has the same orientation as the first
        second = np.flipud(second)

        # We must crop the both images from bottom of image until overlap region
        first_cropped = first[:-int(self.parameters['overlap_region']), :]
        second_cropped = second[:-int(self.parameters['overlap_region']), :]

        stitch_pixel = self.compute_rotation_axis(first_cropped, second_cropped)

        print(stitch_pixel)


    def print_parameters(self):
        """
        Prints parameter values with line formatting
        """
        print()
        print("**************************** Running Auto Vertical Stitch ****************************")
        print("======================== Parameters ========================")
        print("Projections Input Directory: " + self.parameters['projections_input_dir'])
        print("Reconstructed Slices Input Directory: " + self.parameters['recon_slices_input_dir'])
        print("Output Directory: " + self.parameters['output_dir'])
        print("Using common set of flats and darks: " + str(self.parameters['common_flats_darks']))
        print("Flats Directory: " + self.parameters['flats_dir'])
        print("Darks Directory: " + self.parameters['darks_dir'])
        print("Sample moved down: " + str(self.parameters['sample_moved_down']))
        print("Overlap Region Size: " + self.parameters['overlap_region'])
        print("Stitch Reconstructed Slices: " + str(self.parameters['stitch_reconstructed_slices']))
        print("Stitch Projections: " + str(self.parameters['stitch_projections']))
        print("Equalize Intensity: " + str(self.parameters['equalize_intensity']))
        print("Concatenate: " + str(self.parameters['concatenate']))
        print("Which images to stitch - start,stop,step: " + str(self.parameters['images_to_stitch']))
        print("Dry Run: " + str(self.parameters['dry_run']))
        print("============================================================")

    def get_filtered_filenames(self, path, exts=['.tif', '.edf']):
        result = []

        try:
            for ext in exts:
                result += [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]
        except OSError:
            return []

        return sorted(result)

    def compute_rotation_axis(self, first_projection, last_projection):
        """
        Compute the tomographic rotation axis based on cross-correlation technique.
        *first_projection* is the projection at 0 deg, *last_projection* is the projection
        at 180 deg.
        """
        from scipy.signal import fftconvolve
        width = first_projection.shape[1]
        first_projection = first_projection - first_projection.mean()
        last_projection = last_projection - last_projection.mean()

        # The rotation by 180 deg flips the image horizontally, in order
        # to do cross-correlation by convolution we must also flip it
        # vertically, so the image is transposed and we can apply convolution
        # which will act as cross-correlation
        convolved = fftconvolve(first_projection, last_projection[::-1, :], mode='same')
        center = np.unravel_index(convolved.argmax(), convolved.shape)[1]

        return (width / 2.0 + center) / 2

    """****** BORROWED FUNCTIONS ******"""

    def read_image(self, file_name, flip_image):
        """
        Reads in a tiff image from disk at location specified by file_name, returns a numpy array
        :param file_name: Str - path to file
        :param flip_image: Bool - Whether image is to be flipped horizontally or not
        :return: A numpy array of type float
        """
        with tifffile.TiffFile(file_name) as tif:
            image = tif.pages[0].asarray(out='memmap')
        if flip_image is True:
            image = np.fliplr(image)
        return image