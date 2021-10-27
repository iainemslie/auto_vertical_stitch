import os
import glob
import numpy as np
import tifffile
from skimage import exposure

class AutoVerticalStitchFunctions:
    def __init__(self, parameters):
        self.lvl0 = os.path.abspath(parameters["projections_input_dir"])
        self.parameters = parameters
        self.z_dirs = []
        self.ct_dirs = []
        self.ct_stitch_pixel_dict = {}

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
        print(self.ct_stitch_pixel_dict)

        print("--> Stitching Images")
        self.stitch_images()

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
        index = 0
        for ct_dir in self.ct_dirs:
            z_glob_path = os.path.join(ct_dir, 'z??')
            z_list = sorted(glob.glob(z_glob_path))

            # Get list of z-directories within each ct directory
            midpoint_zdir = z_list[int(len(z_list) / 2)]
            one_after_midpoint_zdir = z_list[int(len(z_list) / 2) + 1]
            print("Working on: " + ct_dir)
            print(midpoint_zdir)
            print(one_after_midpoint_zdir)

            # Get the 'middle' z-directories
            midpoint_zdir_tomo = os.path.join(midpoint_zdir, "tomo")
            one_after_midpoint_zdir_tomo = os.path.join(one_after_midpoint_zdir, "tomo")
            # Get the list of images in these 'middle' z-directories
            midpoint_image_list = sorted(glob.glob(os.path.join(midpoint_zdir_tomo, '*.tif')))
            one_after_midpoint_image_list = sorted(glob.glob(os.path.join(one_after_midpoint_zdir_tomo, '*.tif')))

            stitch_pixel_list = []
            # Compute the stitch pixel for every 100th image
            for image_index in range(0, len(midpoint_image_list)+50, 50):
                if image_index > 0:
                    image_index = image_index - 1
                midpoint_first_image_path = midpoint_image_list[image_index]
                one_after_midpoint_first_image_path = one_after_midpoint_image_list[image_index]
                stitch_pixel_list.append(self.compute_stitch_pixel(midpoint_first_image_path,
                                                                   one_after_midpoint_first_image_path))

            print(stitch_pixel_list)
            most_common_value = max(set(stitch_pixel_list), key=stitch_pixel_list.count)
            self.ct_stitch_pixel_dict[ct_dir] = int(most_common_value)
            print("Stitch Pixel: " + str(int(most_common_value)))

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

        #tifffile.imwrite(os.path.join(self.parameters['output_dir'], str(index)+'first.tif'), first)
        #tifffile.imwrite(os.path.join(self.parameters['output_dir'], str(index)+'second.tif'), second)

        # Flip and rotate the images so that they have same orientation as auto_horizontal_stitch
        first = np.rot90(first)
        second = np.rot90(second)
        first = np.fliplr(first)

        # Equalize the histograms and match them so that images are more similar
        first = exposure.equalize_hist(first)
        second = exposure.equalize_hist(second)
        second = exposure.match_histograms(second, first)

        # Threshold the images
        first = first > np.mean(first)
        second = second > np.mean(second)

        #tifffile.imwrite(os.path.join(self.parameters['output_dir'], str(index)+'first_fliprot.tif'), first)
        #tifffile.imwrite(os.path.join(self.parameters['output_dir'], str(index)+'second_fliprot.tif'), second)

        # We must crop the both images from left column of image until overlap region
        first_cropped = first[:, :int(self.parameters['overlap_region'])]
        second_cropped = second[:, :int(self.parameters['overlap_region'])]

        return self.compute_rotation_axis(first_cropped, second_cropped)

    def stitch_images(self):
        for ct_dir in self.ct_dirs:
            print("--> stitching: " + ct_dir)
            z_glob_path = os.path.join(ct_dir, 'z??')
            z_list = sorted(glob.glob(z_glob_path))
            for z_index in range(0, len(z_list) - 1):
                print(z_list[z_index])
                print(z_list[z_index+1])

            first_images = sorted(os.listdir(os.path.join(z_list[0], "tomo")))
            second_images = sorted(os.listdir(os.path.join(z_list[1], "tomo")))

            print(first_images)
            print(second_images)

            z00_im1 = self.read_image(os.path.join(z_list[0], "tomo", first_images[0]), flip_image=False)
            z01_im1 = self.read_image(os.path.join(z_list[1], "tomo", second_images[0]), flip_image=False)

            axis = self.ct_stitch_pixel_dict[ct_dir]
            out_path = self.parameters['output_dir']
            type_str = "tomo"
            out_fmt = os.path.join(out_path, "", type_str + "_stitched_{:>04}.tif".format(0))
            stitched = self.stitch(z00_im1, z01_im1, axis, 0)
            tifffile.imwrite(out_fmt, stitched)


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
    def stitch(self, upper, lower, axis, crop):
        height, width = lower.shape
        if axis > height / 2:
            dy = int(2 * (height - axis) + 0.5)
        else:
            dy = int(2 * axis + 0.5)
            tmp = np.copy(lower)
            lower = upper
            upper = tmp
        result = np.empty((2 * height - dy, width), dtype=lower.dtype)
        ramp = np.linspace(0, 1, dy)

        # Mean values of the overlapping regions must match, which corrects flat-field inconsistency
        # between the two projections
        # We clip the values in upper so that there are no saturated pixel overflow problems
        k = np.mean(lower[height - dy:, :]) / np.mean(upper[:dy, :])
        upper = np.clip(upper * k, np.iinfo(np.uint16).min, np.iinfo(np.uint16).max).astype(np.uint16)

        result[:height - dy, :] = lower[:height - dy, :]
        result[height - dy:height, :] = lower[height - dy:, :] + upper[:dy, :]
        # TODO: Figure out how to add ramp back
        #result[height - dy:height, :] = lower[height - dy:, :] * (1 - ramp) + upper[:dy, :] * ramp
        result[height:, :] = upper[dy:, :]

        return result[slice(int(crop), int(2 * (height - axis) - crop), 1), :]

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
