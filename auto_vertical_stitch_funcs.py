

class AutoVerticalStitchFunctions:
    def __init__(self, parameters):
        self.parameters = parameters

    def run_vertical_auto_stitch(self):
        """
        Main function that calls all other functions
        """

        self.print_parameters()

    def print_parameters(self):
        """
        Prints parameter values with line formatting
        """
        print()
        print("**************************** Running Auto Vertical Stitch ****************************")
        print("======================== Parameters ========================")
        print("Input Directory: " + self.parameters['input_dir'])
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
        print("============================================================")
