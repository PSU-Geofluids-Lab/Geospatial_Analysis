import os
import numpy as np
import h5py
import csv
from abc import ABC, abstractmethod
from Plotting import ImagePlotter
import matplotlib.pyplot as plt
import porespy as ps

class BaseGenerator(ABC):
    """Abstract base class for 2D image generators"""
    def __init__(self, size=(256, 256)):
        self.size = size
        self.data = None
        self.name = self.__class__.__name__
        self.metadata = {
            'generator_type': self.__class__.__name__,
            'size': size
        }
        results_folder = os.path.join("Results", self.name)
        os.makedirs(results_folder, exist_ok=True)
        self.full_path = results_folder

    @abstractmethod
    def generate(self, *args, **kwargs):
        """Generate image data (must be implemented by subclasses)"""
        pass

    def add_metadata(self, key, value):
        """Add custom metadata"""
        self.metadata[key] = value

    def to_csv(self):
        filename = f"{self.full_path}/Generated_Data.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['# ' + f"{k}: {v}" for k, v in self.metadata.items()])
            np.savetxt(f, self.data, delimiter=',')

    def to_png(self):
        filename = f"{self.full_path}/Generated_Data.png"
        ImagePlotter.plot(self.data, save_path=filename)
        filename = f"{self.full_path}/Generated_Data_NoFrills.png"
        ImagePlotter.plot_Nofrills(self.data, save_path=filename)

    def to_hdf5(self):
        filename = f"{self.full_path}/Generated_Data.h5"
        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset('texture', data=self.data)
            for k, v in self.metadata.items():
                dset.attrs[k] = v

    def save_all(self):
      
      self.to_csv()
      self.to_png()
      self.to_hdf5()
      print('All Files saved')


    def make_plot_fractal(self,im,filepath=None):
        """
        Compute the box counting dimension of a binary image and plot it.

        Parameters
        ----------
        im : ndarray
            2D image
        filepath : str, optional
            Path to save the figure. If None, the figure is not saved and is shown instead.

        Returns
        -------
        data : named tuple
            Contains size and count of the boxes, as well as the slope of the
            log-log plot.

        Notes
        -----
        The box counting dimension is calculated as the negative slope of the
        log-log plot of the number of boxes spanning the image vs the box edge
        length.

        """
        if np.unique(im).shape[0] > 2:
            raise ValueError('The image must be binary')
        data = ps.metrics.boxcount(im)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_xlabel('box edge length')
        ax1.set_ylabel('number of boxes spanning phases')
        ax2.set_xlabel('box edge length')
        ax2.set_ylabel('slope')
        ax2.set_xscale('log')
        ax1.plot(data.size, data.count,'-o')
        ax2.plot(data.size, data.slope,'-o')
        if filepath is not None :
            plt.savefig(f'{filepath}/Fractal_dimension.png',bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        return data


    def two_pt_corr(self,im,filepath=None):
        """
        Plot the two-point correlation function of an image.

        Parameters
        ----------
        im : ndarray
            A 2D image of the porous material.

        Returns
        -------
        data : tuple
            A tuple containing the distance and probability arrays from the two-point
            correlation function calculation.

        Notes
        -----
        The two-point correlation function is calculated using Porespy's
        two_point_correlation function.
        """
        data = ps.metrics.two_point_correlation(im)
        fig, ax = plt.subplots(1, 1, figsize=[6, 6])
        ax.plot(data.distance, data.probability, 'r.')
        ax.set_xlabel("distance")
        ax.set_ylabel("two point correlation function")
        if filepath is not None :
            plt.savefig(f'{filepath}/2pt_correlation.png',bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        return data
        

    def make_save_metrics(self):
        """
        Calculate and save all the metrics for the generated image.

        -----
        The function calculates the following metrics and saves them to files:
            - Fractal dimension
            - 2-point correlation
        """
        self.fractal_data = self.make_plot_fractal(self.data,filepath=self.full_path)
        self.two_pt_corr = self.two_pt_corr(self.data,filepath=self.full_path)

        print('Done the metrics : fractal,' \
                    '2pt correlation')