import os
import glob
import time
import nrrd
import imageio
import numpy as np
import scipy.io as sio
from scipy import ndimage


def setWorkspace(path):
    """
    Has to be called before using any of the following classes.
    :param path: object instance of class Path used to set the path attribute of each class. The path object contains
    paths to all subfolders.
    """
    IniFile.path = path
    Image.path = path
    CMTK.path = path
    Region.path = path
    Coordinates.path = path


class Path:

    def __init__(self, path, root=None):
        """
        Creates a path for the main folder (center of operations) and all subfolders for use with the classes
        defined below. Single fish registrations should be made within a sub-folder with root name.

        :param path: String, path to the main folder. This folder should remain untouched and likely contains many
        different root folders.
        :param root: String, name of the sub-folder. Leave blank to merge all files in one single folder, which is
        useful for debugging.
        """
        if root is None:
            self.main = path
            self.imgFolder = path
            self.xformFolder = path
            self.txtFolder = path
            self.gifFolder = path
        else:
            self.main = path
            self.imgFolder = '{}{}'.format(path, '{}/'.format(root))
            self.xformFolder = '{}{}'.format(path, '{}/Transformations/'.format(root))
            self.txtFolder = '{}{}'.format(path, '{}/TextFiles/'.format(root))
            self.gifFolder = '{}{}'.format(path, '{}/Gifs/'.format(root))

    def createFolders(self):
        """
        For use with a root folder. Creates sub-folders to store registration output files.
        :return:
        """
        if not os.path.exists(self.imgFolder):
            os.makedirs(self.imgFolder)
        if not os.path.exists(self.xformFolder):
            os.makedirs(self.xformFolder)
        if not os.path.exists(self.txtFolder):
            os.makedirs(self.txtFolder)
        if not os.path.exists(self.gifFolder):
            os.makedirs(self.gifFolder)


class IniFile:
    """
    Work in progress.
    """
    path = None

    def __init__(self):
        self.path = glob.glob(os.path.join(Image.path.main, '**/*.ini'), recursive=True)[0]
        data = IniFile.importData(self.path)
        self.spacing = np.array([data['x.pixel.sz']/0.000001, data['y.pixel.sz']/0.000001, data['z.spacing']])

    @staticmethod
    def importData(path):
        # Function made by Renaud Bernatchez
        dataDict = {}
        properties = ['x.pixel.sz',
                      'y.pixel.sz',
                      'z.spacing']
        file = open(path)
        data = file.read().split('\n')
        for i, line in enumerate(data):
            lineData = line.split(' ')
            if lineData[0] in properties:
                try:
                    dataDict[lineData[0]] = float(lineData[-1])
                except ValueError:
                    dataDict[lineData[0]] = ' '.join(lineData[2:])
        file.close()
        return dataDict


class Image:

    path = None

    def __init__(self, fileName, spacing, imgFolder=False):
        self.path = Image.path
        self.scale = np.array([1, 1, 1])
        self.fileName = fileName
        self.spacing = spacing
        if imgFolder is False:
            self.array = Image.load(Image.path.main, fileName)
        else:
            self.array = Image.load(Image.path.imgFolder, fileName)
        self.shape = self.array.shape
        self.centerOfMass = ndimage.measurements.center_of_mass(self.array)

    def __getitem__(self, key):
        return self.array[key]

    def changeScale(self, scale):
        # Changes the X and Y pixel dimensions
        if scale == 1:
            pass
        else:
            self.array = ndimage.zoom(self.array, (scale, scale, 1))
            self.scale = np.array([self.scale[0] * scale, self.scale[1] * scale, self.scale[2]])
        self.shape = self.array.shape
        self.centerOfMass = ndimage.measurements.center_of_mass(self.array)

    def resetScale(self):
        # Reloads the image
        self.array = np.transpose(nrrd.read('{}{}'.format(self.path.main, self.fileName))[0], (1, 0, 2))
        self.shape = self.array.shape
        self.centerOfMass = ndimage.measurements.center_of_mass(self.array)
        self.scale = np.array([1, 1, 1])

    def stretchZAxisWithEmptyPlanes(self, scale):
        """
        Adds blank planes between all current planes along the Z axis. Used to facilitate registration by making film
        stacks bigger. This can compensate for different plane spacings in the 3D registration volume.
        :param scale: integer. Stretches the stack by this factor. For example, a stack of 3 planes with scale 7 will
        occupy 2*7+1=15 planes.
        :return: Z-inlated image with empty planes (zeros) in between the pre-existing planes.
        """
        array = np.zeros((self.shape[0], self.shape[1], (self.shape[2]-1)*scale+1))
        for i in range(self.shape[2]):
            array[:, :, i*scale] = self.array[:, :, i]
        self.array = array
        self.shape = self.array.shape

    def removePlanes(self, scale):
        newArray = np.zeros((self.shape[0], self.shape[1], int(((self.shape[2]-1)/scale)+1)))
        for i in range(newArray.shape[2]):
            newArray[:, :, i] = self.array[:, :, i * scale]
        self.array = newArray


    def save(self, fileName):
        # Saves the image as a .nrrd file
        nrrd.write('{}{}'.format(self.path.imgFolder, fileName), np.transpose(self.array, (1, 0, 2)))

    @staticmethod
    def load(path, fileName):
        initial = nrrd.read('{}{}'.format(path, fileName))[0]
        if len(initial.shape) == 2:
            image = np.zeros((initial.shape[0], initial.shape[1], 1))
            image[:, :, 0] = initial
        else:
            image = initial
        image = np.transpose(image, (1, 0, 2))
        return image

    @staticmethod
    def makeGif(image1, image2, fileName, ratio=0.5, RGB=True, frameLength=0.05):
        images = []
        for z in range(image1.shape[2]):
            if RGB is True:
                images.append(
                    ratio * Image.convertToRGB(image1[:, :, z]) + (1-ratio) * Image.convertToRGB(image2[:, :, z],
                                                                                                 color='red'))
            else:
                images.append(
                    ratio * image1[:, :, z] + (1 - ratio) * image2[:, :, z])

        imageio.mimsave('{}{}'.format(Image.path.gifFolder, fileName), images, duration=frameLength)


    @staticmethod
    def convertToRGB(image, color=None):
        """
        Converts a grayscale image to RGB with values ranging from 0 to 1.
        :param image: 2D grayscale numpy array.
        :param color: Color of the output image. By default, only duplicates the image in all 3 channels to make a gray
        image in RGB format.
        :return: a 3-channel RGB image with the specified color.
        """
        if image.max() > 0:
            image = image / image.max()
        if color == 'red':
            RGB = np.zeros((image.shape[0], image.shape[1], 3))
            RGB[:, :, 0] = image
        elif color == 'green':
            RGB = np.zeros((image.shape[0], image.shape[1], 3))
            RGB[:, :, 1] = image
        elif color == 'blue':
            RGB = np.zeros((image.shape[0], image.shape[1], 3))
            RGB[:, :, 2] = image
        else:
            RGB = np.repeat(image[:, :, np.newaxis], repeats=3, axis=2)
        return RGB


class CMTK:

    path = None

    def __init__(self):
        self.path = CMTK.path
        self.registrationTime = None
        self.warpTime = None
        self.scale = None
        self.translation = None

    def alignImages(self):
        os.system('cmtk make_initial_affine --centers-of-mass {0}fixed.nrrd {0}moving.nrrd '
                  '{1}initial.xform'.format(self.path.imgFolder, self.path.xformFolder))
        pass

    def registration(self):
        start = time.time()
        os.system('cmtk registration -o {1}affine.xform --nmi --initial {1}initial.xform --histogram-equalization-flt --histogram-equalization-ref --sampling 4 --omit-original-data --auto-multi-levels 3 --dofs 6,12 {0}fixed.nrrd {0}moving.nrrd'.format(self.path.imgFolder, self.path.xformFolder))
        end = time.time()
        self.registrationTime = end - start

    def warp(self):
        start = time.time()
        os.system('cmtk warp -o {1}warp.xform --initial {1}affine.xform --fast --refine 3 --grid-spacing 40 --jacobian-weight 0.001 --coarsest 6 --sampling 3 --accuracy 3 --omit-original-data --histogram-equalization-ref --histogram-equalization-flt {0}fixed.nrrd {0}moving.nrrd'.format(self.path.imgFolder, self.path.xformFolder))
        end = time.time()
        self.warpTime = end - start

    def reformat(self, reformatType, outputName=None):
        if outputName is None:
            if reformatType == 'initial':
                os.system('cmtk reformatx -o {0}reformatInitial.nrrd --floating {0}moving.nrrd {0}fixed.nrrd {1}initial.xform'.format(self.path.imgFolder, self.path.xformFolder))
            elif reformatType == 'registered':
                os.system('cmtk reformatx -o {0}reformatReg.nrrd --floating {0}moving.nrrd {0}fixed.nrrd {1}affine.xform'.format(self.path.imgFolder, self.path.xformFolder))
            elif reformatType == 'warped':
                os.system('cmtk reformatx -o {0}reformatWarp.nrrd --floating {0}moving.nrrd {0}fixed.nrrd {1}warp.xform'.format(self.path.imgFolder, self.path.xformFolder))
        else:
            if reformatType == 'initial':
                os.system('cmtk reformatx -o {0}{2} --floating {0}moving.nrrd {0}fixed.nrrd {1}initial.xform'.format(self.path.imgFolder, self.path.xformFolder, outputName))
            elif reformatType == 'registered':
                os.system('cmtk reformatx -o {0}{2} --floating {0}moving.nrrd {0}fixed.nrrd {1}affine.xform'.format(self.path.imgFolder, self.path.xformFolder, outputName))
            elif reformatType == 'warped':
                os.system('cmtk reformatx -o {0}{2} --floating {0}moving.nrrd {0}fixed.nrrd {1}warp.xform'.format(self.path.imgFolder, self.path.xformFolder, outputName))

    def transform(self, fileName, outName, inverse=False, mainFolder=False):
        # Standard: from moving to fixed.
        # Inverse: from fixed to moving.
        #
        # The mainFolder option uses input and output text files located in the /Main folder and is used for
        # transforming neuron centroids obtained from segmentation of the films.

        if mainFolder:
            if inverse:
                os.system('cmtk streamxform {3}affine.xform <{0}{1}> {0}{2}'.format(self.path.main, fileName, outName, self.path.xformFolder))
            else:
                os.system('cmtk streamxform -- --inverse {3}affine.xform <{0}{1}> {0}{2}'.format(self.path.main, fileName, outName, self.path.xformFolder))
        else:
            if inverse:
                os.system('cmtk streamxform {3}affine.xform <{0}{1}> {0}{2}'.format(self.path.txtFolder, fileName, outName, self.path.xformFolder))
            else:
                os.system('cmtk streamxform -- --inverse {3}affine.xform <{0}{1}> {0}{2}'.format(self.path.txtFolder, fileName, outName, self.path.xformFolder))

    def computeScaleAndTranslation(self, fixed, moving, customScale=None):
        if customScale is None:
            self.scale = [fixed.spacing[0] / moving.spacing[0], fixed.spacing[1] / moving.spacing[1]]
        else:
            self.scale = customScale
        self.translation = [(moving.centerOfMass[1] / self.scale[0] - fixed.centerOfMass[1]) * self.scale[0],
                            (moving.centerOfMass[0] / self.scale[1] - fixed.centerOfMass[0]) * self.scale[1]]

    def scaleInitialXFormFile(self):
        f = open('{0}initial.xform'.format(self.path.xformFolder), 'r')
        lines = f.readlines()
        line3 = lines[3].split(' ')
        lines[3] = '\txlate {} {} {} \n'.format(self.translation[0], self.translation[1], line3[3])
        lines[5] = '\tscale {} {} 1 \n'.format(self.scale[0], self.scale[1])
        f = open('{0}initial.xform'.format(self.path.xformFolder), 'w')
        for line in lines:
            f.write(line)
        f.close()


class Region:

    path = None

    def __init__(self, ID, image):
        self.path = Region.path
        self.ID = ID
        self.name = sio.loadmat('Names.mat')['MaskDatabaseNames'][0, ID-1][0]
        self.scale = image.scale
        data = sio.loadmat('Regions.mat')
        self.centroid = data['centroids'][ID-1, :]
        self.maxZ_transformed = None
        self.minZ_transformed = None

    def findUpperAndLowerBoundaries(self):
        f = open('{0}transformedCoordinates.txt'.format(self.path.txtFolder), 'r')
        lines = f.readlines()
        positions = []
        for line in lines:
            pieces = line.split(' ')
            positions.extend([float(pieces[2])])
        self.maxZ_transformed = np.max(positions)
        self.minZ_transformed = np.min(positions)
        f = open('RegionBoundaries/coordinates{0}.txt'.format(self.ID), 'r')
        lines = f.readlines()
        positions = []
        for line in lines:
            pieces = line.split(' ')
            positions.extend([float(pieces[2])])
        self.maxZ = np.max(positions)
        self.minZ = np.min(positions)

    def scaleBoundariesTextFile(self):
        f = open('RegionBoundaries/coordinates{0}.txt'.format(self.ID), 'r')
        lines = f.readlines()
        f.close()
        f = open('{0}coordinates.txt'.format(self.path.txtFolder, self.ID), 'w')
        for line in lines:
            pieces = line.split(' ')
            for i in range(2):
                pieces[i] = float(pieces[i]) * self.scale[i]
            f.write('{} {} {}\n'.format(pieces[0], pieces[1], pieces[2]))

    def writeCentroidToTextFile(self):
        f = open('{}centroid.txt'.format(self.path.txtFolder), 'w')
        f.write('{} {} {}\n'.format(self.centroid[0] * self.scale[0], self.centroid[1] * self.scale[1], self.centroid[2]*self.scale[2]))
        f.close()


class Coordinates:

    path = None

    @staticmethod
    def readCentroids(fileName, mainFolder=False):
        if mainFolder:
            f = open('{}{}'.format(Coordinates.path.main, fileName), 'r')
        else:
            f = open('{}{}'.format(Coordinates.path.txtFolder, fileName), 'r')
        lines = f.readlines()
        centroids = np.zeros((len(lines), 3))
        for i in range(len(lines)):
            pieces = lines[i].split(' ')
            centroids[i, :] = pieces[0:3]
        return centroids

    @staticmethod
    def computeDisplacement(centroid, image):
        # Takes scaled centroid and image shape, brings them back to original scale and converts to microns using
        # spacing.
        displacement = [(centroid[0] - (image.shape[1] / image.scale[1])/2) * image.spacing[0],
                        (centroid[1] - (image.shape[0] / image.scale[0])/2) * image.spacing[0], (centroid[2] - (image.shape[2] / image.scale[2]))/2 * image.spacing[2]]
        return displacement

    @staticmethod
    def writeCentroidsToTextFile(centroids, fileName, mainFolder=False):
        if mainFolder:
            f = open('{}{}'.format(Coordinates.path.main, fileName), 'w')
        else:
            f = open('{}{}'.format(Coordinates.path.txtFolder, fileName), 'w')
        for i in range(centroids.shape[0]):
            f.write('{} {} {}\n'.format(centroids[i,0], centroids[i,1], centroids[i,2]))
        f.close()