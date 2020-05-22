import io
import pathlib
import uuid

import numpy as np

import skimage.io
import skimage.color
import skimage.filters
import skimage.metrics
import skimage.morphology
import skimage.segmentation
import skimage.transform

from PIL import Image as PILImage

from sklearn.cluster import KMeans


class Image:
    small_ratio = 0.25

    def __init__(self, array=None):
        if not isinstance(array, np.ndarray):
            raise ValueError("'array' should be a numpy array...")

        self.uuid = str(uuid.uuid4())
        self.array = self._as_uint8_rgba(array)

    def _repr_png_(self):
        """Jupyter Notebook image display hook"""
        return PILImage.fromarray(self.array)._repr_png_()

    @property
    def width(self):
        return self.array.shape[1]

    @property
    def height(self):
        return self.array.shape[0]

    @property
    def short_side(self):
        return min(self.width, self.height)

    @property
    def long_side(self):
        return max(self.width, self.height)

    @property
    def aspect_ratio(self):
        return self.width / self.height

    @property
    def area(self):
        return self.width * self.height

    @property
    def area_non_zero(self):
        return np.sum(self.array[:, :, 0] > 0)

    @property
    def has_transparency(self):
        return np.min(self.array[:, :, 3]) < 255

    @property
    def small(self):
        return self.rescale(self.small_ratio)

    @property
    def rgba(self):
        return self.array

    @property
    def rgba_small(self):
        return self.small

    @property
    def rgba_normalized(self):
        return np.array(self.array, dtype=np.float64) / 255.0

    @property
    def rgba_normalized_small(self):
        return np.array(self.small, dtype=np.float64) / 255.0

    @property
    def alpha(self):
        return self.array[:, :, 3]

    @property
    def rgb(self):
        return self.array[:, :, :3]

    @property
    def rgb_small(self):
        return self.small[:, :, :3]

    @property
    def rgb_normalized(self):
        return np.array(self.array[:, :, :3], dtype=np.float64) / 255.0

    @property
    def rgb_normalized_small(self):
        return np.array(self.small[:, :, :3], dtype=np.float64) / 255.0

    @property
    def lab(self):
        return skimage.color.rgb2lab(self.rgb)

    @property
    def lab_small(self):
        return skimage.color.rgb2lab(self.rgb_small)

    @property
    def lab_normalized(self):
        lab_array = self.lab

        lab_array[:, :, 0] = lab_array[:, :, 0] / 100.0
        lab_array[:, :, 1] = (lab_array[:, :, 1] + 128.0) / 255.0
        lab_array[:, :, 2] = (lab_array[:, :, 2] + 128.0) / 255.0

        return lab_array

    @property
    def lab_normalized_small(self):
        lab_array = self.lab_small

        lab_array[:, :, 0] = lab_array[:, :, 0] / 100.0
        lab_array[:, :, 1] = (lab_array[:, :, 1] + 128.0) / 255.0
        lab_array[:, :, 2] = (lab_array[:, :, 2] + 128.0) / 255.0

        return lab_array

    @property
    def grayscale(self):
        return np.array(skimage.color.rgb2gray(self.rgb) * 255, dtype=np.uint8)

    @property
    def grayscale_rgba(self):
        grayscale_array = np.array(
            skimage.color.rgb2gray(self.rgb) * 255, dtype=np.uint8
        )

        grayscale_array = skimage.color.gray2rgb(grayscale_array)

        return np.dstack((grayscale_array, self.array[:, :, 3]))

    @property
    def blurred(self):
        return self.blur(sigma=1.0)

    @property
    def gradients(self):
        return self.generate_gradients_array(shape="SQUARE", size=8)

    @property
    def empty_mask(self):
        return self.__class__(np.empty_like(self.rgb, dtype=np.uint8))

    @property
    def full_mask(self):
        return self.__class__((np.full_like(self.rgb, 255, dtype=np.uint8)))

    @property
    def inverted(self):
        return self.invert(as_image=True)

    @property
    def average_color(self):
        average_r = int(round(np.mean(self.array[:, :, 0])))
        average_g = int(round(np.mean(self.array[:, :, 1])))
        average_b = int(round(np.mean(self.array[:, :, 2])))

        return average_r, average_g, average_b

    @property
    def average_lightness(self):
        return int(round(np.mean(self.grayscale.flatten())))

    @property
    def average_mask_lightness(self):
        grayscale = self.grayscale.flatten()
        grayscale = np.delete(grayscale, np.where(grayscale == 0))

        return int(round(np.mean(grayscale)))

    @property
    def predominant_color(self):
        minimum_error = 10.0
        predominant_color = None

        for color in self.dominant_colors:
            image_error = np.full(self.rgb.shape, color, dtype=np.uint8)
            image_error = np.dstack((image_error, self.array[:, :, 3]))

            error = skimage.metrics.normalized_root_mse(self.array, image_error)

            if error < minimum_error:
                minimum_error = error
                predominant_color = color

        return predominant_color

    @property
    def dominant_colors(self):
        return self.determine_dominant_colors(quantity=8)

    @property
    def as_pil(self):
        return PILImage.fromarray(self.array)

    @property
    def as_png_bytes(self):
        png_bytes = io.BytesIO()
        self.as_pil.save(png_bytes, format="PNG")
        png_bytes.seek(0)

        return png_bytes.read()

    def update(self, array, new_uuid=False):
        self.array = self._as_uint8_rgba(array)

        if new_uuid:
            self.uuid = str(uuid.uuid4())

    def resize(self, width, height, order=1, anti_aliasing=True, as_image=False):
        array = skimage.transform.resize(
            self.array, (height, width), anti_aliasing=anti_aliasing, order=order
        )

        if as_image:
            return self.__class__(array)
        else:
            return self._as_uint8_rgba(array)

    def resize_long_side_to(self, size, as_image=False):
        scale = size / self.long_side
        return self.rescale(scale, as_image=as_image)

    def resize_short_side_to(self, size, as_image=False):
        scale = size / self.short_side
        return self.rescale(scale, as_image=as_image)

    def rescale(self, scale, as_image=False):
        array = skimage.transform.rescale(self.array, scale, multichannel=True)

        if as_image:
            return self.__class__(array)
        else:
            return self._as_uint8_rgba(array)

    def rotate(self, angle, as_image=False):
        rotate_array = np.array(
            self.as_pil.rotate(angle, resample=PILImage.BICUBIC), dtype=np.uint8
        )

        if as_image:
            return self.__class__(rotate_array)
        else:
            return rotate_array

    def blur(self, sigma=1.0, as_image=False):
        blur_array = np.array(
            skimage.filters.gaussian(self.array, sigma=sigma, multichannel=True) * 255,
            dtype=np.uint8,
        )

        if as_image:
            return self.__class__(blur_array)
        else:
            return blur_array

    def invert(self, as_image=False):
        invert_array = np.array(skimage.util.invert(self.rgb))

        invert_array = np.dstack((invert_array, self.array[:, :, 3]))

        if as_image:
            return self.__class__(invert_array)
        else:
            return invert_array

    def desaturate(self, ratio, as_image=False):
        desaturate_array = np.array(
            (
                (self.rgb_normalized * ratio)
                + (self.__class__(self.grayscale_rgba).rgb_normalized * (1.0 - ratio))
            )
            * 255,
            dtype=np.uint8,
        )

        desaturate_array = np.dstack((desaturate_array, self.array[:, :, 3]))

        if as_image:
            return self.__class__(desaturate_array)
        else:
            return desaturate_array

    def segment(self, segments=24, segment_by="COLOR", compactness=8, sigma=2):
        if segment_by not in ["COLOR", "LIGHTNESS"]:
            segment_by = "COLOR"

        if segment_by == "COLOR":
            array = self.rgb_normalized
        else:
            array = self.__class__(self.grayscale_rgba).rgb_normalized

        return skimage.segmentation.slic(
            array,
            n_segments=segments,
            compactness=compactness,
            sigma=sigma,
            enforce_connectivity=False,
        )

    def determine_dominant_colors(self, quantity=8):
        image = self.resize_long_side_to(256, as_image=True)
        image_kmeans = image.lab.reshape(image.width * image.height, 3)

        kmeans = KMeans(n_clusters=quantity, n_jobs=1).fit_predict(image_kmeans)

        clusters = dict()

        for i in range(quantity):
            clusters[i] = list()

        for i, cluster in enumerate(kmeans):
            clusters[cluster].append(image_kmeans[i])

        dominant_colors = list()

        for lab_tuples in clusters.values():
            l, a, b = zip(*lab_tuples)

            lab_color = np.array((np.mean(l), np.mean(a), np.mean(b)), dtype=np.float64)
            lab_color = np.full((1, 1, 3), lab_color, dtype=np.float64)

            rgb_color = np.array(skimage.color.lab2rgb(lab_color) * 255, dtype=np.uint8)

            dominant_colors.append(tuple(rgb_color[0, 0, :]))

        return dominant_colors

    def generate_color_strip(self, colors=2048, height=1):
        image = self.resize_long_side_to(256, as_image=True)

        segments = skimage.segmentation.slic(
            image.rgb,
            compactness=1,
            n_segments=colors,
            sigma=0,
            enforce_connectivity=False,
        )

        image = skimage.color.label2rgb(segments, image.rgb, kind="avg")
        image = np.array(skimage.color.rgb2hsv(image) * 255, dtype=np.uint8)

        colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)

        color_strip = np.zeros((height, len(colors), 3), dtype=np.uint8)

        for i, color in enumerate(colors):
            color_strip[:, i, :] = color

        color_strip = np.array(skimage.color.hsv2rgb(color_strip) * 255, dtype=np.uint8)

        return self.__class__(color_strip)

    def generate_gradients_array(self, shape="SQUARE", size=1):
        if shape not in ("SQUARE", "DISK"):
            shape = "SQUARE"

        shapes = {"SQUARE": skimage.morphology.square, "DISK": skimage.morphology.disk}

        return skimage.filters.rank.gradient(self.grayscale, shapes[shape](size))

    def generate_otsu_threshold_mask(self, sigma=0.0):
        grayscale = self.grayscale

        threshold = skimage.filters.threshold_otsu(grayscale)

        mask = grayscale <= threshold
        mask = np.array(mask, dtype=np.uint8) * 255

        if sigma > 0.0:
            mask = skimage.filters.gaussian(mask, sigma=sigma)
            mask = np.array(mask * 255, dtype=np.uint8)

        return self.__class__(mask)

    def generate_local_threshold_mask(self, size=51, sigma=0.0):
        if not size % 2:
            size += 1

        grayscale = self.grayscale

        threshold = skimage.filters.threshold_local(grayscale, size)

        mask = grayscale <= threshold
        mask = np.array(mask, dtype=np.uint8) * 255

        if sigma > 0.0:
            mask = skimage.filters.gaussian(mask, sigma=sigma)
            mask = np.array(mask * 255, dtype=np.uint8)

        return self.__class__(mask)

    def get_local_neighborhood(self, point, size, as_image=False):
        y0 = max(point[0] - size, 0)
        x0 = max(point[1] - size, 0)
        y1 = min(point[0] + size, self.height)
        x1 = min(point[1] + size, self.width)

        local_neighborhood_array = self.array[y0:y1, x0:x1, :]

        if as_image:
            return self.__class__(local_neighborhood_array)
        else:
            return local_neighborhood_array

    def calculate_error(self, image):
        return skimage.metrics.mean_squared_error(
            self.rgba_normalized, image.rgba_normalized
        )

    def calculate_point_error(self, image, point, size):
        reference_array = self.get_local_neighborhood(
            point, size, as_image=True
        ).rgba_normalized

        test_array = image.get_local_neighborhood(
            point, size, as_image=True
        ).rgba_normalized

        return skimage.metrics.mean_squared_error(reference_array, test_array)

    def calculate_otsu_threshold(self, channel=0):
        if channel not in (0, 1, 2, 3):
            channel = 0

        return skimage.filters.threshold_otsu(self.array[:, :, channel])

    @classmethod
    def from_png_bytes(cls, png_bytes):
        try:
            array = skimage.io.imread(png_bytes)
        except ValueError:
            raise ValueError("Invalid image data!")

        return cls(array)

    @classmethod
    def from_file(cls, file_path):
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError()

        try:
            array = skimage.io.imread(file_path)
        except ValueError:
            raise ValueError(f"Invalid image data: 'f{file_path}'")

        return cls(array)

    @classmethod
    def copy(cls, image=None, keep_uuid=False):
        if not isinstance(image, cls):
            raise TypeError("'image' should be of type Image...")

        image_copy = cls(image.array)

        if keep_uuid:
            image_copy.uuid = image.uuid

        return image_copy

    @staticmethod
    def _as_uint8_rgba(array):
        # Rescue a few common dtypes before exception
        if array.dtype == np.bool:
            array = np.array(array, dtype=np.uint8) * 255
        elif array.dtype == np.float64:
            if np.max(array) <= 1.0:
                array = array * 255.0

            array = np.array(array, dtype=np.uint8)

        if array.dtype != np.uint8:
            raise TypeError(f"Unsupported image array dtype: '{array.dtype}'")

        # Grayscale to RGB
        if len(array.shape) == 2:
            array = skimage.color.gray2rgb(array)

        # RGB to RGBA
        if array.shape[2] == 3:
            array = np.array(PILImage.fromarray(array).convert("RGBA"))

        return array
