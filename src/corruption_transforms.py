import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


class GaussianNoise:
    _std = {1: 0.05, 2: 0.1, 3: 0.15, 4: 0.25, 5: 0.4}

    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        std = self._std[self.severity]
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))


class GaussianBlur:
    _radius = {1: 0.3, 2: 0.6, 3: 1.0, 4: 1.5, 5: 2.0}

    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        radius = self._radius[self.severity]
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class JPEGCompression:
    _quality = {1: 75, 2: 55, 3: 35, 4: 20, 5: 10}

    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        quality = self._quality[self.severity]
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()


class ResolutionReduction:
    _size = {1: 28, 2: 24, 3: 20, 4: 16, 5: 12}

    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        small = self._size[self.severity]
        return img.resize((small, small), Image.BILINEAR).resize((32, 32), Image.BILINEAR)


class BrightnessChange:
    _factor = {1: 0.6, 2: 0.4, 3: 0.2, 4: 1.5, 5: 2.0}

    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        return ImageEnhance.Brightness(img).enhance(self._factor[self.severity])


CORRUPTION_TYPES = [
    "gaussian_noise",
    "gaussian_blur",
    "jpeg_compression",
    "resolution_reduction",
    "brightness",
]

SEVERITY_LEVELS = [1, 2, 3, 4, 5]


def get_corruption(corruption_type, severity):
    mapping = {
        "gaussian_noise": GaussianNoise,
        "gaussian_blur": GaussianBlur,
        "jpeg_compression": JPEGCompression,
        "resolution_reduction": ResolutionReduction,
        "brightness": BrightnessChange,
    }
    return mapping[corruption_type](severity)
