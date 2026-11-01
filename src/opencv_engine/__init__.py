"""OpenCV: a PlumCraft engine on `opencv-python-headless`. Pictures are ARRAY values, float32 0-1,
H × W × 3 in RGB or H × W for one channel, so they cross to any engine unchanged; OpenCV's BGR and
8-bit formats stay inside this file (`to_u8`, `from_u8`, `encoded`). A `cv2.error` is an op error
on this node, not a crash (`cv`); a C++ crash can't be caught, so run this engine as a process,
as `plumcraft-engine.toml` does."""

import time
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

import cv2
import numpy as np
from plumcraft_engine import Array, ArrayValue, Blob, ConfigError, Cx, Engine, File, Image, Int, NodeError, OpError, Port, Range, Table, image, node
from plumcraft_engine.arrays import thumb

__version__ = "0.1.0"

# Results must not depend on a machine's GPU.
cv2.ocl.setUseOpenCL(False)

# The largest side a node makes, and the most pixels a picture may have here.
MAX_SIDE = 16384
MAX_PIXELS = 256 << 20
MAX_FILE = 1 << 30
# The most rows a table of lines, contours or points carries.
MAX_ROWS = 10000
# The largest kernel Filter 2D takes, a side.
MAX_KERNEL = 15
RGB_NAMES = ("red", "green", "blue")
# Each colour space: its channels, and the conversions to and from RGB.
SPACES = {
    "HSV": (("hue", "saturation", "value"), cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB),
    "Lab": (("L", "a", "b"), cv2.COLOR_RGB2Lab, cv2.COLOR_Lab2RGB),
    "YCrCb": (("Y", "Cr", "Cb"), cv2.COLOR_RGB2YCrCb, cv2.COLOR_YCrCb2RGB),
}
Side = Annotated[int, Range(1, MAX_SIDE, invalid="clamp")]
Kernel = Annotated[int, Range(1, 31, invalid="clamp"), Port(tooltip="Kernel size; an even one becomes the next odd one")]
Space = Literal["HSV", "Lab", "YCrCb"]
Morphology = Literal["erode", "dilate", "open", "close", "gradient", "top hat", "black hat"]
MORPH = {
    "erode": cv2.MORPH_ERODE,
    "dilate": cv2.MORPH_DILATE,
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
    "top hat": cv2.MORPH_TOPHAT,
    "black hat": cv2.MORPH_BLACKHAT,
}
SHAPES = {"rectangle": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS, "ellipse": cv2.MORPH_ELLIPSE}
BORDER = cv2.BORDER_REPLICATE


def cv(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call OpenCV. Its errors are this node's, with OpenCV's own last line as the reason."""
    try:
        return fn(*args, **kwargs)
    except cv2.error as e:
        lines = [line for line in str(e).splitlines() if line.strip()]
        raise OpError(lines[-1].strip() if lines else "OpenCV failed") from None


def picture(a: Any, channels: tuple[int, ...] = (1, 3)) -> np.ndarray:
    """`a` as float32 H × W or H × W × C, checked. An array that isn't one is a config error."""
    data = np.asarray(a)
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    c = data.shape[2] if data.ndim == 3 else 1
    if data.ndim not in (2, 3) or c not in channels or data.size == 0:
        raise ConfigError(f"a picture with {' or '.join(str(n) for n in channels)} channels, not {'×'.join(str(n) for n in data.shape)}")
    if data.size > MAX_PIXELS:
        raise ConfigError("this picture is too large")
    return np.nan_to_num(data.astype("f4"), nan=0.0, posinf=1.0, neginf=0.0)


def to_u8(a: np.ndarray) -> np.ndarray:
    """0-1 to 8 bits, rounded."""
    return (np.clip(a, 0, 1) * 255 + 0.5).astype("u1")


def from_u8(a: np.ndarray) -> np.ndarray:
    return a.astype("f4") / 255


def rgb(a: np.ndarray) -> ArrayValue:
    """Three channels as an RGB picture; one as plain numbers."""
    return ArrayValue(a, space="RGB", names=RGB_NAMES) if a.ndim == 3 else ArrayValue(a)


def odd(n: int) -> int:
    return n | 1


Picture = Annotated[Array, Port(name="picture")]
Seed = Annotated[int, Range(0, 1 << 31, invalid="clamp"), Port(tooltip="The same seed gives the same result")]


def colour_u8(a: Any) -> np.ndarray:
    """A picture as 8-bit RGB, one channel repeated three times."""
    p = to_u8(picture(a))
    return p if p.ndim == 3 else cv2.cvtColor(p, cv2.COLOR_GRAY2RGB)


def grey_u8(a: Any) -> np.ndarray:
    """A picture as one 8-bit channel."""
    p = to_u8(picture(a))
    return cv2.cvtColor(p, cv2.COLOR_RGB2GRAY) if p.ndim == 3 else p


def mask_u8(a: Any, like: np.ndarray, what: str = "mask") -> np.ndarray:
    """A one-channel array as an 8-bit mask, anything above 0 set, the size of `like`."""
    m = (picture(a, (1,)) > 0).astype("u1") * 255
    if m.shape != like.shape[:2]:
        raise ConfigError(f"the {what} is {'×'.join(str(n) for n in m.shape)}, the picture {'×'.join(str(n) for n in like.shape[:2])}")
    return m


def same_size(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ConfigError(f"the pictures differ: {'×'.join(str(n) for n in a.shape)} and {'×'.join(str(n) for n in b.shape)}")


def hex_colour(c: str) -> tuple[int, int, int]:
    """`#rrggbb` as an 8-bit RGB tuple."""
    if len(c) != 7 or c[0] != "#":
        raise ConfigError(f"{c[:20]!r} isn't a colour like #ff0000")
    try:
        return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    except ValueError:
        raise ConfigError(f"{c[:20]!r} isn't a colour like #ff0000") from None


def columns(rows: list[tuple], names: tuple[tuple[str, str], ...]) -> dict:
    """Rows as a table: `names` is (name, type) a column."""
    return {"columns": [{"name": n, "type": t, "values": [r[i] for r in rows]} for i, (n, t) in enumerate(names)]}


def column(table: dict, name: str) -> list:
    """A table's column, or a config error naming what it has."""
    found = next((c["values"] for c in table.get("columns", []) if c.get("name") == name), None)
    if found is None:
        raise ConfigError(f"no column {name!r} in this table")
    return found


def number_rows(table: dict, names: tuple[str, ...]) -> list[tuple[int, ...]]:
    """Rows of these columns as whole pixels."""
    try:
        return list(zip(*(map(lambda v: int(round(float(v))), column(table, n)) for n in names), strict=True))[:MAX_ROWS]
    except (TypeError, ValueError, OverflowError) as e:
        raise ConfigError(f"this table's columns aren't numbers: {e}") from None


def kernel_of(text: str) -> np.ndarray:
    """`0 -1 0; -1 5 -1; 0 -1 0` as a float32 kernel."""
    try:
        rows = [[float(v) for v in row.replace(",", " ").split()] for row in text.split(";") if row.strip()]
    except ValueError:
        raise ConfigError("a kernel is numbers, rows split by `;`") from None
    if not rows or len({len(r) for r in rows}) != 1 or not rows[0]:
        raise ConfigError("a kernel's rows must be the same length")
    if len(rows) > MAX_KERNEL or len(rows[0]) > MAX_KERNEL:
        raise ConfigError(f"a kernel is at most {MAX_KERNEL} a side")
    return np.nan_to_num(np.array(rows, dtype="f4"), nan=0.0, posinf=0.0, neginf=0.0)


class Channels(NamedTuple):
    c1: Array
    c2: Array
    c3: Array


class Sketch(NamedTuple):
    grey: Array
    colour: Array


class Components(NamedTuple):
    labels: Array
    count: Int


class Quantised(NamedTuple):
    picture: Array
    palette: Table


class OpencvEngine(Engine):
    id = "opencv"
    name = "OpenCV"
    version = __version__
    category = "OpenCV"
    family = "opencv"
    # Every building block (Vary, sequences, files), as ANY ports.
    building_blocks = True
    # Its version joins the memo salt: an upgrade never serves an old result.
    dependencies = ("opencv-python-headless",)

    def picture_thumb(self, blob: Blob) -> Any:
        """A picture file's thumbnail, for Load Files, whose values this engine can decode."""
        pixels = cv2.imdecode(np.frombuffer(blob.data, dtype="u1"), cv2.IMREAD_COLOR)
        return None if pixels is None else thumb(from_u8(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)), 160)

    # ---------- pictures in and out ----------

    @node("Test Picture", category=("OpenCV", "Files"))
    def test_picture(self, width: Side = 256, height: Side = 192) -> Array:
        """A source with no file: a colour gradient, a disc and a bar, with edges for the other nodes to find."""
        if width * height * 3 > MAX_PIXELS:
            raise ConfigError("this picture is too large")
        x, y = np.meshgrid(np.linspace(0, 1, width, dtype="f4"), np.linspace(0, 1, height, dtype="f4"))
        out = np.stack([x, y, 1 - x], axis=2)
        side = min(width, height)
        cv2.circle(out, (width // 3, height // 2), side // 4, (1.0, 0.9, 0.2), -1, cv2.LINE_AA)
        cv2.rectangle(out, (width // 2, height // 4), (width - width // 8, height // 2), (0.1, 0.1, 0.1), -1)
        return rgb(out)

    @node("Load Image", category=("OpenCV", "Files"))
    def load_image(self, path: Annotated[str, File("open", (("Pictures", ("png", "jpg", "jpeg", "tif", "tiff", "webp")),))] = "") -> Array:
        """A picture file as RGB, 0-1. A grey file comes as three equal channels."""
        if not path:
            raise ConfigError("choose a picture file")
        # read here, not with cv2.imread: a bad path segfaults inside it, where no `try` can help
        try:
            file = Path(path).expanduser()
            if file.stat().st_size > MAX_FILE:
                raise ConfigError(f"{path}: larger than {MAX_FILE >> 20} MB")
            data = file.read_bytes()
        except (OSError, ValueError) as e:
            raise OpError(f"can't read {path[:200]!r}: {e}") from None
        pixels = cv(cv2.imdecode, np.frombuffer(data, dtype="u1"), cv2.IMREAD_COLOR) if data else None
        if pixels is None:
            raise OpError(f"can't read a picture from {path[:200]!r}")
        return rgb(from_u8(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)))

    @node("Encode Image", category=("OpenCV", "Files"))
    def encode_image(self, picture_in: Picture, format: Literal["png", "jpeg", "webp"] = "png") -> Image:
        """A picture as a file's bytes, 8 bits a channel, for Write Files or another engine. The inverse of Decode Image."""
        a = to_u8(picture(picture_in))
        ok, data = cv(cv2.imencode, f".{format}", cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if a.ndim == 3 else a)
        if not ok:
            raise OpError(f"OpenCV couldn't encode a {format}")
        return image(data.tobytes(), pixels=picture_in)

    @node("Decode Image", category=("OpenCV", "Files"))
    def decode_image(self, encoded: Image) -> Array:
        """A picture file's bytes, from Encode Image or another engine, as RGB, 0-1. The inverse of Encode Image."""
        pixels = cv(cv2.imdecode, np.frombuffer(encoded.data, dtype="u1"), cv2.IMREAD_COLOR)
        if pixels is None:
            raise OpError(f"can't read this {encoded.meta.get('media')}")
        return rgb(from_u8(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)))

    # ---------- colour ----------

    @node("To Grey", category=("OpenCV", "Colour"))
    def to_grey(self, picture_in: Picture) -> Array:
        """An RGB picture as one channel of luminance. Colour can't be recovered, so it has no inverse."""
        return ArrayValue(cv(cv2.cvtColor, picture(picture_in, (3,)), cv2.COLOR_RGB2GRAY))

    @node("To Colour Space", category=("OpenCV", "Colour"))
    def to_space(self, picture_in: Picture, space: Space = "HSV") -> Array:
        """RGB in another colour space, in OpenCV's float units: HSV hue 0-360, Lab L 0-100. The inverse of From Colour Space."""
        names, forward, _ = SPACES[space]
        return ArrayValue(cv(cv2.cvtColor, picture(picture_in, (3,)), forward), space=space, names=names)

    @node("From Colour Space", category=("OpenCV", "Colour"))
    def from_space(self, picture_in: Picture, space: Space = "HSV") -> Array:
        """A picture in another colour space, back to RGB. The inverse of To Colour Space."""
        # the array's own space, when it says, so a wired space can't be misread
        known = getattr(picture_in, "space", None)
        _, _, back = SPACES[known if known in SPACES else space]
        return rgb(cv(cv2.cvtColor, picture(picture_in, (3,)), back))

    @node("Split Channels", category=("OpenCV", "Colour"))
    def split(self, picture_in: Picture) -> Channels:
        """Three channels as three arrays. The inverse of Merge Channels."""
        a = picture(picture_in, (3,))
        return Channels(a[:, :, 0], a[:, :, 1], a[:, :, 2])

    @node("Merge Channels", category=("OpenCV", "Colour"))
    def merge(self, c1: Array, c2: Array, c3: Array) -> Array:
        """Three one-channel arrays of one size as an RGB picture. The inverse of Split Channels."""
        planes = [picture(c, (1,)) for c in (c1, c2, c3)]
        if len({p.shape for p in planes}) != 1:
            raise ConfigError(f"the channels differ in size: {', '.join('×'.join(str(n) for n in p.shape) for p in planes)}")
        return rgb(np.stack(planes, axis=2))

    @node("Equalize Histogram", category=("OpenCV", "Colour"))
    def equalize(self, picture_in: Picture) -> Array:
        """Spread a picture's tones across the range: one channel directly, RGB by its luminance, so colours keep their hue."""
        a = picture(picture_in)
        if a.ndim == 2:
            return ArrayValue(from_u8(cv(cv2.equalizeHist, to_u8(a))))
        ycc = cv2.cvtColor(to_u8(a), cv2.COLOR_RGB2YCrCb)
        ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
        return rgb(from_u8(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)))

    # ---------- filters ----------

    @node("Gaussian Blur", category=("OpenCV", "Filters"))
    def gaussian_blur(self, picture_in: Picture, sigma: Annotated[float, Range(0.1, 50, invalid="clamp")] = 2.0) -> Array:
        """A soft blur. The kernel follows from sigma."""
        return rgb(cv(cv2.GaussianBlur, picture(picture_in), (0, 0), sigma, borderType=BORDER))

    @node("Median Blur", category=("OpenCV", "Filters"))
    def median_blur(self, picture_in: Picture, size: Kernel = 5) -> Array:
        """Each pixel the median of its neighbours: removes specks and keeps edges."""
        return rgb(from_u8(cv(cv2.medianBlur, to_u8(picture(picture_in)), odd(size))))

    @node("Bilateral Filter", category=("OpenCV", "Filters"))
    def bilateral(
        self,
        picture_in: Picture,
        diameter: Annotated[int, Range(1, 15, invalid="clamp")] = 7,
        sigma_colour: Annotated[float, Range(0.001, 1, invalid="clamp")] = 0.1,
        sigma_space: Annotated[float, Range(0.1, 50, invalid="clamp")] = 5.0,
    ) -> Array:
        """Smooths flat areas and keeps edges. Two pixels mix when they are near each other and near in colour."""
        return rgb(cv(cv2.bilateralFilter, picture(picture_in), diameter, sigma_colour, sigma_space, borderType=BORDER))

    @node("Laplacian", category=("OpenCV", "Filters"))
    def laplacian(self, picture_in: Picture, size: Annotated[int, Range(1, 7, invalid="clamp")] = 3) -> Array:
        """Second derivative: edges as signed values on zero. A value view of `heat` shows them well."""
        return ArrayValue(cv(cv2.Laplacian, picture(picture_in), cv2.CV_32F, ksize=odd(size), borderType=BORDER))

    @node("Sobel", category=("OpenCV", "Filters"))
    def sobel(self, picture_in: Picture, direction: Literal["x", "y"] = "x", size: Annotated[int, Range(1, 7, invalid="clamp")] = 3) -> Array:
        """First derivative along one direction: signed values on zero."""
        dx, dy = (1, 0) if direction == "x" else (0, 1)
        return ArrayValue(cv(cv2.Sobel, picture(picture_in), cv2.CV_32F, dx, dy, ksize=odd(size), borderType=BORDER))

    # ---------- edges and thresholds ----------

    @node("Canny", category=("OpenCV", "Edges"))
    def canny(
        self,
        picture_in: Picture,
        low: Annotated[float, Range(0, 1, invalid="clamp")] = 0.2,
        high: Annotated[float, Range(0, 1, invalid="clamp")] = 0.5,
    ) -> Array:
        """Thin edges as a mask, 0 or 1. Thresholds are 0-1 where OpenCV's are 0-255. A low above the high swaps with it."""
        a = picture(picture_in)
        grey = to_u8(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a)
        return ArrayValue(from_u8(cv(cv2.Canny, grey, min(low, high) * 255, max(low, high) * 255)))

    @node("Threshold", category=("OpenCV", "Edges"))
    def threshold(
        self,
        picture_in: Picture,
        level: Annotated[float, Range(0, 1, invalid="clamp")] = 0.5,
        method: Literal["fixed", "otsu", "triangle"] = "fixed",
        invert: bool = False,
    ) -> Array:
        """One channel as a mask, 0 or 1. Otsu and triangle find the level themselves, and level is then ignored."""
        a = to_u8(picture(picture_in, (1,)))
        kind = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        kind |= {"fixed": 0, "otsu": cv2.THRESH_OTSU, "triangle": cv2.THRESH_TRIANGLE}[method]
        return ArrayValue(from_u8(cv(cv2.threshold, a, level * 255, 255, kind)[1]))

    @node("Adaptive Threshold", category=("OpenCV", "Edges"))
    def adaptive_threshold(
        self,
        picture_in: Picture,
        size: Annotated[int, Range(3, 99, invalid="clamp")] = 15,
        offset: Annotated[float, Range(-1, 1, invalid="clamp")] = 0.02,
        invert: bool = False,
    ) -> Array:
        """One channel as a mask, each pixel against its neighbourhood's mean less `offset`: survives uneven lighting."""
        kind = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        a = to_u8(picture(picture_in, (1,)))
        return ArrayValue(from_u8(cv(cv2.adaptiveThreshold, a, 255, cv2.ADAPTIVE_THRESH_MEAN_C, kind, odd(size), offset * 255)))

    # ---------- morphology ----------

    @node("Morphology", category=("OpenCV", "Morphology"))
    def morphology(
        self,
        picture_in: Picture,
        operation: Morphology = "open",
        shape: Literal["rectangle", "cross", "ellipse"] = "ellipse",
        size: Kernel = 5,
        iterations: Annotated[int, Range(1, 20, invalid="clamp")] = 1,
    ) -> Array:
        """Grow, shrink or tidy shapes in a mask or picture. Open removes specks; close fills holes."""
        element = cv2.getStructuringElement(SHAPES[shape], (odd(size), odd(size)))
        return rgb(cv(cv2.morphologyEx, picture(picture_in), MORPH[operation], element, iterations=iterations, borderType=BORDER))

    # ---------- geometry ----------

    @node("Resize", category=("OpenCV", "Geometry"))
    def resize(
        self,
        picture_in: Picture,
        width: Side = 128,
        height: Annotated[int, Range(0, MAX_SIDE, invalid="clamp"), Port(tooltip="0 keeps the shape")] = 0,
        method: Literal["area", "linear", "cubic", "lanczos", "nearest"] = "area",
    ) -> Array:
        """A picture at another size. Area suits shrinking, and cubic or Lanczos enlarging."""
        a = picture(picture_in)
        h = height or max(1, round(a.shape[0] * width / a.shape[1]))
        if width * h * 3 > MAX_PIXELS:
            raise ConfigError("the result would be too large")
        how = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4, "nearest": cv2.INTER_NEAREST}[
            method
        ]
        return rgb(cv(cv2.resize, a, (width, h), interpolation=how))

    @node("Rotate", category=("OpenCV", "Geometry"))
    def rotate(self, picture_in: Picture, turn: Literal["90 clockwise", "180", "90 anticlockwise"] = "90 clockwise") -> Array:
        """A picture turned by quarter turns, nothing lost. Three of one turn, or one of the opposite, undo it."""
        how = {"90 clockwise": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180, "90 anticlockwise": cv2.ROTATE_90_COUNTERCLOCKWISE}[turn]
        return rgb(cv(cv2.rotate, picture(picture_in), how))

    @node("Flip", category=("OpenCV", "Geometry"))
    def flip(self, picture_in: Picture, axis: Literal["left to right", "top to bottom", "both"] = "left to right") -> Array:
        """A picture mirrored. A second flip undoes it."""
        return rgb(cv(cv2.flip, picture(picture_in), {"left to right": 1, "top to bottom": 0, "both": -1}[axis]))

    @node("Pyramid Down", category=("OpenCV", "Geometry"))
    def pyramid_down(self, picture_in: Picture) -> Array:
        """Half the size, blurred first. Pyramid Up goes back in size, though the detail is gone."""
        return rgb(cv(cv2.pyrDown, picture(picture_in)))

    @node("Pyramid Up", category=("OpenCV", "Geometry"))
    def pyramid_up(self, picture_in: Picture) -> Array:
        """Twice the size, smoothed. Pyramid Down goes back."""
        a = picture(picture_in)
        if a.shape[0] * a.shape[1] * 4 * 3 > MAX_PIXELS:
            raise ConfigError("the result would be too large")
        return rgb(cv(cv2.pyrUp, a))

    # ---------- more colour ----------

    @node("Normalize", category=("OpenCV", "Colour"))
    def normalize(self, picture_in: Picture) -> Array:
        """A picture stretched to fill 0-1. A flat picture becomes zeros."""
        a = picture(picture_in)
        low, high = float(a.min()), float(a.max())
        return rgb((a - low) / (high - low) if high > low else np.zeros_like(a))

    @node("Convert Scale", category=("OpenCV", "Colour"))
    def convert_scale(
        self,
        picture_in: Picture,
        gain: Annotated[float, Range(-10, 10, invalid="clamp")] = 1.2,
        offset: Annotated[float, Range(-1, 1, invalid="clamp")] = 0.0,
    ) -> Array:
        """Contrast and brightness: each value times gain, plus offset, clipped to 0-1."""
        return rgb(np.clip(picture(picture_in) * gain + offset, 0, 1).astype("f4"))

    @node("CLAHE", category=("OpenCV", "Colour"))
    def clahe(
        self,
        picture_in: Picture,
        limit: Annotated[float, Range(0.1, 40, invalid="clamp"), Port(tooltip="How far contrast may rise; lower is gentler")] = 2.0,
        tiles: Annotated[int, Range(1, 32, invalid="clamp")] = 8,
    ) -> Array:
        """Equalize Histogram by region, limited so noise isn't amplified: brings out detail in dark and bright areas alike."""
        tool = cv2.createCLAHE(clipLimit=limit, tileGridSize=(tiles, tiles))
        a = to_u8(picture(picture_in))
        if a.ndim == 2:
            return ArrayValue(from_u8(cv(tool.apply, a)))
        lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv(tool.apply, lab[:, :, 0])
        return rgb(from_u8(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)))

    @node("In Range", category=("OpenCV", "Edges"))
    def in_range(
        self,
        picture_in: Picture,
        low1: float = 0.0,
        high1: float = 1.0,
        low2: float = 0.0,
        high2: float = 1.0,
        low3: float = 0.0,
        high3: float = 1.0,
    ) -> Array:
        """A mask, 1 where every channel lies in its range. Pairs with To Colour Space, to pick a hue (0-360 in HSV). One channel uses the first range."""
        a = picture(picture_in)
        if a.ndim == 2:
            return ArrayValue(from_u8(cv(cv2.inRange, a, low1, high1)))
        return ArrayValue(from_u8(cv(cv2.inRange, a, np.array([low1, low2, low3], "f4"), np.array([high1, high2, high3], "f4"))))

    # ---------- more filters ----------

    @node("Box Filter", category=("OpenCV", "Filters"))
    def box_filter(self, picture_in: Picture, size: Kernel = 5) -> Array:
        """Each pixel the plain mean of its neighbours: the fastest blur."""
        return rgb(cv(cv2.blur, picture(picture_in), (odd(size), odd(size)), borderType=BORDER))

    @node("Filter 2D", category=("OpenCV", "Filters"))
    def filter_2d(
        self,
        picture_in: Picture,
        kernel: Annotated[str, Port(tooltip="Numbers by row, rows split by ;")] = "0 -1 0; -1 5 -1; 0 -1 0",
        normalise: Annotated[bool, Port(tooltip="Divide by the kernel's sum, when that isn't 0")] = False,
    ) -> Array:
        """Any convolution, from a kernel you type. The default sharpens."""
        k = kernel_of(kernel)
        total = float(k.sum())
        return rgb(cv(cv2.filter2D, picture(picture_in), cv2.CV_32F, k / total if normalise and total else k, borderType=BORDER))

    @node("Scharr", category=("OpenCV", "Filters"))
    def scharr(self, picture_in: Picture, direction: Literal["x", "y"] = "x") -> Array:
        """A sharper-angled Sobel, size 3: signed values on zero."""
        dx, dy = (1, 0) if direction == "x" else (0, 1)
        return ArrayValue(cv(cv2.Scharr, picture(picture_in), cv2.CV_32F, dx, dy, borderType=BORDER))

    # ---------- features: pictures to tables, and back ----------

    @node("Hough Lines", category=("OpenCV", "Features"))
    def hough_lines(
        self,
        edges: Annotated[Array, Port(tooltip="An edge mask, from Canny")],
        votes: Annotated[int, Range(1, 1000, invalid="clamp")] = 40,
        min_length: Annotated[int, Range(0, MAX_SIDE, invalid="clamp")] = 30,
        max_gap: Annotated[int, Range(0, MAX_SIDE, invalid="clamp")] = 5,
    ) -> Table:
        """Straight line segments in an edge mask, as a table x1, y1, x2, y2, length. Draw Segments puts them back on a picture."""
        mask = (picture(edges, (1,)) > 0).astype("u1") * 255
        found = cv(cv2.HoughLinesP, mask, 1, np.pi / 180, votes, minLineLength=min_length, maxLineGap=max_gap)
        rows = [] if found is None else found.reshape(-1, 4)[:MAX_ROWS].tolist()
        cols = (("x1", "INT"), ("y1", "INT"), ("x2", "INT"), ("y2", "INT"), ("length", "FLOAT"))
        return columns([(*r, float(np.hypot(r[2] - r[0], r[3] - r[1]))) for r in rows], cols)

    @node("Draw Segments", category=("OpenCV", "Features"))
    def draw_segments(
        self, picture_in: Picture, segments: Table, colour: str = "#ff0000", thickness: Annotated[int, Range(1, 32, invalid="clamp")] = 1
    ) -> Array:
        """A table of x1, y1, x2, y2 drawn on a picture."""
        out = colour_u8(picture_in).copy()
        for x1, y1, x2, y2 in number_rows(segments, ("x1", "y1", "x2", "y2")):
            cv(cv2.line, out, (x1, y1), (x2, y2), hex_colour(colour), thickness, cv2.LINE_AA)
        return rgb(from_u8(out))

    @node("Find Contours", category=("OpenCV", "Features"))
    def find_contours(self, mask: Array, which: Literal["outer", "all"] = "outer") -> Table:
        """The outlines of shapes in a mask, as a table: area, perimeter, box, and points as text `x,y x,y …`. Draw Contours puts them back."""
        m = (picture(mask, (1,)) > 0).astype("u1") * 255
        mode = cv2.RETR_EXTERNAL if which == "outer" else cv2.RETR_LIST
        found = cv(cv2.findContours, m, mode, cv2.CHAIN_APPROX_SIMPLE)[0]
        rows = []
        for i, c in enumerate(found[:MAX_ROWS]):
            x, y, w, h = cv2.boundingRect(c)
            points = " ".join(f"{px},{py}" for px, py in c.reshape(-1, 2).tolist())
            rows.append((i, float(cv2.contourArea(c)), float(cv2.arcLength(c, True)), x, y, w, h, points))
        cols = (("id", "INT"), ("area", "FLOAT"), ("perimeter", "FLOAT"), ("x", "INT"), ("y", "INT"), ("width", "INT"), ("height", "INT"), ("points", "STRING"))
        return columns(rows, cols)

    @node("Draw Contours", category=("OpenCV", "Features"))
    def draw_contours(
        self,
        picture_in: Picture,
        contours: Table,
        colour: str = "#00ff00",
        thickness: Annotated[int, Range(-1, 32, invalid="clamp"), Port(tooltip="-1 fills them")] = 1,
    ) -> Array:
        """The contours from Find Contours drawn on a picture. Thickness 0 is taken as 1."""
        out = colour_u8(picture_in).copy()
        try:
            shapes = [
                np.array([p.split(",") for p in str(t).split()], dtype="i4").reshape(-1, 1, 2) for t in column(contours, "points")[:MAX_ROWS] if str(t).strip()
            ]
        except ValueError:
            raise ConfigError("a contour's points are `x,y x,y …`") from None
        if shapes:
            cv(cv2.drawContours, out, shapes, -1, hex_colour(colour), thickness or 1, cv2.LINE_AA)
        return rgb(from_u8(out))

    @node("Good Features", category=("OpenCV", "Features"))
    def good_features(
        self,
        picture_in: Picture,
        corners: Annotated[int, Range(1, MAX_ROWS, invalid="clamp")] = 100,
        quality: Annotated[float, Range(0.001, 1, invalid="clamp")] = 0.05,
        spacing: Annotated[float, Range(1, 1000, invalid="clamp")] = 8.0,
    ) -> Table:
        """Strong corners, as a table x, y. Draw Points puts them back on a picture."""
        found = cv(cv2.goodFeaturesToTrack, grey_u8(picture_in), corners, quality, spacing)
        rows = [] if found is None else found.reshape(-1, 2).tolist()
        return columns([tuple(r) for r in rows], (("x", "FLOAT"), ("y", "FLOAT")))

    @node("ORB Keypoints", category=("OpenCV", "Features"))
    def orb_keypoints(self, picture_in: Picture, features: Annotated[int, Range(1, MAX_ROWS, invalid="clamp")] = 200) -> Table:
        """Distinctive points that survive scale and rotation, as a table x, y, size, angle, response. Draw Points puts them back."""
        found = cv(cv2.ORB_create(nfeatures=features).detect, grey_u8(picture_in), None)
        cols = (("x", "FLOAT"), ("y", "FLOAT"), ("size", "FLOAT"), ("angle", "FLOAT"), ("response", "FLOAT"))
        return columns([(k.pt[0], k.pt[1], k.size, k.angle, k.response) for k in found], cols)

    @node("Draw Points", category=("OpenCV", "Features"))
    def draw_points(self, picture_in: Picture, points: Table, colour: str = "#ffff00", radius: Annotated[int, Range(1, 64, invalid="clamp")] = 3) -> Array:
        """A table of x, y drawn on a picture as circles."""
        out = colour_u8(picture_in).copy()
        for x, y in number_rows(points, ("x", "y")):
            cv(cv2.circle, out, (x, y), radius, hex_colour(colour), 1, cv2.LINE_AA)
        return rgb(from_u8(out))

    # ---------- more morphology ----------

    @node("Structuring Element", category=("OpenCV", "Morphology"))
    def structuring_element(self, shape: Literal["rectangle", "cross", "ellipse"] = "ellipse", size: Kernel = 5) -> Array:
        """The shape Morphology uses, 0 and 1, to look at. Also usable as a mask."""
        return ArrayValue(cv2.getStructuringElement(SHAPES[shape], (odd(size), odd(size))).astype("f4"))

    @node("Distance Transform", category=("OpenCV", "Morphology"))
    def distance_transform(self, mask: Array) -> Array:
        """For each set pixel of a mask, how far it is, in pixels, to the nearest unset one."""
        m = (picture(mask, (1,)) > 0).astype("u1")
        return ArrayValue(cv(cv2.distanceTransform, m, cv2.DIST_L2, 5).astype("f4"))

    @node("Connected Components", category=("OpenCV", "Morphology"))
    def connected_components(self, mask: Array) -> Components:
        """Number each separate shape in a mask: integer labels, 0 for background (the viewer's labels view colours them), and how many."""
        count, labels = cv(cv2.connectedComponents, (picture(mask, (1,)) > 0).astype("u1"), connectivity=8, ltype=cv2.CV_32S)
        return Components(labels, int(count) - 1)

    # ---------- more geometry ----------

    @node("Crop", category=("OpenCV", "Geometry"))
    def crop(
        self,
        picture_in: Picture,
        x: Annotated[int, Range(0, MAX_SIDE, invalid="clamp")] = 0,
        y: Annotated[int, Range(0, MAX_SIDE, invalid="clamp")] = 0,
        width: Side = 64,
        height: Side = 64,
    ) -> Array:
        """A rectangle of a picture. One reaching past the edge is cut there; one entirely outside is a config error."""
        a = picture(picture_in)
        out = a[y : y + height, x : x + width]
        if out.size == 0:
            raise ConfigError(f"that rectangle is outside this {a.shape[1]}×{a.shape[0]} picture")
        return rgb(out.copy())

    @node("Warp Affine", category=("OpenCV", "Geometry"))
    def warp_affine(
        self,
        picture_in: Picture,
        angle: Annotated[float, Range(-360, 360, invalid="clamp")] = 15.0,
        scale: Annotated[float, Range(0.01, 100, invalid="clamp")] = 1.0,
        shift_x: Annotated[float, Range(-16384, 16384, invalid="clamp")] = 0.0,
        shift_y: Annotated[float, Range(-16384, 16384, invalid="clamp")] = 0.0,
    ) -> Array:
        """Rotate about the centre, scale and shift, keeping the size; the corners repeat the edge. The opposite angle undoes the rotation, softened."""
        a = picture(picture_in)
        h, w = a.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        matrix[:, 2] += (shift_x, shift_y)
        return rgb(cv(cv2.warpAffine, a, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=BORDER))

    # ---------- photo effects ----------

    @node("Stylization", category=("OpenCV", "Photo"))
    def stylization(
        self,
        picture_in: Picture,
        sigma_s: Annotated[float, Range(0, 200, invalid="clamp")] = 60.0,
        sigma_r: Annotated[float, Range(0, 1, invalid="clamp")] = 0.45,
    ) -> Array:
        """A painted look: smooth regions with strong edges."""
        return rgb(from_u8(cv(cv2.stylization, colour_u8(picture_in), sigma_s=sigma_s, sigma_r=sigma_r)))

    @node("Pencil Sketch", category=("OpenCV", "Photo"))
    def pencil_sketch(
        self,
        picture_in: Picture,
        sigma_s: Annotated[float, Range(0, 200, invalid="clamp")] = 60.0,
        sigma_r: Annotated[float, Range(0, 1, invalid="clamp")] = 0.07,
        shade: Annotated[float, Range(0, 0.1, invalid="clamp")] = 0.05,
    ) -> Sketch:
        """A pencil drawing, in grey and in colour."""
        grey, colour = cv(cv2.pencilSketch, colour_u8(picture_in), sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade)
        return Sketch(ArrayValue(from_u8(grey)), rgb(from_u8(colour)))

    @node("Detail Enhance", category=("OpenCV", "Photo"))
    def detail_enhance(
        self,
        picture_in: Picture,
        sigma_s: Annotated[float, Range(0, 200, invalid="clamp")] = 10.0,
        sigma_r: Annotated[float, Range(0, 1, invalid="clamp")] = 0.15,
    ) -> Array:
        """Brings out fine detail, for a crisper picture."""
        return rgb(from_u8(cv(cv2.detailEnhance, colour_u8(picture_in), sigma_s=sigma_s, sigma_r=sigma_r)))

    @node("Edge-Preserving Filter", category=("OpenCV", "Photo"))
    def edge_preserving(
        self,
        picture_in: Picture,
        sigma_s: Annotated[float, Range(0, 200, invalid="clamp")] = 60.0,
        sigma_r: Annotated[float, Range(0, 1, invalid="clamp")] = 0.4,
    ) -> Array:
        """Smooths while keeping edges, faster than the bilateral filter."""
        return rgb(from_u8(cv(cv2.edgePreservingFilter, colour_u8(picture_in), flags=cv2.RECURS_FILTER, sigma_s=sigma_s, sigma_r=sigma_r)))

    @node("Inpaint", category=("OpenCV", "Photo"))
    def inpaint(
        self,
        picture_in: Picture,
        mask: Annotated[Array, Port(tooltip="What to fill: any array above 0, as Threshold or In Range make")],
        radius: Annotated[float, Range(1, 64, invalid="clamp")] = 3.0,
        method: Literal["telea", "navier-stokes"] = "telea",
    ) -> Array:
        """Fill a masked area from what surrounds it, to remove a mark or an object."""
        a = colour_u8(picture_in)
        how = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        return rgb(from_u8(cv(cv2.inpaint, a, mask_u8(mask, a), radius, how)))

    @node("Denoise", category=("OpenCV", "Photo"))
    def denoise(
        self,
        picture_in: Picture,
        strength: Annotated[float, Range(0, 100, invalid="clamp")] = 5.0,
        colour_strength: Annotated[float, Range(0, 100, invalid="clamp")] = 5.0,
    ) -> Array:
        """Fast Non-Local Means: removes noise by averaging similar patches. Slow on large pictures, and it can't stop part way once started."""
        return rgb(from_u8(cv(cv2.fastNlMeansDenoisingColored, colour_u8(picture_in), None, strength, colour_strength, 7, 21)))

    @node("Cartoon", category=("OpenCV", "Photo"))
    def cartoon(
        self,
        picture_in: Picture,
        levels: Annotated[int, Range(0, 4, invalid="clamp"), Port(tooltip="Pyramid levels: more is flatter and faster")] = 2,
        smoothing: Annotated[int, Range(1, 14, invalid="clamp"), Port(tooltip="Bilateral passes")] = 7,
        outline: Annotated[int, Range(3, 99, invalid="clamp"), Port(tooltip="Adaptive threshold's neighbourhood: larger gives bolder outlines")] = 9,
    ) -> Array:
        """A drawing from a photo: flattened colour with dark outlines. `examples/cartoon.json` is the same steps as a group you can open and change."""
        a = colour_u8(picture_in)
        h, w = a.shape[:2]
        flat = a
        for _ in range(levels):
            if min(flat.shape[:2]) < 8:
                break
            flat = cv(cv2.pyrDown, flat)
        for _ in range(smoothing):
            flat = cv(cv2.bilateralFilter, flat, 9, 9, 7)
        flat = cv(cv2.resize, flat, (w, h), interpolation=cv2.INTER_LINEAR)
        grey = cv(cv2.medianBlur, cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), 7)
        lines = cv(cv2.adaptiveThreshold, grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, odd(outline), 2)
        return rgb(from_u8(cv(cv2.bitwise_and, flat, flat, mask=lines)))

    # ---------- clustering and segmentation ----------

    @node("K-Means Colours", category=("OpenCV", "Segmentation"))
    def kmeans_colours(
        self,
        picture_in: Picture,
        colours: Annotated[int, Range(2, 64, invalid="clamp")] = 6,
        attempts: Annotated[int, Range(1, 10, invalid="clamp")] = 1,
        seed: Seed = 1,
    ) -> Quantised:
        """A picture reduced to a few colours, and those colours as a table."""
        a = picture(picture_in, (3,))
        pixels = a.reshape(-1, 3)
        if len(pixels) < colours:
            raise ConfigError(f"{colours} colours from {len(pixels)} pixels")
        cv2.setRNGSeed(seed)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)
        _, labels, centres = cv(cv2.kmeans, pixels, colours, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=colours)
        rows = [(float(r), float(g), float(b), int(n)) for (r, g, b), n in zip(centres.tolist(), counts.tolist(), strict=True)]
        return Quantised(
            rgb(centres[labels.ravel()].reshape(a.shape)), columns(rows, (("red", "FLOAT"), ("green", "FLOAT"), ("blue", "FLOAT"), ("pixels", "INT")))
        )

    @node("Watershed", category=("OpenCV", "Segmentation"))
    def watershed(
        self, picture_in: Picture, markers: Annotated[Array, Port(tooltip="Integer labels, from Connected Components: each seed grows; 0 is unknown")]
    ) -> Array:
        """Grow labelled seeds across a picture until they meet: integer labels, with -1 on the boundaries."""
        a = colour_u8(picture_in)
        seeds = np.asarray(markers)
        if seeds.ndim != 2 or seeds.shape != a.shape[:2]:
            raise ConfigError(f"the markers are {'×'.join(str(n) for n in seeds.shape)}, the picture {'×'.join(str(n) for n in a.shape[:2])}")
        out = np.nan_to_num(seeds.astype("f8"), nan=0.0).clip(-1, 1 << 30).astype("i4")
        cv(cv2.watershed, cv2.cvtColor(a, cv2.COLOR_RGB2BGR), out)
        return ArrayValue(out)

    @node("Grab Cut", category=("OpenCV", "Segmentation"))
    def grab_cut(
        self,
        picture_in: Picture,
        left: Annotated[float, Range(0, 1, invalid="clamp"), Port(tooltip="The box around the subject, as fractions of the picture")] = 0.1,
        top: Annotated[float, Range(0, 1, invalid="clamp")] = 0.1,
        right: Annotated[float, Range(0, 1, invalid="clamp")] = 0.9,
        bottom: Annotated[float, Range(0, 1, invalid="clamp")] = 0.9,
        iterations: Annotated[int, Range(1, 20, invalid="clamp")] = 3,
        seed: Seed = 1,
    ) -> Array:
        """Separate a subject from its background, given a box around it: a mask, 1 for the subject."""
        a = colour_u8(picture_in)
        h, w = a.shape[:2]
        x0, y0, x1, y1 = int(min(left, right) * w), int(min(top, bottom) * h), int(max(left, right) * w), int(max(top, bottom) * h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise ConfigError("the box is too small")
        if min(x0, y0, w - x1, h - y1) < 1 and (x0, y0, x1, y1) == (0, 0, w, h):
            raise ConfigError("the box must leave some background outside it")
        cv2.setRNGSeed(seed)
        mask = np.zeros((h, w), "u1")
        cv(
            cv2.grabCut,
            cv2.cvtColor(a, cv2.COLOR_RGB2BGR),
            mask,
            (x0, y0, x1 - x0, y1 - y0),
            np.zeros((1, 65)),
            np.zeros((1, 65)),
            iterations,
            cv2.GC_INIT_WITH_RECT,
        )
        return ArrayValue(((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype("f4"))

    # ---------- combining pictures ----------

    @node("Add Weighted", category=("OpenCV", "Composite"))
    def add_weighted(
        self,
        a: Array,
        b: Array,
        weight_a: Annotated[float, Range(-10, 10, invalid="clamp")] = 0.5,
        weight_b: Annotated[float, Range(-10, 10, invalid="clamp")] = 0.5,
        offset: Annotated[float, Range(-1, 1, invalid="clamp")] = 0.0,
    ) -> Array:
        """Two pictures of one size mixed: a × weight a + b × weight b + offset."""
        x, y = picture(a), picture(b)
        same_size(x, y)
        return rgb(cv(cv2.addWeighted, x, weight_a, y, weight_b, offset))

    @node("Abs Diff", category=("OpenCV", "Composite"))
    def abs_diff(self, a: Array, b: Array) -> Array:
        """Where two pictures of one size differ, and by how much."""
        x, y = picture(a), picture(b)
        same_size(x, y)
        return rgb(cv(cv2.absdiff, x, y))

    @node("Bitwise", category=("OpenCV", "Composite"))
    def bitwise(self, a: Array, b: Array | None = None, operation: Literal["and", "or", "xor", "not"] = "and", mask: Array | None = None) -> Array:
        """Masks combined bit by bit, or a picture kept where a mask is set (And with b the same picture). `not` uses only a."""
        x = to_u8(picture(a))
        fn = {"and": cv2.bitwise_and, "or": cv2.bitwise_or, "xor": cv2.bitwise_xor}.get(operation)
        keep = None if mask is None else mask_u8(mask, x)
        if fn is None:
            return rgb(from_u8(cv(cv2.bitwise_not, x, mask=keep)))
        if b is None:
            raise ConfigError(f"{operation} needs b")
        y = to_u8(picture(b))
        same_size(x, y)
        return rgb(from_u8(cv(fn, x, y, mask=keep)))

    # ---------- video ----------

    @node("Video Frame", category=("OpenCV", "Files"))
    def video_frame(
        self,
        path: Annotated[str, File("open", (("Video", ("mp4", "mov", "avi", "mkv", "webm")),))] = "",
        seconds: Annotated[float, Range(0, 86400, invalid="clamp")] = 0.0,
    ) -> Array:
        """One frame of a video file, as RGB, 0-1. Past the end is an op error."""
        if not path:
            raise ConfigError("choose a video file")
        # checked here first: OpenCV's readers are not safe with a bad path
        try:
            file = Path(path).expanduser()
            found = file.is_file()
        except (OSError, ValueError) as e:
            raise OpError(f"can't read {path[:200]!r}: {e}") from None
        if not found:
            raise OpError(f"no file {path[:200]!r}")
        video = cv(cv2.VideoCapture, str(file))
        try:
            if not video.isOpened():
                raise OpError(f"can't read a video from {path[:200]!r}")
            video.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = cv(video.read)
        finally:
            video.release()
        if not ok or frame is None:
            raise OpError(f"no frame at {seconds:g} s")
        return rgb(from_u8(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    # ---------- testing ----------

    @node("Slow", category=("OpenCV", "Testing"))
    def slow(self, array: Array, seconds: Annotated[float, Range(0, 600, invalid="clamp")] = 0.05, *, cx: Cx) -> Array:
        """Passes its input through after `seconds`, reporting progress and stopping on cancel. How a long node behaves."""
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            if cx.cancelled():
                raise NodeError.cancelled()
            cx.progress((time.monotonic() - start) / max(seconds, 1e-9), "waiting")
            time.sleep(0.01)
        return array
