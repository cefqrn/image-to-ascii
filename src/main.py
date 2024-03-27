from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import numpy as np

from argparse import ArgumentParser, FileType
from string import digits, ascii_letters, punctuation
from math import ceil, floor

DEFAULT_CHAR_SET = digits + ascii_letters + punctuation + " "
DEFAULT_CONTRAST = 1
DEFAULT_FONT = "Courier"
DEFAULT_FONT_SIZE = 16

DEFAULT_H_TILE_COUNT = 80

DEFAULT_TILE_WIDTH = 0.6
DEFAULT_TILE_HEIGHT = 1.5


def pad_with(a: np.ndarray, h_padding: int, v_padding: int) -> np.ndarray:
    """return the array padded by repeating edge values"""
    left = h_padding // 2
    right = h_padding - left

    top = v_padding // 2
    bottom = v_padding - top

    return np.pad(a, ((top, bottom), (left, right)), "edge")


def to_tiles(frame: np.ndarray, tile_width: int, tile_height: int) -> np.ndarray:
    """
    convert frame to a 2d array of tiles of the specified dimensions
    Frame dimensions must be a multiple of given dimensions
    """

    h, w, *_ = frame.shape

    v_tile_count = h // tile_height
    h_tile_count = w // tile_width

    # https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
    return (
        frame
        .reshape(v_tile_count, tile_height, h_tile_count, tile_width)
        .swapaxes(1, 2)
    )


def to_char_glyph(font: ImageFont.ImageFont, tile_width: int, tile_height: int, c: str) -> np.ndarray:
    """return the glyph of a character in a tile of the specified dimensions"""
    img = Image.new("L", (tile_width, tile_height))

    draw = ImageDraw.Draw(img)
    draw.text((round(tile_width / 2), round(tile_height / 2)), c, 0xff, font, anchor="mm")

    return np.asarray(img)


def main() -> int:
    # parse command line arguments
    parser = ArgumentParser(description="Convert an image to ascii art.")

    parser.add_argument(
        dest="image_file", type=FileType("rb"),
        help="image file to process"
    )

    parser.add_argument(
        "-d", dest="use_tile_dimensions", action="store_true",
        help="use tile dimensions to determine tile counts"
    )

    parser.add_argument(
        "-c", dest="contrast", metavar="multiple", type=float, default=DEFAULT_CONTRAST,
        help=f"contrast multiplier (default: {DEFAULT_CONTRAST!r})"
    )
    parser.add_argument(
        "-C", dest="char_set", metavar="chars", type=str, default=DEFAULT_CHAR_SET,
        help=f"available characters (default: {DEFAULT_CHAR_SET.replace('%', '%%')!r})"
    )
    parser.add_argument(
        "-f", dest="font", metavar="font", type=str, default=DEFAULT_FONT,
        help=f"target font (default: {DEFAULT_FONT!r})"
    )
    parser.add_argument(
        "-s", dest="font_size", metavar="size", type=int, required=False,
        help=f"size of the font in pixels (default: {DEFAULT_FONT_SIZE!r})"
    )

    tile_counts = parser.add_argument_group("tile counts")
    tile_counts.add_argument(
        "-x", dest="h_tile_count", metavar="count", type=int, default=DEFAULT_H_TILE_COUNT,
        help=f"horizontal tile count (default: {DEFAULT_H_TILE_COUNT!r})"
    )
    tile_counts.add_argument(
        "-y", dest="v_tile_count", metavar="count", type=int, required=False,
        help="vertical tile count"
    )

    tile_dimensions = parser.add_argument_group("tile dimensions")
    tile_dimensions.add_argument(
        "-w", dest="tile_width", metavar="multiple", type=float, default=0.6,
        help=f"width of tile in multiple of font size (default: {DEFAULT_TILE_WIDTH!r})"
    )
    tile_dimensions.add_argument(
        "-l", dest="tile_height", metavar="multiple", type=float, default=1.5,
        help=f"height of tile in multiple of font size (default: {DEFAULT_TILE_HEIGHT!r})"
    )

    args = parser.parse_args()

    # open image
    with Image.open(args.image_file) as img:
        w, h = img.size
        frame = np.asarray(ImageEnhance.Contrast(img.convert("L")).enhance(args.contrast))

    # more argument parsing
    font_size: int | None = args.font_size
    tile_width: int
    tile_height: int
    if args.use_tile_dimensions:
        if font_size is None:
            font_size = DEFAULT_FONT_SIZE

        # ceil because the characters need to fit into the tile
        tile_width = ceil(
            args.tile_width
            if args.absolute_tile_width
            else args.tile_width * font_size
        )

        tile_height = ceil(
            args.tile_height
            if args.absolute_tile_height
            else args.tile_height * font_size
        )

        # pad frame to multiple of tile dimensions
        frame = pad_with(frame, -w % tile_width, -h % tile_height)
    else:
        h_tile_count = args.h_tile_count
        v_tile_count = args.v_tile_count \
            if args.v_tile_count is not None \
            else round(h_tile_count * h/w * args.tile_width/args.tile_height)

        tile_width_neg, h_pad = divmod(-w, h_tile_count)
        tile_height_neg, v_pad = divmod(-h, v_tile_count)

        tile_width = -tile_width_neg
        tile_height = -tile_height_neg

        # largest font size that fits within the calculated tile dimensions
        if font_size is None:
            font_size = floor(min(tile_width / args.tile_width, tile_height / args.tile_height))

        # pad frame to multiple of tile counts
        frame = pad_with(frame, h_pad, v_pad)

    font = ImageFont.truetype(args.font, font_size)

    # get char glyphs
    char_set: str = args.char_set
    char_glyphs = np.array([
        to_char_glyph(font, tile_width, tile_height, c)
        for c in char_set
    ], dtype=np.int16)  # int16 to prevent overflow in the subtraction


    def closest_char(tile: np.ndarray) -> str:
        """returns the character with the closest glyph to tile"""
        return char_set[np.abs(char_glyphs - tile).sum((1, 2)).argmin()]


    # print output
    for row in to_tiles(frame, tile_width, tile_height):
        print("".join(map(closest_char, row)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
