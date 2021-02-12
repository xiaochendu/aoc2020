from __future__  import annotations
from collections import Counter, defaultdict, namedtuple, deque
from itertools   import permutations, combinations, cycle, product, islice, chain, repeat
from functools   import lru_cache
from typing      import Dict, Tuple, Set, List, Iterator, Optional
from sys         import maxsize

import math
import re

import numpy as np

def read_data(input: str, parser=str, sep='\n', testing=False) -> list:
    if testing:
        sections = input.split(sep)
    else:
        sections = open(input).read().split(sep)
    return [parser(section) for section in sections]

test_string = """Tile 2311:
..##.#..#.
##..#.....
#...##..#.
####.#...#
##.##.###.
##...#.###
.#.#.#..##
..#....#..
###...#.#.
..###..###

Tile 1951:
#.##...##.
#.####...#
.....#..##
#...######
.##.#....#
.###.#####
###.##.##.
.###....#.
..#.#..#.#
#...##.#..

Tile 1171:
####...##.
#..##.#..#
##.#..#.#.
.###.####.
..###.####
.##....##.
.#...####.
#.##.####.
####..#...
.....##...

Tile 1427:
###.##.#..
.#..#.##..
.#.##.#..#
#.#.#.##.#
....#...##
...##..##.
...#.#####
.#.####.#.
..#..###.#
..##.#..#.

Tile 1489:
##.#.#....
..##...#..
.##..##...
..#...#...
#####...#.
#..#.#.#.#
...#.#.#..
##.#...##.
..##.##.##
###.##.#..

Tile 2473:
#....####.
#..#.##...
#.##..#...
######.#.#
.#...#.#.#
.#########
.###.#..#.
########.#
##...##.#.
..###.#.#.

Tile 2971:
..#.#....#
#...###...
#.#.###...
##.##..#..
.#####..##
.#..####.#
#..#.#..#.
..####.###
..#.#.###.
...#.#.#.#

Tile 2729:
...#.#.#.#
####.#....
..#.#.....
....#..#.#
.##..##.#.
.#.####...
####.#.#..
##.####...
##..#.##..
#.##...##.

Tile 3079:
#.#.#####.
.#..######
..#.......
######....
####.#..#.
.#...#.##.
#.#####.##
..#.###...
..#.......
..#.###..."""
test_ins = read_data(test_string, sep="\n\n", testing=True)

test_str = """Tile 2311:
..##.#..#.
##..#.....
#...##..#.
####.#...#
##.##.###.
##...#.###
.#.#.#..##
..#....#..
###...#.#.
..###..###"""

tile, data = re.split(r":\n", test_str)
tile_num, *_ = re.match(r"Tile (\d+)", tile).groups()
data_arr = np.array([list(line) for line in data.split("\n")])

# Part I  
# Assemble the tiles into an image. What do you get if you multiply together the IDs of the four corner tiles?

class Tile:
    def __init__(self, id: int, data: str):
        self._id = int(id)
        self._data = self.parse_data(data)
        self._loc = None
    
    def __str__(self):
        return f"Tile ID: {self._id} \n\n {self._data.shape}"

    def __repr__(self):
        return str(self._id)

    def parse_data(self, data):
        table = str.maketrans('.#', '01')
        translated_pic = data.translate(table)
        # print(translated_pic)
        return np.array([list(line) for line in translated_pic.split("\n")], dtype=int)

    def rotate_left(self, times=1):
        self._data = np.rot90(self._data, k=times)
    
    def flip(self):
        self._data = np.flipud(self._data)    

    @property
    def loc(self):
        return self._loc
    
    @loc.setter
    def loc(self, coords: Tuple[int, int]):
        self._loc = coords

    def get_edges(self):
        for i in (0, -1):
            yield ''.join(map(str, self._data[i, :]))
            yield ''.join(map(str, self._data[:, i]))

    @property
    def data(self):
        return self._data

    @property
    def edges(self):
        # returns edges from top, left, down, right
        return list(self.get_edges())

    @staticmethod
    def canonical(edge: List[int]):
        # standarize flipped edges
        str_edge = ''.join(map(str, edge))
        return min(str_edge, str_edge[::-1])

def create_tiles(input: List[str]):
    all_tiles = {}
    for data in input:
        tile, pic = re.split(r":\n", data)
        tile_num, *_ = re.match(r"Tile (\d+)", tile).groups()
        all_tiles[int(tile_num)] = Tile(tile_num, pic)
    return all_tiles

def count_edges(tiles: dict):
    edges_count = Counter([Tile.canonical(edge) for tile in tiles.values() for edge in list(tile.edges)])
    return edges_count

def is_outermost(edge, edges_count): return edges_count[Tile.canonical(edge)] == 1

def get_corner_tiles(tiles: dict, edges_count):
    """Yield the IDs of the tiles at four corners"""
    for id, tile in tiles.items():
        if sum([is_outermost(tile, edges_count) for tile in tile.edges]) == 2:
            yield int(id)

def run_part1(input: List[str]):
    all_tiles = create_tiles(input)
    edges_count = count_edges(all_tiles)
    edges_ids = get_corner_tiles(all_tiles, edges_count)

    return math.prod(edges_ids)

# run_part1(test_ins)

real_ins = read_data("input.txt", sep="\n\n")
print("Part 2: Product of the corner tiles is:", run_part1(real_ins))


# Part II
# How many # are not part of a sea monster?

def rotate_into_postion(tile_id, final_positions, all_tiles) -> bool:
    '''given a tile, rotate it into final position based constraints by the surrounding tiles'''
    curr_tile = all_tiles[tile_id]
    surrounding_edges = get_surrounding_edges(tile_id, final_positions, all_tiles)
    for i in range(4):
        curr_tile.rotate_left()
        curr_tile_edges = all_tiles[tile_id].edges
        if all_edges_match(curr_tile_edges, surrounding_edges):
            return True
    curr_tile.flip()
    curr_tile_edges = all_tiles[tile_id].edges
    if all_edges_match(curr_tile_edges, surrounding_edges):
        return True
    for i in range(4):
        curr_tile.rotate_left()
        curr_tile_edges = all_tiles[tile_id].edges
        if all_edges_match(curr_tile_edges, surrounding_edges):
            return True
    return False

def in_bounds(row, col, matrix) -> bool:
    nrows, ncols = matrix.shape
    return 0 <= row < nrows and 0 <= col < ncols

def get_neighbors(tile_id, final_positions, all_tiles):
    deltas = [(-1, 0), (0, -1), (1, 0), (0, 1)]

    curr_tile = all_tiles[tile_id]
    r, c = curr_tile.loc
    # print("curr r,c", r, c)
    for dr, dc in deltas:
        new_r, new_c = r + dr, c + dc
        if not in_bounds(new_r, new_c, final_positions) or final_positions[new_r, new_c] == 0:
            yield None
        else:
            yield final_positions[new_r, new_c]

def get_surrounding_edges(tile_id, final_positions, all_tiles):
    neighbor_ids = get_neighbors(tile_id, final_positions, all_tiles)
    # print(neighbor_ids)
    # take the relevant edge from each of the adjacent tiles
    surrounding_edges = [all_tiles[n].edges[idx-2] if n else None for idx, n in enumerate(neighbor_ids)]
    # print("surrounding", surrounding_edges)
    return surrounding_edges

def all_edges_match(curr_tile_edges, surrounding_edges) -> bool:
    return all(x == y for x, y in zip(curr_tile_edges, surrounding_edges) if y)

def generate_final_image(final_positions, all_tiles):
    tiles_per_row, tiles_per_col = final_positions.shape
    pixels_per_row, pixels_per_col = all_tiles[final_positions[0, 0]].data.shape

    final_image = np.zeros((tiles_per_row * (pixels_per_row - 2), tiles_per_col * (pixels_per_col - 2)), dtype=int)

    for r, row in enumerate(final_positions):
        for c, tile in enumerate(row):
            row_start = r * (pixels_per_row - 2)
            row_end = row_start + pixels_per_row - 2
            col_start = c * (pixels_per_col - 2)
            col_end = col_start + pixels_per_col - 2
            final_image[row_start:row_end, col_start:col_end] = all_tiles[tile].data[1:-1, 1:-1]
    
    return final_image

def run_part2(input: List[str]):
    all_tiles = create_tiles(input)

    edges_count = count_edges(all_tiles)
    edges_ids = get_corner_tiles(all_tiles, edges_count)


    first_corner_tile = next(edges_ids)
    starting_tile = all_tiles[first_corner_tile]

    # check if (top, left, down, right) has `top` and `left` as edges
    # if not rotate the tile into position
    while not all(np.equal(list(map(lambda x: is_outermost(x, edges_count), starting_tile.edges)), (1, 1, 0, 0))):
        starting_tile.rotate_left()

    num_tiles = len(all_tiles)
    # layout of final tiles
    final_positions = np.zeros(num_tiles, dtype=int).reshape(int(np.sqrt(num_tiles)), int(np.sqrt(num_tiles)))

    # set position of first tile in edge_ids
    final_positions[0, 0] = first_corner_tile
    all_tiles[first_corner_tile].loc = (0, 0)

    remaining_tiles = list(all_tiles.keys())
    remaining_tiles.remove(first_corner_tile)

    for r, row in enumerate(final_positions):
        for c, col in enumerate(row):
            # print(remaining_tiles)
            if final_positions[r, c] == 0:
                for tile_id in remaining_tiles:
                    # print("testing out tile", tile_id)
                    final_positions[r, c] = tile_id
                    all_tiles[tile_id].loc = (r, c)

                    surrounding_edges = get_surrounding_edges(tile_id, final_positions, all_tiles)
                    curr_tile_edges = all_tiles[tile_id].edges
                    all_edges_match(curr_tile_edges, surrounding_edges)
                    if rotate_into_postion(tile_id, final_positions, all_tiles):

                        remaining_tiles.remove(tile_id)
                        break

    # Final part of the algorithm
    pattern = ("                  # \n"
               "#    ##    ##    ###\n"
               " #  #  #  #  #  #   ")

    # translate: '#' -> 1, ' ' -> 0
    table = str.maketrans(' #', '01')
    translated_pattern = pattern.translate(table)
   
    # try all orientations of the kernel
    kernel = np.array([list(line) for line in translated_pattern.split("\n")], dtype=int)
    kernels = []
    for i in range(4):
        kernel = np.rot90(kernel)
        kernels.append(kernel.copy())
    kernel = np.flipud(kernel)
    kernels.append(kernel.copy())
    for i in range(3):
        kernel = np.rot90(kernel)
        kernels.append(kernel.copy())

    final_image = generate_final_image(final_positions, all_tiles)

    # visualize the final images
    # print(['\n'.join([''.join(str(x) for x in line)]) for line in final_image])
    from scipy.signal import convolve2d
    # num_matches = max(np.sum((kernel.sum() == convolve2d(img, kernel, mode='valid'))) for img in final_images)

    matches_in_each_kernel = [np.sum((kernel.sum() == convolve2d(final_image, kernel, mode='valid'))) for kernel in kernels]
    num_matches = max(matches_in_each_kernel)
    return final_image.sum() - num_matches * kernel.sum()

print("Part 2: Final roughness measure is:", run_part2(real_ins))