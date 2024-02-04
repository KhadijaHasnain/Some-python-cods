import os
from getkey import getkey
import re
import sys

def create_grid(w: int, h: int):
    # Calculate the dimensions of the grid
    grid_w = 3 * w + 2
    grid_h = 3 * h + 2

    # Initialize the grid with empty spaces
    grid = [['  ' for _ in range(grid_w)] for _ in range(grid_h)]

    # Set the boundaries of the central playing area
    for i in range(grid_h):
        grid[i][0] = ' |'
        grid[i][-1] = '| '
    for j in range(grid_w):
        grid[0][j] = '--'
        grid[-1][j] = '--'

    return grid

def check_move(tetramino: list, grid: list):
    initial_coordinates, _, offset = tetramino
    for x, y in initial_coordinates:
        x_offset, y_offset = offset
        new_x, new_y = x + x_offset, y + y_offset
        print(f"Checking move to ({new_x}, {new_y})")
        
        if not (0 <= new_x < len(grid[0]) and 0 <= new_y < len(grid)):
            print("Out of bounds")
            return False  # The tetramino is out of the grid

        if grid[new_y][new_x] != '  ':
            print("Cell not empty")
            return False  # The cell is not empty

    print("Move is valid")
    return True


def check_win(grid: list):
    for row in grid:
        for cell in row:
            if cell == '  ':  # There's an empty cell
                return False
    return True  # All cells are filled

def import_card(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Extract the dimensions of the board
    dimensions = tuple(map(int, lines[0].split(',')))
    
    # Initialize the list of tetraminos
    tetraminos = []
    
    # Regular expression pattern for extracting the coordinates and color code
    pattern = r'\((\d+), (\d+)\);;(\d+;\d+;\d+)'
    
    for line in lines[1:]:
        # Split the line into blocks and color code
        blocks_str, color_code = line.split(';;')
        
        # Extract the coordinates from the blocks string
        blocks = [tuple(map(int, re.findall(r'\d+', block))) for block in blocks_str.split(';') if block]
        
        # Append the tetramino to the list
        tetraminos.append([blocks, color_code.strip(), (0, 0)])  # Set the initial position to (0, 0)
        
    return dimensions, tetraminos

def rotate_tetramino(tetramino: list, clockwise: bool = True):
    initial_coordinates, color, offset = tetramino
    rotated_coordinates = [(y, x) for x, y in initial_coordinates] if clockwise else [(y, -x) for x, y in initial_coordinates]
    return [rotated_coordinates, color, offset]

def place_tetraminos(tetraminos: list, grid: list):
    for tetramino in tetraminos:
        initial_coordinates, color, offset = tetramino
        for x, y in initial_coordinates:
            x_offset, y_offset = offset
            grid[y + y_offset][x + x_offset] = {'color': color}
    return grid

def setup_tetraminos(tetraminos: list, grid: list):
    for i, tetramino in enumerate(tetraminos):
        initial_coordinates, color, offset = tetramino
        new_coordinates = [(x + i % 3 * 3, y + i // 3 * 3) for x, y in initial_coordinates]
        tetraminos[i] = [new_coordinates, color, offset]
    return place_tetraminos(tetraminos, grid), tetraminos

def print_grid(grid):
    for row in grid:
        for cell in row:
            if 'color' in cell and cell['color'] is not None:
                color = cell['color']
                print(f"\033[{color}m\u2588\033[0m", end="")
            else:
                print("\u2588", end="")
        print()

def main():
    # Import the card
    dimensions, tetraminos = import_card(sys.argv[1])

    # Create the grid
    grid = create_grid(*dimensions)

    # Setup the tetraminos
    grid, tetraminos = setup_tetraminos(tetraminos, grid)

    # Game loop
    while True:
        # Print the grid
        print_grid(grid)

        # Check if the game is won
        if check_win(grid):
            print("You've won!")
            break

        # Get the player's input
        key = getkey()

        # Handle the player's input
        if key == 'q':  # Quit the game
            break
        elif key == 'r':  # Rotate the current tetramino
            tetraminos[0] = rotate_tetramino(tetraminos[0])
        # Add more controls here...

        # Check if the move is valid
        if not check_move(tetraminos[0], grid):
            print("Invalid move!")
            continue

        # Place the tetramino
        grid = place_tetraminos([tetraminos[0]], grid)

        # Remove the placed tetramino from the list
        tetraminos.pop(0)

        # Check if there are no more tetraminos
        if not tetraminos:
            print("No more pieces!")
            break

if __name__ == "__main__":
    main()