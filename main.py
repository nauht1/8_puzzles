# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 22:11:49 2023

@author: thuan
"""
import pygame
import sys
import random
import os
import math
import collections
import tkinter
import tkinter.filedialog
from PIL import Image
from itertools import product
from queue import PriorityQueue

WIDTH = 1100
HEIGHT = 720
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500

TILE_SIZE = SCREEN_WIDTH // 3

WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
GREEN = (50, 168, 82)
D_GREEN = (48, 112, 66)
RED = (222, 35, 85)
F_RED = (247, 10, 49)
BLUE = (38, 64, 235)
D_YELLOW = (201, 199, 60)
GRAY = (224, 223, 213)
ORANGE = (222, 120, 87)
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('8-Puzzle')

game_display = pygame.Surface((WIDTH, HEIGHT))

#Time
clock = pygame.time.Clock()
elapsed_time = 0

#Font
font = pygame.font.Font(pygame.font.get_default_font(), 40)
time_font = pygame.font.Font(pygame.font.get_default_font(), 20)

#step
step = 0

d = 15000

file_path = ""

num_tile = 3
def promt_file():
    top = tkinter.Tk()
    top.withdraw()
    file_name = tkinter.filedialog.askopenfilename(parent = top)
    top.destroy()
    return file_name

def remove_images():
    dir_path = 'images/'
    for image in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image)
        if os.path.isfile(image_path):
            os.remove(image_path)

####ref stackoverflow
def slice_tiles():
    img = Image.open(os.path.join(file_path))
    img = img.convert('RGB')
    img.save(os.path.join('images/', 'original_image.jpg'))
    img = img.resize((TILE_SIZE * num_tile, TILE_SIZE * num_tile))
    w, h = img.size
    d = w // num_tile
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    count = 1
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join('images/', f'{count}.jpg')
        img.crop(box).save(out)
        count += 1
####

def load_images():
    images = []
    for i in range(1, 9):
        image_path = os.path.join('images/', f'{i}.jpg')
        if os.path.exists(image_path):
            image = pygame.image.load(image_path)
            image = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
        else:
            placeholder_image = pygame.Surface((TILE_SIZE, TILE_SIZE))
            placeholder_image.fill(GRAY)

            pygame.draw.rect(placeholder_image, BLACK, (0, 0, TILE_SIZE, TILE_SIZE), 1)

            text_num = font.render(str(i), True, BLACK)
            text_num_rect = text_num.get_rect()
            text_num_rect.center = (TILE_SIZE // 2, TILE_SIZE // 2)
            placeholder_image.blit(text_num, text_num_rect)

            image = pygame.transform.scale(placeholder_image, (TILE_SIZE, TILE_SIZE))

        images.append(image)
    return images

board = [6, 7, 1, 2, 4, 0, 5, 3, 8]
last_board = board
output_board = [1, 2, 3, 4, 5, 6, 7, 8, 0]
print(board)
tile_images = load_images()

empty_tile_index = board.index(0)

tile_positions = [(i % 3, i // 3) for i in range(9)]

#button rect
btn_load_rect = pygame.Rect(SCREEN_WIDTH + 95, 100, 180, 50)
btn_delete_rect = pygame.Rect(SCREEN_WIDTH + 95, 160, 180, 50)
btn_select_rect = pygame.Rect(SCREEN_WIDTH + 95, 220, 180, 50)
btn_shuffle_rect= pygame.Rect(SCREEN_WIDTH + 340, 100, 180, 50)
btn_reset_rect = pygame.Rect(SCREEN_WIDTH + 340, 160, 180, 50)
btn_solve_rect = pygame.Rect(SCREEN_WIDTH + 340, 220, 180, 50)

def draw():
    game_display.fill(WHITE)
    for i, val in enumerate(board):
        if val != 0:
            x, y = tile_positions[i]
            game_display.blit(tile_images[val - 1], (x * TILE_SIZE + 50, y * TILE_SIZE + 100))
    pygame.display.update()

def draw_title():
    text = font.render("8 Puzzles", True, GREEN, WHITE)
    game_display.blit(text, (200, 20))

def format_time(ms): 
    s = ms // 1000
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"Time lapse: {h:02}:{m:02}:{s:02}"

def draw_step(step):
    step_text = time_font.render(f"Step: {step}", True, BLACK)
    step_rect = step_text.get_rect(topleft = (450, 70))
    game_display.blit(step_text, step_rect)
    
def draw_time():
    time_text = time_font.render(format_time(elapsed_time), True, BLACK)
    time_rect = time_text.get_rect(topleft =(50, 70))
    game_display.blit(time_text, time_rect)

def draw_button():
    pygame.draw.rect(game_display, BLUE, btn_shuffle_rect)
    shuffle_text = font.render("Shuffle", True, WHITE)
    shuffle_text_rect = shuffle_text.get_rect()
    shuffle_text_rect.center = btn_shuffle_rect.center
    game_display.blit(shuffle_text, shuffle_text_rect)
    
    pygame.draw.rect(game_display, RED, btn_reset_rect)
    reset_text = font.render("Reset", True, WHITE)
    reset_text_rect = reset_text.get_rect()
    reset_text_rect.center = btn_reset_rect.center
    game_display.blit(reset_text, reset_text_rect)
    
    pygame.draw.rect(game_display, D_YELLOW, btn_load_rect)
    load_text = font.render("Load", True, WHITE)
    load_text_rect = load_text.get_rect()
    load_text_rect.center = btn_load_rect.center
    game_display.blit(load_text, load_text_rect)
    
    pygame.draw.rect(game_display, F_RED, btn_delete_rect)
    delete_text = font.render("Delete", True, WHITE)
    delete_text_rect = delete_text.get_rect()
    delete_text_rect.center = btn_delete_rect.center
    game_display.blit(delete_text, delete_text_rect)

    pygame.draw.rect(game_display, ORANGE, btn_solve_rect)
    solve_text = font.render("Solve", True, WHITE)
    solve_text_rect = solve_text.get_rect()
    solve_text_rect.center = btn_solve_rect.center
    game_display.blit(solve_text, solve_text_rect)
    
    pygame.draw.rect(game_display, GREEN, btn_select_rect)
    text = font.render(selected_algorithm, True, WHITE)
    text_rect = text.get_rect()
    text_rect.center = btn_select_rect.center
    game_display.blit(text, text_rect)
def draw_text_using(s):
    text = time_font.render(f'Using {s}', True, BLACK)
    game_display.blit(text, (620, 70))
    
def move_tile(direction):
    global empty_tile_index
    if direction == 'up' and empty_tile_index < 6:
        board[empty_tile_index], board[empty_tile_index + 3] = board[empty_tile_index + 3], board[empty_tile_index]
        empty_tile_index += 3
    elif direction == 'down' and empty_tile_index > 2:
        board[empty_tile_index], board[empty_tile_index - 3] = board[empty_tile_index - 3], board[empty_tile_index]
        empty_tile_index -= 3
    elif direction == 'left' and empty_tile_index % 3 < 2:
        board[empty_tile_index], board[empty_tile_index + 1] = board[empty_tile_index + 1], board[empty_tile_index]
        empty_tile_index += 1
    elif direction == 'right' and empty_tile_index % 3 > 0:
        board[empty_tile_index], board[empty_tile_index - 1] = board[empty_tile_index - 1], board[empty_tile_index]
        empty_tile_index -= 1

def is_win(board):
    return board == output_board

def generate_childs(current_state):
    child_states = []
    empty_tile_index = current_state.index(0)
    
    if empty_tile_index < 6:
        new_state = current_state.copy()
        new_state[empty_tile_index], new_state[empty_tile_index + 3] = new_state[empty_tile_index + 3], new_state[empty_tile_index]
        child_states.append(new_state)
    
    if empty_tile_index > 2: 
        new_state = current_state.copy()
        new_state[empty_tile_index], new_state[empty_tile_index - 3] = new_state[empty_tile_index - 3], new_state[empty_tile_index]
        child_states.append(new_state)
    
    if empty_tile_index % 3 < 2: 
        new_state = current_state.copy()
        new_state[empty_tile_index], new_state[empty_tile_index + 1] = new_state[empty_tile_index + 1], new_state[empty_tile_index]
        child_states.append(new_state)
    
    if empty_tile_index % 3 > 0:
        new_state = current_state.copy()
        new_state[empty_tile_index], new_state[empty_tile_index - 1] = new_state[empty_tile_index - 1], new_state[empty_tile_index]
        child_states.append(new_state)
    
    return child_states

def bfs(board):
    queue = collections.deque()
    visited = set()

    start_state = (board, [])
    queue.append((start_state, 0))
    
    while queue:
        (current_state, path), depth = queue.popleft()
        visited.add(tuple(current_state))
        
        if depth >= d:
            continue
        if is_win(current_state):
            return path

        for child_state in generate_childs(current_state):
            if tuple(child_state) not in visited:
                queue.append(((child_state, path + [child_state]), depth + 1))
    return None

def dfs(board):
    stack = []
    visited = set()
    
    start_state = (board, [])
    stack.append((start_state, 0))

    while stack:
        (current_state, path), depth = stack.pop()
        visited.add(tuple(current_state))
        
        if depth >= d:
            continue
        if is_win(current_state):
            return path
        
        for child_state in generate_childs(current_state):
            if tuple(child_state) not in visited:
                stack.append(((child_state, path + [child_state]), depth + 1))
    return None

def ucs(board):
    start_state = (board, [], 0)
    priority_queue = PriorityQueue()
    visited = set()
    
    priority_queue.put(start_state)
    while not priority_queue.empty():
        current_state, path, cost = priority_queue.get()
        visited.add(tuple(current_state))
        
        if is_win(current_state):
            return path
        
        for child_state in generate_childs(current_state):
            if tuple(child_state) not in visited:
                child_cost = cost + 1
                priority_queue.put((child_state, path + [child_state], child_cost))
    return None
def ids(board, max_depth):
    for depth in range(1, max_depth + 1):
        path = dfs_stack(board, depth)
        if path is not None:
            return path
    return None

def dfs_stack(start_state, depth):
    stack = [(start_state, [])]

    while stack:
        current_state, path = stack.pop()
        if len(path) >= depth:
            continue

        if is_win(current_state):
            return path

        for child_state in generate_childs(current_state):
            if child_state not in path:
                new_path = path + [child_state]
                stack.append((child_state, new_path))
    return None

def matthatan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def heuristic(current_state):
    distance = 0
    for i, tile in enumerate(current_state):
        if tile != 0:
            x1, y1 = i % 3, i // 3
            x2, y2 = (tile - 1) % 3, (tile - 1) // 3
            distance += matthatan_distance(x1, y1, x2, y2)
    return distance
    
def greedy(board, heuristic):
    queue = [(board, [])]
    visited_list = []
    while queue:
        queue.sort(key=lambda x: heuristic(x[0]))
        current_state, path = queue.pop(0)

        if is_win(current_state):
            return path
        
        visited_list.append(current_state)
        for child_state in generate_childs(current_state):
            if child_state not in visited_list:
                queue.append((child_state, path + [child_state]))
    return None

def a_star(board, heuristic):
    queue = [(board, [])]
    visited_list = []
    while queue:
        queue.sort(key=lambda x: len(x[1]) + heuristic(x[0]))
        current_state, path = queue.pop(0)

        if is_win(current_state):
            return path
        
        visited_list.append(current_state)
        for child_state in generate_childs(current_state):
            if child_state not in visited_list:
                queue.append((child_state, path + [child_state]))
    return None

def select_algorithm():
    global menu_open
    menu_open = not menu_open
    
def change_algorithm(new_algorithm):
    global selected_algorithm, menu_open
    selected_algorithm = new_algorithm
    menu_open = False
    
def draw_select_item():
    if menu_open:
        for i, algorithm in enumerate(algorithms):
            item_rect = pygame.Rect(SCREEN_WIDTH + 95, 270 + 50 * i, 180, 50)
            pygame.draw.rect(game_display, D_GREEN, item_rect, 2)
            text = font.render(algorithm, True, BLACK)
            text_rect = text.get_rect(center=item_rect.center)
            game_display.blit(text, text_rect)
            
def draw_win_image():
    image_path = 'images/original_image.jpg'
    if (os.path.isfile(image_path)):
        win_image = pygame.image.load(image_path)
        win_image = pygame.transform.scale(win_image, (300, 300))
        game_display.blit(win_image, (SCREEN_WIDTH + 280, 400))
    
solved = False
solved_path = []
algorithms = ['BFS', 'DFS', 'UCS', 'IDS', 'Greedy', 'A*']
selected_algorithm = algorithms[0]
menu_open = False
running = True
start_time, end_time, execution_time = 0, 0, 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and not is_win(board):
            step += 1
            if event.key == pygame.K_UP:
                move_tile('up')
            elif event.key == pygame.K_DOWN:
                move_tile('down')
            elif event.key == pygame.K_LEFT:
                move_tile('left')
            elif event.key == pygame.K_RIGHT:
                move_tile('right')
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                    if btn_select_rect.collidepoint(event.pos):
                        select_algorithm()
                    if menu_open:
                        for i, algorithm in enumerate(algorithms):
                            item_rect = pygame.Rect(SCREEN_WIDTH + 120, 270 + 50 * i, 180, 50)
                            if item_rect.collidepoint(event.pos):
                                change_algorithm(algorithm)
                                print(selected_algorithm)
            if btn_load_rect.collidepoint(event.pos):
                file_path = promt_file()
                if file_path:
                    slice_tiles()
                    tile_images = load_images()
            if btn_shuffle_rect.collidepoint(event.pos):
                random.shuffle(board)
                print(board)
                solved = False
                is_win(board)
                elapsed_time = 0
                step = 0
            if btn_reset_rect.collidepoint(event.pos):
                board = last_board
                print(board)
                solved = False
                is_win(board)
                elapsed_time = 0
                step = 0
            if btn_delete_rect.collidepoint(event.pos):
                remove_images()
                tile_images = load_images()
            if not solved and selected_algorithm == "BFS" and not is_win(board) and btn_solve_rect.collidepoint(event.pos):
                last_board = board
                start_time = pygame.time.get_ticks()
                solved_path = bfs(board)
                end_time = pygame.time.get_ticks()
                execution_time = end_time - start_time
                if solved_path:
                    solved = True
                print(f"{selected_algorithm}: {solved_path}")
                print(f"{selected_algorithm}: {execution_time} ms")
            elif not solved and selected_algorithm == "DFS" and not is_win(board) and btn_solve_rect.collidepoint(event.pos):
                last_board = board
                start_time = pygame.time.get_ticks()
                solved_path = dfs(board)
                end_time = pygame.time.get_ticks()
                execution_time = end_time - start_time
                if solved_path:
                    solved = True
                print(f"{selected_algorithm}: {solved_path}")
                print(f"{selected_algorithm}: {execution_time} ms")
            elif not solved and selected_algorithm == "UCS" and not is_win(board) and btn_solve_rect.collidepoint(event.pos):
                last_board = board
                start_time = pygame.time.get_ticks()
                solved_path = ucs(board)
                end_time = pygame.time.get_ticks()
                execution_time = end_time - start_time
                if solved_path:
                    solved = True
                print(f"{selected_algorithm}: {solved_path}")
                print(f"{selected_algorithm}: {execution_time} ms")
            elif not solved and selected_algorithm == "IDS" and not is_win(board) and btn_solve_rect.collidepoint(event.pos):
                last_board = board
                max_depth = 40
                start_time = pygame.time.get_ticks()
                solved_path = ids(board, max_depth)
                end_time = pygame.time.get_ticks()
                execution_time = end_time - start_time
                if solved_path:
                    solved = True
                print(f"{selected_algorithm}: {solved_path}")
                print(f"{selected_algorithm}: {execution_time} ms")
            elif not solved and selected_algorithm == "Greedy" and not is_win(board) and btn_solve_rect.collidepoint(event.pos):
                last_board = board
                start_time = pygame.time.get_ticks()
                solved_path = greedy(board, heuristic)
                end_time = pygame.time.get_ticks()
                execution_time = end_time - start_time
                if solved_path:
                    solved = True
                print(f"{selected_algorithm}: {solved_path}")
                print(f"{selected_algorithm}: {execution_time} ms")
            elif not solved and selected_algorithm == "A*" and not is_win(board) and btn_solve_rect.collidepoint(event.pos):
                last_board = board
                start_time = pygame.time.get_ticks()
                solved_path = a_star(board, heuristic)
                end_time = pygame.time.get_ticks()
                execution_time = end_time - start_time
                if solved_path:
                    solved = True
                print(f"{selected_algorithm}: {solved_path}")
                print(f"{selected_algorithm}: {execution_time} ms")
            
    if not is_win(board):
        elapsed_time += clock.get_rawtime()
        clock.tick()
        draw_time()
        
    draw()
    draw_win_image()
    draw_select_item()
    draw_title()
    draw_step(step)
    draw_time()
    draw_button()
    
    if solved:
        start_time, end_time, execution_time = 0, 0, 0
        if selected_algorithm == "BFS" and solved_path:
            draw_text_using('BFS')
            next_state = solved_path.pop(0)
            board = next_state
            step += 1
            pygame.time.delay(100)
        elif selected_algorithm == "DFS" and solved_path:
            draw_text_using('DFS')
            next_state = solved_path.pop(0)
            board = next_state
            step += 1
            pygame.time.delay(50)
        elif selected_algorithm == "UCS" and solved_path:
            draw_text_using("UCS")
            next_state = solved_path.pop(0)
            board = next_state
            step += 1
            pygame.time.delay(100)
        elif selected_algorithm == "IDS" and solved_path:
            draw_text_using("IDS")
            next_state = solved_path.pop(0)
            board = next_state
            step += 1
            pygame.time.delay(100)
        elif selected_algorithm == "Greedy" and solved_path:
            draw_text_using("Greedy")
            next_state = solved_path.pop(0)
            board = next_state
            step += 1
            pygame.time.delay(100)
        elif selected_algorithm == "A*" and solved_path:
            draw_text_using("A*")
            next_state = solved_path.pop(0)
            board = next_state
            step += 1
            pygame.time.delay(100)
    if is_win(board):
        text = font.render('You win!', True, D_YELLOW)
        game_display.blit(text, (200, SCREEN_HEIGHT + 145))
    screen.blit(game_display, (0, 0))
    pygame.display.update()

pygame.quit()