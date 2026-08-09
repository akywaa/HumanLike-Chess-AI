import chess.pgn
import numpy as np
import os

INPUT_FILE = "filtered_1700_2100.pgn"
MAX_POSITIONS = 5000000

def board_to_matrix(board):
    # 18 layers: 12 piece planes + turn + 4 castling + en passant
    matrix = np.zeros((18, 8, 8), dtype=np.int8)
    
    for square, piece in board.piece_map().items():
        row = square // 8
        col = square % 8
        piece_idx = piece.piece_type - 1
        if not piece.color:
            piece_idx += 6
        matrix[piece_idx, row, col] = 1

    if board.turn:
        matrix[12, :, :] = 1
        
    if board.has_kingside_castling_rights(chess.WHITE): matrix[13, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE): matrix[14, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK): matrix[15, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK): matrix[16, :, :] = 1
    
    if board.ep_square is not None:
        row = board.ep_square // 8
        col = board.ep_square % 8
        matrix[17, row, col] = 1

    return matrix

print("Pre-allocating files on disk to save RAM...")

# Create memmap arrays directly on disk to save RAM.
X_array = np.lib.format.open_memmap("X_chess_data.npy", mode='w+', dtype=np.int8, shape=(MAX_POSITIONS, 18, 8, 8))
Y_array = np.lib.format.open_memmap("Y_chess_data.npy", mode='w+', dtype=np.int16, shape=(MAX_POSITIONS,))

move_to_id = {}
positions_count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as pgn:
    while positions_count < MAX_POSITIONS:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break

        board = game.board()

        for move in game.mainline_moves():
            if positions_count >= MAX_POSITIONS:
                break
                
            matrix = board_to_matrix(board)
            move_uci = move.uci()

            if move_uci not in move_to_id:
                move_to_id[move_uci] = len(move_to_id)

            # Write matrix and move straight to disk.
            X_array[positions_count] = matrix
            Y_array[positions_count] = move_to_id[move_uci]

            board.push(move)
            positions_count += 1

        if positions_count % 50000 == 0:
            print(f"Positions processed: {positions_count}")
            # Flush caches to disk so RAM stays low.
            X_array.flush()
            Y_array.flush()

np.save("move_to_id.npy", move_to_id)

# Close the files properly.
del X_array
del Y_array

print(f"Collection done. Total positions: {positions_count}. Unique moves: {len(move_to_id)}")
print("Files X_chess_data.npy, Y_chess_data.npy and move_to_id.npy created successfully")