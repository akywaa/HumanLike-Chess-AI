import sys
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import time

DEEP_ANALYZE = True
MINIMAX_DEPTH = 3

# Max allowed score drop (1.5 pawns) before a move counts as a blunder.
BLUNDER_THRESHOLD = 150

# Per-move time budget in seconds; overridden by the UCI clock in "go".
DEFAULT_MAX_TIME = 2.0
MAX_TIME = DEFAULT_MAX_TIME
MIN_MOVE_TIME = 0.05
MAX_MOVE_TIME = 10.0
MAX_SEARCH_DEPTH = 4

search_start = 0
timeout_flag = False

nodes_count = 0

def check_time():
    global nodes_count, timeout_flag, search_start
    nodes_count += 1
    if nodes_count & 1023 == 0:
        if time.time() - search_start > MAX_TIME:
            timeout_flag = True

def mvv_lva_score(board, move):
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        v_type = victim.piece_type if victim else chess.PAWN
        a_type = attacker.piece_type if attacker else chess.PAWN
        vals = {chess.PAWN: 10, chess.KNIGHT: 32, chess.BISHOP: 33, chess.ROOK: 50, chess.QUEEN: 90, chess.KING: 100}
        return vals.get(v_type, 10) * 100 - vals.get(a_type, 10)

    if board.gives_check(move):
        return -1
    return -10

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ChessResNet(nn.Module):
    def __init__(self, num_classes):
        super(ChessResNet, self).__init__()
        self.conv_initial = nn.Conv2d(18, 128, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(128)
        self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(8)])
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        x = self.res_blocks(x)
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def board_to_matrix(board):
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
    # layers 13-16 castling rights, 17 en passant
    if board.has_kingside_castling_rights(chess.WHITE):
        matrix[13, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        matrix[14, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        matrix[15, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        matrix[16, :, :] = 1
    if board.ep_square is not None:
        row = board.ep_square // 8
        col = board.ep_square % 8
        matrix[17, row, col] = 1
    return matrix

# Piece-square tables (centipawns), White-oriented: row 0 = rank 8, row 7 = rank 1.
# Black pieces mirror the index via sq ^ 56.
PST_PAWN = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0,
]

PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

PST_ROOK = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0,
]

PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

PST_KING = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]

PIECE_PST = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING,
}


def evaluate_board(board, engine_color):
    if board.is_checkmate():
        return -999999 if board.turn == engine_color else 999999
    if board.is_game_over():
        return 0

    piece_values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
    score = 0

    # Material + placement. PSTs are White-oriented; black uses sq ^ 56.
    engine_is_white = (engine_color == chess.WHITE)
    for pt, val in piece_values.items():
        score += len(board.pieces(pt, engine_color)) * val
        score -= len(board.pieces(pt, not engine_color)) * val

        pst = PIECE_PST[pt]
        for sq in board.pieces(pt, engine_color):
            score += pst[sq] if engine_is_white else pst[sq ^ 56]
        for sq in board.pieces(pt, not engine_color):
            score -= pst[sq] if not engine_is_white else pst[sq ^ 56]

    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    for sq in center_squares:
        piece = board.piece_at(sq)
        if piece:
            if piece.color == engine_color:
                score += 30
            else:
                score -= 30

    if not board.has_castling_rights(engine_color):
        if board.king(engine_color) not in (chess.G1, chess.C1, chess.G8, chess.C8):
            score -= 50

    for color in [chess.WHITE, chess.BLACK]:
        multiplier = 1 if color == engine_color else -1

        pawns = board.pieces(chess.PAWN, color)
        pawn_files = [chess.square_file(sq) for sq in pawns]

        for f in range(8):
            count_on_file = pawn_files.count(f)
            if count_on_file > 1:
                score -= 15 * (count_on_file - 1) * multiplier

        rooks = board.pieces(chess.ROOK, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        enemy_pawn_files = [chess.square_file(sq) for sq in enemy_pawns]

        for sq in rooks:
            f = chess.square_file(sq)
            if pawn_files.count(f) == 0:
                if enemy_pawn_files.count(f) == 0:
                    score += 20 * multiplier
                else:
                    score += 10 * multiplier

        king_sq = board.king(color)
        if king_sq is not None:
            king_file = chess.square_file(king_sq)
            if pawn_files.count(king_file) == 0:
                if len(board.pieces(chess.QUEEN, not color)) + len(board.pieces(chess.ROOK, not color)) > 0:
                    score -= 20 * multiplier

    return score

def quiescence(board, alpha, beta, is_maximizing, engine_color, q_depth=0):
    global timeout_flag
    if timeout_flag or q_depth > 3:
        return evaluate_board(board, engine_color)

    stand_pat = evaluate_board(board, engine_color)

    if is_maximizing:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
    else:
        if stand_pat <= alpha:
            return alpha
        if stand_pat < beta:
            beta = stand_pat

    capture_moves = [m for m in board.legal_moves if board.is_capture(m)]
    capture_moves.sort(key=lambda m: mvv_lva_score(board, m), reverse=True)

    if is_maximizing:
        max_eval = stand_pat
        for move in capture_moves:
            check_time()
            if timeout_flag:
                break
            board.push(move)
            ev = quiescence(board, alpha, beta, False, engine_color, q_depth + 1)
            board.pop()
            max_eval = max(max_eval, ev)
            alpha = max(alpha, ev)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = stand_pat
        for move in capture_moves:
            check_time()
            if timeout_flag:
                break
            board.push(move)
            ev = quiescence(board, alpha, beta, True, engine_color, q_depth + 1)
            board.pop()
            min_eval = min(min_eval, ev)
            beta = min(beta, ev)
            if beta <= alpha:
                break
        return min_eval

def minimax(board, depth, alpha, beta, is_maximizing, engine_color):
    global timeout_flag
    if timeout_flag:
        return evaluate_board(board, engine_color)

    if board.is_game_over():
        return evaluate_board(board, engine_color)

    if depth == 0:
        return quiescence(board, alpha, beta, is_maximizing, engine_color, 0)

    legal_moves = list(board.legal_moves)
    legal_moves.sort(key=lambda m: mvv_lva_score(board, m), reverse=True)

    if is_maximizing:
        max_eval = -float('inf')
        for move in legal_moves:
            check_time()
            if timeout_flag:
                break
            board.push(move)
            ev = minimax(board, depth - 1, alpha, beta, False, engine_color)
            board.pop()
            max_eval = max(max_eval, ev)
            alpha = max(alpha, ev)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            check_time()
            if timeout_flag:
                break
            board.push(move)
            ev = minimax(board, depth - 1, alpha, beta, True, engine_color)
            board.pop()
            min_eval = min(min_eval, ev)
            beta = min(beta, ev)
            if beta <= alpha:
                break
        return min_eval

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
move_to_id = np.load(os.path.join(BASE_DIR, "move_to_id.npy"), allow_pickle=True).item()
id_to_move = {v: k for k, v in move_to_id.items()}
num_classes = len(move_to_id)

model = ChessResNet(num_classes)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "chess_resnet_model.pth"), map_location=torch.device('cpu')))
model.eval()

board = chess.Board()

while True:
    try:
        line = sys.stdin.readline().strip()
    except:
        break

    if not line:
        continue

    if line == "uci":
        print("id name MyChessResNet 1900")
        print("id author YourName")
        print("uciok")
        sys.stdout.flush()

    elif line == "isready":
        print("readyok")
        sys.stdout.flush()

    elif line.startswith("position"):
        parts = line.split()
        if "startpos" in parts:
            board.reset()
            if "moves" in parts:
                moves_idx = parts.index("moves")
                for m in parts[moves_idx + 1:]:
                    board.push(chess.Move.from_uci(m))
        elif "fen" in parts:
            fen_idx = parts.index("fen")
            if "moves" in parts:
                moves_idx = parts.index("moves")
                fen_str = " ".join(parts[fen_idx + 1:moves_idx])
                board.set_fen(fen_str)
                for m in parts[moves_idx + 1:]:
                    board.push(chess.Move.from_uci(m))
            else:
                fen_str = " ".join(parts[fen_idx + 1:])
                board.set_fen(fen_str)

    elif line.startswith("go"):
        # UCI time control: spend (clock + increment) / 30 per move.
        parts = line.split()
        max_time = DEFAULT_MAX_TIME
        if "movetime" in parts:
            max_time = int(parts[parts.index("movetime") + 1]) / 1000.0
        elif "wtime" in parts and "btime" in parts:
            time_key = "wtime" if board.turn == chess.WHITE else "btime"
            inc_key = "winc" if board.turn == chess.WHITE else "binc"
            time_left_ms = int(parts[parts.index(time_key) + 1])
            inc_ms = int(parts[parts.index(inc_key) + 1]) if inc_key in parts else 0
            max_time = (time_left_ms + inc_ms) / 1000.0 / 30.0
        MAX_TIME = max(MIN_MOVE_TIME, min(max_time, MAX_MOVE_TIME))

        matrix = board_to_matrix(board)
        tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)

        probabilities = torch.softmax(outputs, dim=1).squeeze()

        top_k = 10
        sorted_indices = torch.argsort(probabilities, descending=True)

        candidate_moves = []
        for idx in sorted_indices:
            move_id = idx.item()
            if move_id in id_to_move:
                predicted_move_uci = id_to_move[move_id]
                move = chess.Move.from_uci(predicted_move_uci)
                if move in board.legal_moves:
                    candidate_moves.append({"move": move, "prob": probabilities[idx].item()})
            if len(candidate_moves) >= top_k:
                break

        if not candidate_moves:
            for move in list(board.legal_moves)[:top_k]:
                candidate_moves.append({"move": move, "prob": 0.0})

        our_color = board.turn

        # Pre-move position score: the baseline used to detect blunders.
        current_score = evaluate_board(board, our_color)

        # Deeper safety search when the clock allows (classical games).
        max_depth = MINIMAX_DEPTH
        if MAX_TIME >= 3.0:
            max_depth = min(MAX_SEARCH_DEPTH, MINIMAX_DEPTH + 1)

        search_start = time.time()
        timeout_flag = False
        nodes_count = 0

        # Static baseline: a timeout can never leave scores at -inf.
        for item in candidate_moves:
            board.push(item["move"])
            if board.is_checkmate():
                item["score"] = 999999
            else:
                item["score"] = evaluate_board(board, our_color)
            item["last_completed_score"] = item["score"]
            board.pop()

        if DEEP_ANALYZE:
            for depth in range(1, max_depth + 1):
                if timeout_flag:
                    break

                for item in candidate_moves:
                    move = item["move"]
                    board.push(move)

                    if board.is_checkmate():
                        item["score"] = 999999
                    else:
                        item["score"] = minimax(board, depth - 1, -float('inf'), float('inf'), False, our_color)

                    board.pop()

                    if timeout_flag:
                        break

                if not timeout_flag:
                    for item in candidate_moves:
                        item["last_completed_score"] = item["score"]

        # Keep only moves that don't drop the score more than BLUNDER_THRESHOLD.
        safe_moves = [item for item in candidate_moves if item["last_completed_score"] >= current_score - BLUNDER_THRESHOLD]

        if not safe_moves:
            # Self-preservation: pick the move that loses the least material.
            best_legal_move = max(candidate_moves, key=lambda x: x["last_completed_score"])["move"]
        else:
            # Most human-like safe move: the one with the highest NN probability.
            safe_moves.sort(key=lambda x: x["prob"], reverse=True)

            # Opening variety (first 15 moves): weighted random among top-3 safe moves.
            if len(safe_moves) > 1 and board.fullmove_number < 15:
                pool = safe_moves[:3]
                moves = [m["move"] for m in pool]
                weights = [m["prob"] + 0.001 for m in pool]
                best_legal_move = random.choices(moves, weights=weights, k=1)[0]
            else:
                best_legal_move = safe_moves[0]["move"]

        print(f"bestmove {best_legal_move.uci()}")
        sys.stdout.flush()

    elif line == "quit":
        break