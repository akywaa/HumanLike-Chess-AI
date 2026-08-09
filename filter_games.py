import chess.pgn

INPUT_FILE = "lichess_db_standard_rated_2015-06.pgn"
OUTPUT_FILE = "filtered_1700_2100.pgn"

MIN_ELO = 1700
MAX_ELO = 2100
GAMES_TO_COLLECT = 100000

games_saved = 0
games_checked = 0

print(f"Filtering games, target: {GAMES_TO_COLLECT}")

with open(INPUT_FILE, "r", encoding="utf-8") as pgn_in, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as pgn_out:

    while games_saved < GAMES_TO_COLLECT:
        game = chess.pgn.read_game(pgn_in)

        if game is None:
            break

        games_checked += 1
        if games_checked % 5000 == 0:
            print(f"Games checked: {games_checked}, matches: {games_saved}")

        headers = game.headers
        event = headers.get("Event", "")
        white_elo_str = headers.get("WhiteElo", "?")
        black_elo_str = headers.get("BlackElo", "?")

        if "Blitz" not in event and "Rapid" not in event:
            continue

        try:
            white_elo = int(white_elo_str)
            black_elo = int(black_elo_str)
        except ValueError:
            continue

        if (MIN_ELO <= white_elo <= MAX_ELO) and (MIN_ELO <= black_elo <= MAX_ELO):
            if sum(1 for _ in game.mainline_moves()) < 10:
                continue

            print(game, file=pgn_out, end="\n\n")
            games_saved += 1

print(f"Done. Saved {games_saved} games to {OUTPUT_FILE}.")