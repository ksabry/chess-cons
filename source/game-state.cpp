#include "game-state.h"

GameState::GameState() :
	board {
		Tile::WhiteRook  , Tile::WhitePawn, Tile::Empty, Tile::Empty, Tile::Empty, Tile::Empty, Tile::BlackPawn, Tile::BlackRook  ,
		Tile::WhiteKnight, Tile::WhitePawn, Tile::Empty, Tile::Empty, Tile::Empty, Tile::Empty, Tile::BlackPawn, Tile::BlackKnight,
		Tile::WhiteBishop, Tile::WhitePawn, Tile::Empty, Tile::Empty, Tile::Empty, Tile::Empty, Tile::BlackPawn, Tile::BlackBishop,
		Tile::WhiteQueen , Tile::WhitePawn, Tile::Empty, Tile::Empty, Tile::Empty, Tile::Empty, Tile::BlackPawn, Tile::BlackQueen ,
		Tile::WhiteKing  , Tile::WhitePawn, Tile::Empty, Tile::Empty, Tile::Empty, Tile::Empty, Tile::BlackPawn, Tile::BlackKing  ,
		Tile::WhiteBishop, Tile::WhitePawn, Tile::Empty, Tile::Empty, Tile::Empty, Tile::Empty, Tile::BlackPawn, Tile::BlackBishop,
		Tile::WhiteKnight, Tile::WhitePawn, Tile::Empty, Tile::Empty, Tile::Empty, Tile::Empty, Tile::BlackPawn, Tile::BlackKnight,
		Tile::WhiteRook  , Tile::WhitePawn, Tile::Empty, Tile::Empty, Tile::Empty, Tile::Empty, Tile::BlackPawn, Tile::BlackRook  ,
	},
	move(Tile::White),
	castlesAvailable(CastleMove::BlackQueenSide | CastleMove::BlackKingSide | CastleMove::WhiteQueenSide | CastleMove::WhiteKingSide),
	enPassantPawn(9),
	whiteKingX(4),
	whiteKingY(0),
	blackKingX(4),
	blackKingY(7)
#ifdef PRINT_LAST_MOVE
	,
	lastMoveStartX(-1),
	lastMoveStartY(-1),
	lastMoveEndX(-1),
	lastMoveEndY(-1)
#endif
{
}

GameState::GameState(
	Tile* board,
	Tile move,
	CastleMove castlesAvailable,
	int_fast32_t enPassantPawn,
	int_fast32_t whiteKingX,
	int_fast32_t whiteKingY,
	int_fast32_t blackKingX,
	int_fast32_t blackKingY
)
{
	for (int_fast32_t i = 0; i < 64; i++)
	{
		this->board[i] = board[i];
	}
	this->move = move;
	this->castlesAvailable = castlesAvailable;
	this->enPassantPawn = enPassantPawn;
	this->whiteKingX = whiteKingX;
	this->whiteKingY = whiteKingY;
	this->blackKingX = blackKingX;
	this->blackKingY = blackKingY;
}

int_fast32_t GameState::WhitePieceScore() const
{
	int_fast32_t total = 0;
	for (int_fast32_t y = 0; y < 8; y++)
	{
		for (int_fast32_t x = 0; x < 8; x++)
		{
			Tile tile = GetTile(x, y);
			switch (tile)
			{
				// Values from an AlphaZero paper
				case Tile::WhitePawn:
					total += 100;
					break;
				case Tile::WhiteRook:
					total += 563;
					break;
				case Tile::WhiteKnight:
					total += 305;
					break;
				case Tile::WhiteBishop:
					total += 333;
					break;
				case Tile::WhiteQueen:
					total += 950;
					break;
			}
		}
	}
	return total;
}

int_fast32_t GameState::BlackPieceScore() const
{
	int_fast32_t total = 0;
	for (int_fast32_t y = 0; y < 8; y++)
	{
		for (int_fast32_t x = 0; x < 8; x++)
		{
			Tile tile = GetTile(x, y);
			switch (tile)
			{
				// Values from an AlphaZero paper
				case Tile::BlackPawn:
					total += 100;
					break;
				case Tile::BlackRook:
					total += 563;
					break;
				case Tile::BlackKnight:
					total += 305;
					break;
				case Tile::BlackBishop:
					total += 333;
					break;
				case Tile::BlackQueen:
					total += 950;
					break;
			}
		}
	}
	return total;
}

// TODO: we should probably cache this value on the instance, but we need to be careful with certain cases such as the repeated calls during a castling validity check
bool GameState::IsWhiteInCheck() const
{
	int_fast32_t tx, ty;

	// Check cardinal directions
	for (tx = whiteKingX - 1; tx >= 0; tx--)
	{
		Tile tile = GetTile(tx, whiteKingY);
		if (tile == Tile::BlackRook || tile == Tile::BlackQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
	}
	for (tx = whiteKingX + 1; tx < 8; tx++)
	{
		Tile tile = GetTile(tx, whiteKingY);
		if (tile == Tile::BlackRook || tile == Tile::BlackQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
	}
	for (ty = whiteKingY - 1; ty >= 0; ty--)
	{
		Tile tile = GetTile(whiteKingX, ty);
		if (tile == Tile::BlackRook || tile == Tile::BlackQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
	}
	for (ty = whiteKingY + 1; ty < 8; ty++)
	{
		Tile tile = GetTile(whiteKingX, ty);
		if (tile == Tile::BlackRook || tile == Tile::BlackQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
	}

	// Check diagonals
	tx = whiteKingX - 1;
	ty = whiteKingY - 1;
	while (tx >= 0 && ty >= 0)
	{
		Tile tile = GetTile(tx, ty);
		if (tile == Tile::BlackBishop || tile == Tile::BlackQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
		tx--;
		ty--;
	}
	tx = whiteKingX - 1;
	ty = whiteKingY + 1;
	while (tx >= 0 && ty < 8)
	{
		Tile tile = GetTile(tx, ty);
		if (tile == Tile::BlackBishop || tile == Tile::BlackQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
		tx--;
		ty++;
	}
	tx = whiteKingX + 1;
	ty = whiteKingY - 1;
	while (tx < 8 && ty >= 0)
	{
		Tile tile = GetTile(tx, ty);
		if (tile == Tile::BlackBishop || tile == Tile::BlackQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
		tx++;
		ty--;
	}
	tx = whiteKingX + 1;
	ty = whiteKingY + 1;
	while (tx < 8 && ty < 8)
	{
		Tile tile = GetTile(tx, ty);
		if (tile == Tile::BlackBishop || tile == Tile::BlackQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
		tx++;
		ty++;
	}

	// Check knights
	if (
		(
			whiteKingX >= 2 &&
			(
				whiteKingY >= 1 && GetTile(whiteKingX - 2, whiteKingY - 1) == Tile::BlackKnight ||
				whiteKingY <= 6 && GetTile(whiteKingX - 2, whiteKingY + 1) == Tile::BlackKnight
			)
		) ||
		(
			whiteKingX >= 1 &&
			(
				whiteKingY >= 2 && GetTile(whiteKingX - 1, whiteKingY - 2) == Tile::BlackKnight ||
				whiteKingY <= 5 && GetTile(whiteKingX - 1, whiteKingY + 2) == Tile::BlackKnight
			)
		) ||
		(
			whiteKingX <= 5 &&
			(
				whiteKingY >= 1 && GetTile(whiteKingX + 2, whiteKingY - 1) == Tile::BlackKnight ||
				whiteKingY <= 6 && GetTile(whiteKingX + 2, whiteKingY + 1) == Tile::BlackKnight
			)
		) ||
		(
			whiteKingX <= 6 &&
			(
				whiteKingY >= 2 && GetTile(whiteKingX + 1, whiteKingY - 2) == Tile::BlackKnight ||
				whiteKingY <= 5 && GetTile(whiteKingX + 1, whiteKingY + 2) == Tile::BlackKnight
			)
		)
	)
	{
		return true;
	}

	// Check kings and pawns
	// Note that while two kings next to each other is an invalid position, the code which checks this very validity relies on this method so this is necessary to implement
	if (
		(
			whiteKingY >= 1 && GetTile(whiteKingX, whiteKingY - 1) == Tile::BlackKing ||
			whiteKingY <= 6 && GetTile(whiteKingX, whiteKingY + 1) == Tile::BlackKing
		) ||
		(
			whiteKingX >= 1 &&
			(
				GetTile(whiteKingX - 1, whiteKingY) == Tile::BlackKing ||
				whiteKingY >= 1 && GetTile(whiteKingX - 1, whiteKingY - 1) == Tile::BlackKing ||
				whiteKingY <= 6 && (GetTile(whiteKingX - 1, whiteKingY + 1) == Tile::BlackKing || GetTile(whiteKingX - 1, whiteKingY + 1) == Tile::BlackPawn)
				)
		) ||
		(
			whiteKingX <= 6 &&
			(
				GetTile(whiteKingX + 1, whiteKingY) == Tile::BlackKing ||
				whiteKingY >= 1 && GetTile(whiteKingX + 1, whiteKingY - 1) == Tile::BlackKing ||
				whiteKingY <= 6 && (GetTile(whiteKingX + 1, whiteKingY + 1) == Tile::BlackKing || GetTile(whiteKingX + 1, whiteKingY + 1) == Tile::BlackPawn)
			)
		)
	)
	{
		return true;
	}

	return false;
}

// TODO: we should probably cache this value on the instance, but we need to be careful with certain cases such as the repeated calls during a castling validity check
bool GameState::IsBlackInCheck() const
{
	int_fast32_t tx, ty;

	// Check cardinal directions
	for (tx = blackKingX - 1; tx >= 0; tx--)
	{
		Tile tile = GetTile(tx, blackKingY);
		if (tile == Tile::WhiteRook || tile == Tile::WhiteQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
	}
	for (tx = blackKingX + 1; tx < 8; tx++)
	{
		Tile tile = GetTile(tx, blackKingY);
		if (tile == Tile::WhiteRook || tile == Tile::WhiteQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
	}
	for (ty = blackKingY - 1; ty >= 0; ty--)
	{
		Tile tile = GetTile(blackKingX, ty);
		if (tile == Tile::WhiteRook || tile == Tile::WhiteQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
	}
	for (ty = blackKingY + 1; ty < 8; ty++)
	{
		Tile tile = GetTile(blackKingX, ty);
		if (tile == Tile::WhiteRook || tile == Tile::WhiteQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
	}

	// Check diagonals
	tx = blackKingX - 1;
	ty = blackKingY - 1;
	while (tx >= 0 && ty >= 0)
	{
		Tile tile = GetTile(tx, ty);
		if (tile == Tile::WhiteBishop || tile == Tile::WhiteQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
		tx--;
		ty--;
	}
	tx = blackKingX - 1;
	ty = blackKingY + 1;
	while (tx >= 0 && ty < 8)
	{
		Tile tile = GetTile(tx, ty);
		if (tile == Tile::WhiteBishop || tile == Tile::WhiteQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
		tx--;
		ty++;
	}
	tx = blackKingX + 1;
	ty = blackKingY - 1;
	while (tx < 8 && ty >= 0)
	{
		Tile tile = GetTile(tx, ty);
		if (tile == Tile::WhiteBishop || tile == Tile::WhiteQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
		tx++;
		ty--;
	}
	tx = blackKingX + 1;
	ty = blackKingY + 1;
	while (tx < 8 && ty < 8)
	{
		Tile tile = GetTile(tx, ty);
		if (tile == Tile::WhiteBishop || tile == Tile::WhiteQueen)
		{
			return true;
		}
		else if (tile != Tile::Empty)
		{
			break;
		}
		tx++;
		ty++;
	}

	// Check knights
	if (
		(
			blackKingX >= 2 &&
			(
				blackKingY >= 1 && GetTile(blackKingX - 2, blackKingY - 1) == Tile::WhiteKnight ||
				blackKingY <= 6 && GetTile(blackKingX - 2, blackKingY + 1) == Tile::WhiteKnight
			)
		) ||
		(
			blackKingX >= 1 &&
			(
				blackKingY >= 2 && GetTile(blackKingX - 1, blackKingY - 2) == Tile::WhiteKnight ||
				blackKingY <= 5 && GetTile(blackKingX - 1, blackKingY + 2) == Tile::WhiteKnight
			)
		) ||
		(
			blackKingX <= 5 &&
			(
				blackKingY >= 1 && GetTile(blackKingX + 2, blackKingY - 1) == Tile::WhiteKnight ||
				blackKingY <= 6 && GetTile(blackKingX + 2, blackKingY + 1) == Tile::WhiteKnight
			)
		) ||
		(
			blackKingX <= 6 &&
			(
				blackKingY >= 2 && GetTile(blackKingX + 1, blackKingY - 2) == Tile::WhiteKnight ||
				blackKingY <= 5 && GetTile(blackKingX + 1, blackKingY + 2) == Tile::WhiteKnight
			)
		)
	)
	{
		return true;
	}

	// Check kings and pawns
	// Note that while two kings next to each other is an invalid position, the code which checks this very validity relies on this method so this is necessary to implement
	if (
		(
			blackKingY >= 1 && GetTile(blackKingX, blackKingY - 1) == Tile::WhiteKing ||
			blackKingY <= 6 && GetTile(blackKingX, blackKingY + 1) == Tile::WhiteKing
		) ||
		(
			blackKingX >= 1 &&
			(
				GetTile(blackKingX - 1, blackKingY) == Tile::WhiteKing ||
				blackKingY >= 1 && (GetTile(blackKingX - 1, blackKingY - 1) == Tile::WhiteKing || GetTile(blackKingX - 1, blackKingY - 1) == Tile::WhitePawn) ||
				blackKingY <= 6 && GetTile(blackKingX - 1, blackKingY + 1) == Tile::WhiteKing
				)
		) ||
		(
			blackKingX <= 6 &&
			(
				GetTile(blackKingX + 1, blackKingY) == Tile::WhiteKing ||
				blackKingY >= 1 && (GetTile(blackKingX + 1, blackKingY - 1) == Tile::WhiteKing || GetTile(blackKingX + 1, blackKingY - 1) == Tile::WhitePawn) ||
				blackKingY <= 6 && GetTile(blackKingX + 1, blackKingY + 1) == Tile::WhiteKing
			)
		)
	)
	{
		return true;
	}

	return false;
}

void GameState::NextGameStates(std::vector<GameState> & output) const
{
	GameState nextState;
	int_fast32_t x, y, tx, ty;
	CastleMove castlesAvailableModifier;

	if (move == Tile::White)
	{
		for (x = 0; x < 8; x++)
		{
			for (y = 0; y < 8; y++)
			{
				Tile tile = GetTile(x, y);
				castlesAvailableModifier = CastleMove::WhiteKingSide | CastleMove::WhiteQueenSide | CastleMove::BlackKingSide | CastleMove::BlackQueenSide;
				switch (tile)
				{
				case Tile::WhitePawn:
					if (GetTile(x, y + 1) == Tile::Empty)
					{
						if (y == 6)
						{
							// White Pawn Promotion
							CloneForNextMove;
							nextState.SetTile(x, 6, Tile::Empty);
							nextState.SetTile(x, 7, Tile::WhiteQueen);
							SetLastMove(x, 6, x, 7);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 6, Tile::Empty);
							nextState.SetTile(x, 7, Tile::WhiteKnight);
							SetLastMove(x, 6, x, 7);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 6, Tile::Empty);
							nextState.SetTile(x, 7, Tile::WhiteBishop);
							SetLastMove(x, 6, x, 7);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 6, Tile::Empty);
							nextState.SetTile(x, 7, Tile::WhiteRook);
							SetLastMove(x, 6, x, 7);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							// White Pawn Forward One
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x, y + 1, Tile::WhitePawn);
							SetLastMove(x, y, x, y + 1);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}

							if (y == 1 && GetTile(x, 3) == Tile::Empty)
							{
								// White Pawn Forward Two
								CloneForNextMove;
								nextState.SetTile(x, 1, Tile::Empty);
								nextState.SetTile(x, 3, Tile::WhitePawn);
								SetLastMove(x, 1, x, 3);
								if (!nextState.IsWhiteInCheck())
								{
									nextState.enPassantPawn = x;
									output.push_back(nextState);
								}
							}
						}
					}
					if (x != 0 && (GetTile(x - 1, y + 1) & Tile::Black) != Tile::Empty)
					{
						if (y == 6)
						{
							// White Pawn Taking Left Promotion
							CloneForNextMove;
							nextState.SetTile(x, 6, Tile::Empty);
							nextState.SetTile(x - 1, 7, Tile::WhiteQueen);
							SetLastMove(x, 6, x - 1, 7);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
							
							CloneForNextMove;
							nextState.SetTile(x, 6, Tile::Empty);
							nextState.SetTile(x - 1, 7, Tile::WhiteKnight);
							SetLastMove(x, 6, x - 1, 7);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 6, Tile::Empty);
							nextState.SetTile(x - 1, 7, Tile::WhiteBishop);
							SetLastMove(x, 6, x - 1, 7);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 6, Tile::Empty);
							nextState.SetTile(x - 1, 7, Tile::WhiteRook);
							SetLastMove(x, 6, x - 1, 7);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							// White Pawn Taking Left
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 1, y + 1, Tile::WhitePawn);
							SetLastMove(x, y, x - 1, y + 1);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
					}
					if (x != 7 && (GetTile(x + 1, y + 1) & Tile::Black) != Tile::Empty)
					{
						if (y == 6)
						{
							// White Pawn Taking Right Promotion
							CloneForNextMove;
							nextState.SetTile(x, 6, Tile::Empty);
							nextState.SetTile(x + 1, 7, Tile::WhiteQueen);
							SetLastMove(x, 6, x + 1, 7);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 6, Tile::Empty);
							nextState.SetTile(x + 1, 7, Tile::WhiteKnight);
							SetLastMove(x, 6, x + 1, 7);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 6, Tile::Empty);
							nextState.SetTile(x + 1, 7, Tile::WhiteBishop);
							SetLastMove(x, 6, x + 1, 7);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 6, Tile::Empty);
							nextState.SetTile(x + 1, 7, Tile::WhiteRook);
							SetLastMove(x, 6, x + 1, 7);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							// White Pawn Taking Right
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 1, y + 1, Tile::WhitePawn);
							SetLastMove(x, y, x + 1, y + 1);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
					}
					if (y == 4 && (enPassantPawn == x - 1 || enPassantPawn == x + 1))
					{
						// White Pawn Taking En Passant
						CloneForNextMove;
						nextState.SetTile(x, y, Tile::Empty);
						nextState.SetTile(enPassantPawn, 4, Tile::Empty);
						nextState.SetTile(enPassantPawn, 5, Tile::WhitePawn);
						SetLastMove(x, y, enPassantPawn, 5);
						if (!nextState.IsWhiteInCheck())
						{
							output.push_back(nextState);
						}
					}
					break;

				case Tile::WhiteRook:
					if (y == 0 && x == 0)
					{
						castlesAvailableModifier = CastleMove::WhiteKingSide | CastleMove::BlackKingSide | CastleMove::BlackQueenSide;
					}
					else if (y == 0 && x == 7)
					{
						castlesAvailableModifier = CastleMove::WhiteQueenSide | CastleMove::BlackKingSide | CastleMove::BlackQueenSide;
					}

					for (tx = x - 1; tx >= 0; tx--)
					{
						if (GetTile(tx, y) == Tile::Empty)
						{
							// White Rook Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, y, tile);
							SetLastMove(x, y, tx, y);
							if (!nextState.IsWhiteInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						else
						{
							// White Rook Taking
							if (!IsWhite(GetTile(tx, y)))
							{
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, y, tile);
								nextState.castlesAvailable &= castlesAvailableModifier;
								SetLastMove(x, y, tx, y);
								if (!nextState.IsWhiteInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					for (tx = x + 1; tx < 8; tx++)
					{
						if (GetTile(tx, y) == Tile::Empty)
						{
							// White Rook Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, y, tile);
							SetLastMove(x, y, tx, y);
							if (!nextState.IsWhiteInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(tx, y)))
							{
								// White Rook Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, y, tile);
								SetLastMove(x, y, tx, y);
								if (!nextState.IsWhiteInCheck())
								{
									nextState.castlesAvailable &= castlesAvailableModifier;
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					for (ty = y - 1; ty >= 0; ty--)
					{
						if (GetTile(x, ty) == Tile::Empty)
						{
							// White Rook Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x, ty, tile);
							SetLastMove(x, y, x, ty);
							if (!nextState.IsWhiteInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(x, ty)))
							{
								// White Rook Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(x, ty, tile);
								SetLastMove(x, y, x, ty);
								if (!nextState.IsWhiteInCheck())
								{
									nextState.castlesAvailable &= castlesAvailableModifier;
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					for (ty = y + 1; ty < 8; ty++)
					{
						if (GetTile(x, ty) == Tile::Empty)
						{
							// White Rook Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x, ty, tile);
							SetLastMove(x, y, x, ty);
							if (!nextState.IsWhiteInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(x, ty)))
							{
								// White Rook Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(x, ty, tile);
								SetLastMove(x, y, x, ty);
								if (!nextState.IsWhiteInCheck())
								{
									nextState.castlesAvailable &= castlesAvailableModifier;
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					break;

				case Tile::WhiteKnight:
					if (x >= 2)
					{
						if (y >= 1 && !IsWhite(GetTile(x - 2, y - 1)))
						{
							// White Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 2, y - 1, tile);
							SetLastMove(x, y, x - 2, y - 1);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						if (y <= 6 && !IsWhite(GetTile(x - 2, y + 1)))
						{
							// White Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 2, y + 1, tile);
							SetLastMove(x, y, x - 2, y + 1);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
					}
					if (x >= 1)
					{
						if (y >= 2 && !IsWhite(GetTile(x - 1, y - 2)))
						{
							// White Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 1, y - 2, tile);
							SetLastMove(x, y, x - 1, y - 2);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						if (y <= 5 && !IsWhite(GetTile(x - 1, y + 2)))
						{
							// White Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 1, y + 2, tile);
							SetLastMove(x, y, x - 1, y + 2);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
					}
					if (x <= 5)
					{
						if (y >= 1 && !IsWhite(GetTile(x + 2, y - 1)))
						{
							// White Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 2, y - 1, tile);
							SetLastMove(x, y, x + 2, y - 1);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						if (y <= 6 && !IsWhite(GetTile(x + 2, y + 1)))
						{
							// White Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 2, y + 1, tile);
							SetLastMove(x, y, x + 2, y + 1);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
					}
					if (x <= 6)
					{
						if (y >= 2 && !IsWhite(GetTile(x + 1, y - 2)))
						{
							// White Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 1, y - 2, tile);
							SetLastMove(x, y, x + 1, y - 2);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						if (y <= 5 && !IsWhite(GetTile(x + 1, y + 2)))
						{
							// White Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 1, y + 2, tile);
							SetLastMove(x, y, x + 1, y + 2);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
					}
					break;

				case Tile::WhiteBishop:
					tx = x - 1;
					ty = y - 1;
					while (tx >= 0 && ty >= 0)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// White Bishop Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(tx, ty)))
							{
								// White Bishop Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsWhiteInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx--;
						ty--;
					}
					tx = x - 1;
					ty = y + 1;
					while (tx >= 0 && ty < 8)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// White Bishop Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(tx, ty)))
							{
								// White Bishop Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsWhiteInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx--;
						ty++;
					}
					tx = x + 1;
					ty = y - 1;
					while (tx < 8 && ty >= 0)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// White Bishop Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(tx, ty)))
							{
								// White Bishop Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsWhiteInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx++;
						ty--;
					}
					tx = x + 1;
					ty = y + 1;
					while (tx < 8 && ty < 8)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// White Bishop Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(tx, ty)))
							{
								// White Bishop Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsWhiteInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx++;
						ty++;
					}
					break;

				case Tile::WhiteQueen:
					for (tx = x - 1; tx >= 0; tx--)
					{
						if (GetTile(tx, y) == Tile::Empty)
						{
							// White Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, y, tile);
							SetLastMove(x, y, tx, y);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(tx, y)))
							{
								// White Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, y, tile);
								SetLastMove(x, y, tx, y);
								if (!nextState.IsWhiteInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					for (tx = x + 1; tx < 8; tx++)
					{
						if (GetTile(tx, y) == Tile::Empty)
						{
							// White Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, y, tile);
							SetLastMove(x, y, tx, y);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(tx, y)))
							{
								// White Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, y, tile);
								SetLastMove(x, y, tx, y);
								if (!nextState.IsWhiteInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					for (ty = y - 1; ty >= 0; ty--)
					{
						if (GetTile(x, ty) == Tile::Empty)
						{
							// White Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x, ty, tile);
							SetLastMove(x, y, x, ty);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(x, ty)))
							{
								// White Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(x, ty, tile);
								SetLastMove(x, y, x, ty);
								if (!nextState.IsWhiteInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					for (ty = y + 1; ty < 8; ty++)
					{
						if (GetTile(x, ty) == Tile::Empty)
						{
							// White Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x, ty, tile);
							SetLastMove(x, y, x, ty);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(x, ty)))
							{
								// White Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(x, ty, tile);
								SetLastMove(x, y, x, ty);
								if (!nextState.IsWhiteInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					tx = x - 1;
					ty = y - 1;
					while (tx >= 0 && ty >= 0)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// White Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(tx, ty)))
							{
								// White Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsWhiteInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx--;
						ty--;
					}
					tx = x - 1;
					ty = y + 1;
					while (tx >= 0 && ty < 8)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// White Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(tx, ty)))
							{
								// White Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsWhiteInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx--;
						ty++;
					}
					tx = x + 1;
					ty = y - 1;
					while (tx < 8 && ty >= 0)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// White Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(tx, ty)))
							{
								// White Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsWhiteInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx++;
						ty--;
					}
					tx = x + 1;
					ty = y + 1;
					while (tx < 8 && ty < 8)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// White Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsWhiteInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsWhite(GetTile(tx, ty)))
							{
								// White Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsWhiteInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx++;
						ty++;
					}
					break;

				case Tile::WhiteKing:
					castlesAvailableModifier = CastleMove::BlackKingSide | CastleMove::BlackQueenSide;
					if (y >= 1 && !IsWhite(GetTile(x, y - 1)))
					{
						// White King Moving/Taking
						CloneForNextMove;
						nextState.SetTile(x, y, Tile::Empty);
						nextState.SetTile(x, y - 1, tile);
						nextState.whiteKingX = x; nextState.whiteKingY = y - 1;
						SetLastMove(x, y, x, y - 1);
						if (!nextState.IsWhiteInCheck())
						{
							nextState.castlesAvailable &= castlesAvailableModifier;
							output.push_back(nextState);
						}
					}
					if (y <= 6 && !IsWhite(GetTile(x, y + 1)))
					{
						// White King Moving/Taking
						CloneForNextMove;
						nextState.SetTile(x, y, Tile::Empty);
						nextState.SetTile(x, y + 1, tile);
						nextState.whiteKingX = x; nextState.whiteKingY = y + 1;
						SetLastMove(x, y, x, y + 1);
						if (!nextState.IsWhiteInCheck())
						{
							nextState.castlesAvailable &= castlesAvailableModifier;
							output.push_back(nextState);
						}
					}
					if (x >= 1)
					{
						if (!IsWhite(GetTile(x - 1, y)))
						{
							// White King Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 1, y, tile);
							nextState.whiteKingX = x - 1; nextState.whiteKingY = y;
							SetLastMove(x, y, x - 1, y);
							if (!nextState.IsWhiteInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						if (y >= 1 && !IsWhite(GetTile(x - 1, y - 1)))
						{
							// White King Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 1, y - 1, tile);
							nextState.whiteKingX = x - 1; nextState.whiteKingY = y - 1;
							SetLastMove(x, y, x - 1, y - 1);
							if (!nextState.IsWhiteInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						if (y <= 6 && !IsWhite(GetTile(x - 1, y + 1)))
						{
							// White King Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 1, y + 1, tile);
							nextState.whiteKingX = x - 1; nextState.whiteKingY = y + 1;
							SetLastMove(x, y, x - 1, y + 1);
							if (!nextState.IsWhiteInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
					}
					if (x <= 6)
					{
						if (!IsWhite(GetTile(x + 1, y)))
						{
							// White King Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 1, y, tile);
							nextState.whiteKingX = x + 1; nextState.whiteKingY = y;
							SetLastMove(x, y, x + 1, y);
							if (!nextState.IsWhiteInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						if (y >= 1 && !IsWhite(GetTile(x + 1, y - 1)))
						{
							// White King Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 1, y - 1, tile);
							nextState.whiteKingX = x + 1; nextState.whiteKingY = y - 1;
							SetLastMove(x, y, x + 1, y - 1);
							if (!nextState.IsWhiteInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						if (y <= 6 && !IsWhite(GetTile(x + 1, y + 1)))
						{
							// White King Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 1, y + 1, tile);
							nextState.whiteKingX = x + 1; nextState.whiteKingY = y + 1;
							SetLastMove(x, y, x + 1, y + 1);
							if (!nextState.IsWhiteInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
					}
					break;
				}
			}
		}
		if (!IsWhiteInCheck())
		{
			if ((castlesAvailable & CastleMove::WhiteKingSide) != CastleMove::None && GetTile(5, 0) == Tile::Empty && GetTile(6, 0) == Tile::Empty)
			{
				CloneForNextMove;
				nextState.SetTile(4, 0, Tile::Empty);
				nextState.SetTile(7, 0, Tile::Empty);
				nextState.SetTile(5, 0, Tile::WhiteKing);
				nextState.whiteKingX = 5; nextState.whiteKingY = 0;
				if (!nextState.IsWhiteInCheck())
				{
					nextState.SetTile(5, 0, Tile::WhiteRook);
					nextState.SetTile(6, 0, Tile::WhiteKing);
					nextState.whiteKingX = 6;
					SetLastMove(4, 0, 6, 0);
					if (!nextState.IsWhiteInCheck())
					{
						nextState.castlesAvailable &= CastleMove::BlackKingSide | CastleMove::BlackQueenSide;
						output.push_back(nextState);
					}
				}
			}
			if ((castlesAvailable & CastleMove::WhiteQueenSide) != CastleMove::None && GetTile(1, 0) == Tile::Empty && GetTile(2, 0) == Tile::Empty && GetTile(3, 0) == Tile::Empty)
			{
				CloneForNextMove;
				nextState.SetTile(0, 0, Tile::Empty);
				nextState.SetTile(4, 0, Tile::Empty);
				nextState.SetTile(3, 0, Tile::WhiteKing);
				nextState.whiteKingX = 3; nextState.whiteKingY = 0;
				if (!nextState.IsWhiteInCheck())
				{
					nextState.SetTile(2, 0, Tile::WhiteKing);
					nextState.SetTile(3, 0, Tile::WhiteRook);
					nextState.whiteKingX = 2;
					SetLastMove(4, 0, 2, 0);
					if (!nextState.IsWhiteInCheck())
					{
						nextState.castlesAvailable &= CastleMove::BlackKingSide | CastleMove::BlackQueenSide;
						output.push_back(nextState);
					}
				}
			}
		}
	}
	else
	{
		for (x = 0; x < 8; x++)
		{
			for (y = 0; y < 8; y++)
			{
				Tile tile = GetTile(x, y);
				castlesAvailableModifier = CastleMove::WhiteKingSide | CastleMove::WhiteQueenSide | CastleMove::BlackKingSide | CastleMove::BlackQueenSide;
				switch (tile)
				{
				case Tile::BlackPawn:
					if (GetTile(x, y - 1) == Tile::Empty)
					{
						if (y == 1)
						{
							// Black Pawn Promotion
							CloneForNextMove;
							nextState.SetTile(x, 1, Tile::Empty);
							nextState.SetTile(x, 0, Tile::BlackQueen);
							SetLastMove(x, 1, x, 0);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 1, Tile::Empty);
							nextState.SetTile(x, 0, Tile::BlackKnight);
							SetLastMove(x, 1, x, 0);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 1, Tile::Empty);
							nextState.SetTile(x, 0, Tile::BlackBishop);
							SetLastMove(x, 1, x, 0);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 1, Tile::Empty);
							nextState.SetTile(x, 0, Tile::BlackRook);
							SetLastMove(x, 1, x, 0);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							// Black Pawn Forward One
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x, y - 1, Tile::BlackPawn);
							SetLastMove(x, y, x, y - 1);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}

							if (y == 6 && GetTile(x, 4) == Tile::Empty)
							{
								// Black Pawn Forward Two
								CloneForNextMove;
								nextState.SetTile(x, 6, Tile::Empty);
								nextState.SetTile(x, 4, Tile::BlackPawn);
								SetLastMove(x, 6, x, 4);
								if (!nextState.IsBlackInCheck())
								{
									nextState.enPassantPawn = x;
									output.push_back(nextState);
								}
							}
						}
					}
					if (x != 7 && (GetTile(x + 1, y - 1) & Tile::White) != Tile::Empty)
					{
						if (y == 1)
						{
							// Black Pawn Taking Left Promotion
							CloneForNextMove;
							nextState.SetTile(x, 1, Tile::Empty);
							nextState.SetTile(x + 1, 0, Tile::BlackQueen);
							SetLastMove(x, 1, x + 1, 0);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 1, Tile::Empty);
							nextState.SetTile(x + 1, 0, Tile::BlackKnight);
							SetLastMove(x, 1, x + 1, 0);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 1, Tile::Empty);
							nextState.SetTile(x + 1, 0, Tile::BlackBishop);
							SetLastMove(x, 1, x + 1, 0);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 1, Tile::Empty);
							nextState.SetTile(x + 1, 0, Tile::BlackRook);
							SetLastMove(x, 1, x + 1, 0);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							// Black Pawn Taking Left
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 1, y - 1, Tile::BlackPawn);
							SetLastMove(x, y, x + 1, y - 1);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
					}
					if (x != 0 && (GetTile(x - 1, y - 1) & Tile::White) != Tile::Empty)
					{
						if (y == 1)
						{
							// Black Pawn Taking Right Promotion
							CloneForNextMove;
							nextState.SetTile(x, 1, Tile::Empty);
							nextState.SetTile(x - 1, 0, Tile::BlackQueen);
							SetLastMove(x, 1, x - 1, 0);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 1, Tile::Empty);
							nextState.SetTile(x - 1, 0, Tile::BlackKnight);
							SetLastMove(x, 1, x - 1, 0);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 1, Tile::Empty);
							nextState.SetTile(x - 1, 0, Tile::BlackBishop);
							SetLastMove(x, 1, x - 1, 0);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}

							CloneForNextMove;
							nextState.SetTile(x, 1, Tile::Empty);
							nextState.SetTile(x - 1, 0, Tile::BlackRook);
							SetLastMove(x, 1, x - 1, 0);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							// Black Pawn Taking Right
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 1, y - 1, Tile::BlackPawn);
							SetLastMove(x, y, x - 1, y - 1);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
					}
					if (y == 3 && (enPassantPawn == x - 1 || enPassantPawn == x + 1))
					{
						// Black Pawn Taking En Passant
						CloneForNextMove;
						nextState.SetTile(x, y, Tile::Empty);
						nextState.SetTile(enPassantPawn, 3, Tile::Empty);
						nextState.SetTile(enPassantPawn, 2, Tile::BlackPawn);
						SetLastMove(x, y, enPassantPawn, 2);
						if (!nextState.IsBlackInCheck())
						{
							output.push_back(nextState);
						}
					}
					break;

				case Tile::BlackRook:
					if (y == 7 && x == 0)
					{
						castlesAvailableModifier = CastleMove::WhiteKingSide | CastleMove::WhiteQueenSide | CastleMove::BlackKingSide;
					}
					else if (y == 7 && x == 7)
					{
						castlesAvailableModifier = CastleMove::WhiteKingSide | CastleMove::WhiteQueenSide | CastleMove::BlackQueenSide;
					}

					for (tx = x - 1; tx >= 0; tx--)
					{
						if (GetTile(tx, y) == Tile::Empty)
						{
							// Black Rook Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, y, tile);
							SetLastMove(x, y, tx, y);
							if (!nextState.IsBlackInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(tx, y)))
							{
								// Black Rook Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, y, tile);
								SetLastMove(x, y, tx, y);
								if (!nextState.IsBlackInCheck())
								{
									nextState.castlesAvailable &= castlesAvailableModifier;
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					for (tx = x + 1; tx < 8; tx++)
					{
						if (GetTile(tx, y) == Tile::Empty)
						{
							// Black Rook Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, y, tile);
							SetLastMove(x, y, tx, y);
							if (!nextState.IsBlackInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(tx, y)))
							{
								// Black Rook Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, y, tile);
								SetLastMove(x, y, tx, y);
								if (!nextState.IsBlackInCheck())
								{
									nextState.castlesAvailable &= castlesAvailableModifier;
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					for (ty = y - 1; ty >= 0; ty--)
					{
						if (GetTile(x, ty) == Tile::Empty)
						{
							// Black Rook Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x, ty, tile);
							SetLastMove(x, y, x, ty);
							if (!nextState.IsBlackInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(x, ty)))
							{
								// Black Rook Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(x, ty, tile);
								SetLastMove(x, y, x, ty);
								if (!nextState.IsBlackInCheck())
								{
									nextState.castlesAvailable &= castlesAvailableModifier;
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					for (ty = y + 1; ty < 8; ty++)
					{
						if (GetTile(x, ty) == Tile::Empty)
						{
							// Black Rook Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x, ty, tile);
							SetLastMove(x, y, x, ty);
							if (!nextState.IsBlackInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(x, ty)))
							{
								// Black Rook Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(x, ty, tile);
								SetLastMove(x, y, x, ty);
								if (!nextState.IsBlackInCheck())
								{
									nextState.castlesAvailable &= castlesAvailableModifier;
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					break;

				case Tile::BlackKnight:
					if (x >= 2)
					{
						if (y >= 1 && !IsBlack(GetTile(x - 2, y - 1)))
						{
							// Black Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 2, y - 1, tile);
							SetLastMove(x, y, x - 2, y - 1);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						if (y <= 6 && !IsBlack(GetTile(x - 2, y + 1)))
						{
							// Black Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 2, y + 1, tile);
							SetLastMove(x, y, x - 2, y + 1);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
					}
					if (x >= 1)
					{
						if (y >= 2 && !IsBlack(GetTile(x - 1, y - 2)))
						{
							// Black Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 1, y - 2, tile);
							SetLastMove(x, y, x - 1, y - 2);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						if (y <= 5 && !IsBlack(GetTile(x - 1, y + 2)))
						{
							// Black Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 1, y + 2, tile);
							SetLastMove(x, y, x - 1, y + 2);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
					}
					if (x <= 5)
					{
						if (y >= 1 && !IsBlack(GetTile(x + 2, y - 1)))
						{
							// Black Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 2, y - 1, tile);
							SetLastMove(x, y, x + 2, y - 1);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						if (y <= 6 && !IsBlack(GetTile(x + 2, y + 1)))
						{
							// Black Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 2, y + 1, tile);
							SetLastMove(x, y, x + 2, y + 1);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
					}
					if (x <= 6)
					{
						if (y >= 2 && !IsBlack(GetTile(x + 1, y - 2)))
						{
							// Black Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 1, y - 2, tile);
							SetLastMove(x, y, x + 1, y - 2);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						if (y <= 5 && !IsBlack(GetTile(x + 1, y + 2)))
						{
							// Black Knight Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 1, y + 2, tile);
							SetLastMove(x, y, x + 1, y + 2);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
					}
					break;

				case Tile::BlackBishop:
					tx = x - 1;
					ty = y - 1;
					while (tx >= 0 && ty >= 0)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// Black Bishop Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(tx, ty)))
							{
								// Black Bishop Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsBlackInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx--;
						ty--;
					}
					tx = x - 1;
					ty = y + 1;
					while (tx >= 0 && ty < 8)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// Black Bishop Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(tx, ty)))
							{
								// Black Bishop Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsBlackInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx--;
						ty++;
					}
					tx = x + 1;
					ty = y - 1;
					while (tx < 8 && ty >= 0)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// Black Bishop Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(tx, ty)))
							{
								// Black Bishop Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsBlackInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx++;
						ty--;
					}
					tx = x + 1;
					ty = y + 1;
					while (tx < 8 && ty < 8)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// Black Bishop Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(tx, ty)))
							{
								// Black Bishop Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsBlackInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx++;
						ty++;
					}
					break;

				case Tile::BlackQueen:
					for (tx = x - 1; tx >= 0; tx--)
					{
						if (GetTile(tx, y) == Tile::Empty)
						{
							// Black Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, y, tile);
							SetLastMove(x, y, tx, y);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(tx, y)))
							{
								// Black Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, y, tile);
								SetLastMove(x, y, tx, y);
								if (!nextState.IsBlackInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					for (tx = x + 1; tx < 8; tx++)
					{
						if (GetTile(tx, y) == Tile::Empty)
						{
							// Black Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, y, tile);
							SetLastMove(x, y, tx, y);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(tx, y)))
							{
								// Black Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, y, tile);
								SetLastMove(x, y, tx, y);
								if (!nextState.IsBlackInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					for (ty = y - 1; ty >= 0; ty--)
					{
						if (GetTile(x, ty) == Tile::Empty)
						{
							// Black Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x, ty, tile);
							SetLastMove(x, y, x, ty);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(x, ty)))
							{
								// Black Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(x, ty, tile);
								SetLastMove(x, y, x, ty);
								if (!nextState.IsBlackInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					for (ty = y + 1; ty < 8; ty++)
					{
						if (GetTile(x, ty) == Tile::Empty)
						{
							// Black Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x, ty, tile);
							SetLastMove(x, y, x, ty);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(x, ty)))
							{
								// Black Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(x, ty, tile);
								SetLastMove(x, y, x, ty);
								if (!nextState.IsBlackInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
					}
					tx = x - 1;
					ty = y - 1;
					while (tx >= 0 && ty >= 0)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// Black Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(tx, ty)))
							{
								// Black Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsBlackInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx--;
						ty--;
					}
					tx = x - 1;
					ty = y + 1;
					while (tx >= 0 && ty < 8)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// Black Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(tx, ty)))
							{
								// Black Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsBlackInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx--;
						ty++;
					}
					tx = x + 1;
					ty = y - 1;
					while (tx < 8 && ty >= 0)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// Black Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(tx, ty)))
							{
								// Black Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsBlackInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx++;
						ty--;
					}
					tx = x + 1;
					ty = y + 1;
					while (tx < 8 && ty < 8)
					{
						if (GetTile(tx, ty) == Tile::Empty)
						{
							// Black Queen Moving
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(tx, ty, tile);
							SetLastMove(x, y, tx, ty);
							if (!nextState.IsBlackInCheck())
							{
								output.push_back(nextState);
							}
						}
						else
						{
							if (!IsBlack(GetTile(tx, ty)))
							{
								// Black Queen Taking
								CloneForNextMove;
								nextState.SetTile(x, y, Tile::Empty);
								nextState.SetTile(tx, ty, tile);
								SetLastMove(x, y, tx, ty);
								if (!nextState.IsBlackInCheck())
								{
									output.push_back(nextState);
								}
							}
							break;
						}
						tx++;
						ty++;
					}
					break;

				case Tile::BlackKing:
					castlesAvailableModifier = CastleMove::WhiteKingSide | CastleMove::WhiteQueenSide;
					if (y >= 1 && !IsBlack(GetTile(x, y - 1)))
					{
						// Black King Moving/Taking
						CloneForNextMove;
						nextState.SetTile(x, y, Tile::Empty);
						nextState.SetTile(x, y - 1, tile);
						nextState.blackKingX = x; nextState.blackKingY = y - 1;
						SetLastMove(x, y, x, y - 1);
						if (!nextState.IsBlackInCheck())
						{
							nextState.castlesAvailable &= castlesAvailableModifier;
							output.push_back(nextState);
						}
					}
					if (y <= 6 && !IsBlack(GetTile(x, y + 1)))
					{
						// Black King Moving/Taking
						CloneForNextMove;
						nextState.SetTile(x, y, Tile::Empty);
						nextState.SetTile(x, y + 1, tile);
						nextState.blackKingX = x; nextState.blackKingY = y + 1;
						SetLastMove(x, y, x, y + 1);
						if (!nextState.IsBlackInCheck())
						{
							nextState.castlesAvailable &= castlesAvailableModifier;
							output.push_back(nextState);
						}
					}
					if (x >= 1)
					{
						if (!IsBlack(GetTile(x - 1, y)))
						{
							// Black King Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 1, y, tile);
							nextState.blackKingX = x - 1; nextState.blackKingY = y;
							SetLastMove(x, y, x - 1, y);
							if (!nextState.IsBlackInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						if (y >= 1 && !IsBlack(GetTile(x - 1, y - 1)))
						{
							// Black King Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 1, y - 1, tile);
							nextState.blackKingX = x - 1; nextState.blackKingY = y - 1;
							SetLastMove(x, y, x - 1, y - 1);
							if (!nextState.IsBlackInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						if (y <= 6 && !IsBlack(GetTile(x - 1, y + 1)))
						{
							// Black King Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x - 1, y + 1, tile);
							nextState.blackKingX = x - 1; nextState.blackKingY = y + 1;
							SetLastMove(x, y, x - 1, y + 1);
							if (!nextState.IsBlackInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
					}
					if (x <= 6)
					{
						if (!IsBlack(GetTile(x + 1, y)))
						{
							// Black King Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 1, y, tile);
							nextState.blackKingX = x + 1; nextState.blackKingY = y;
							SetLastMove(x, y, x + 1, y);
							if (!nextState.IsBlackInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						if (y >= 1 && !IsBlack(GetTile(x + 1, y - 1)))
						{
							// Black King Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 1, y - 1, tile);
							nextState.blackKingX = x + 1; nextState.blackKingY = y - 1;
							SetLastMove(x, y, x + 1, y - 1);
							if (!nextState.IsBlackInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
						if (y <= 6 && !IsBlack(GetTile(x + 1, y + 1)))
						{
							// Black King Moving/Taking
							CloneForNextMove;
							nextState.SetTile(x, y, Tile::Empty);
							nextState.SetTile(x + 1, y + 1, tile);
							nextState.blackKingX = x + 1; nextState.blackKingY = y + 1;
							SetLastMove(x, y, x + 1, y + 1);
							if (!nextState.IsBlackInCheck())
							{
								nextState.castlesAvailable &= castlesAvailableModifier;
								output.push_back(nextState);
							}
						}
					}
					break;
				}
			}
		}
		if (!IsBlackInCheck())
		{
			if ((castlesAvailable & CastleMove::BlackKingSide) != CastleMove::None && GetTile(5, 7) == Tile::Empty && GetTile(6, 7) == Tile::Empty)
			{
				CloneForNextMove;
				nextState.SetTile(4, 7, Tile::Empty);
				nextState.SetTile(7, 7, Tile::Empty);
				nextState.SetTile(5, 7, Tile::BlackKing);
				nextState.blackKingX = 5; nextState.blackKingY = 7;
				if (!nextState.IsBlackInCheck())
				{
					nextState.SetTile(5, 7, Tile::BlackRook);
					nextState.SetTile(6, 7, Tile::BlackKing);
					nextState.blackKingX = 6;
					SetLastMove(4, 7, 6, 7);
					if (!nextState.IsBlackInCheck())
					{
						nextState.castlesAvailable &= CastleMove::WhiteKingSide | CastleMove::WhiteQueenSide;
						output.push_back(nextState);
					}
				}
			}
			if ((castlesAvailable & CastleMove::BlackQueenSide) != CastleMove::None && GetTile(1, 7) == Tile::Empty && GetTile(2, 7) == Tile::Empty && GetTile(3, 7) == Tile::Empty)
			{
				CloneForNextMove;
				nextState.SetTile(0, 7, Tile::Empty);
				nextState.SetTile(4, 7, Tile::Empty);
				nextState.SetTile(3, 7, Tile::BlackKing);
				nextState.blackKingX = 3; nextState.blackKingY = 7;
				if (!nextState.IsBlackInCheck())
				{
					nextState.SetTile(2, 7, Tile::BlackKing);
					nextState.SetTile(3, 7, Tile::BlackRook);
					nextState.blackKingX = 2;
					SetLastMove(4, 7, 2, 7);
					if (!nextState.IsBlackInCheck())
					{
						nextState.castlesAvailable &= CastleMove::WhiteKingSide | CastleMove::WhiteQueenSide;
						output.push_back(nextState);
					}
				}
			}
		}
	}
}

GameStateSignature GameState::UniqueSignature()
{
	if (computedSignature)
	{
		return signature;
	}

	computedSignature = true;
	uint64_t emptyMask;
	int_fast32_t pieceCount = 0;
	memset(signature.data, 0, sizeof(signature.data));
	
	for (int_fast32_t tileIndex = 0; tileIndex < 64; tileIndex++)
	{
		if (board[tileIndex] != Tile::Empty)
		{
			uint8_t pieceValue;
			switch (board[tileIndex])
			{
			case Tile::Empty:       pieceValue = 0; break;
			case Tile::WhitePawn:   pieceValue = 1; break;
			case Tile::WhiteRook:   pieceValue = 2; break;
			case Tile::WhiteKnight: pieceValue = 3; break;
			case Tile::WhiteBishop: pieceValue = 4; break;
			case Tile::WhiteQueen:  pieceValue = 5; break;
			case Tile::WhiteKing:   pieceValue = 6; break;
			case Tile::BlackPawn:   pieceValue = 7; break;
			case Tile::BlackRook:   pieceValue = 8; break;
			case Tile::BlackKnight: pieceValue = 9; break;
			case Tile::BlackBishop: pieceValue = 10; break;
			case Tile::BlackQueen:  pieceValue = 11; break;
			case Tile::BlackKing:   pieceValue = 12; break;
			}

			signature.data[pieceCount / 2] |= pieceValue << ((pieceCount % 2) * 4);
			emptyMask |= 1;
			pieceCount += 1;
		}
		emptyMask <<= 1;
	}
	// pieceCount is at greatest 32, at most up to index 15 has been filled
	*reinterpret_cast<uint64_t *>(&signature.data[24]) = emptyMask;
	signature.data[23] = static_cast<uint8_t>(enPassantPawn);
	signature.data[22] = static_cast<uint8_t>(castlesAvailable);
	return signature;
}

std::ostream& operator <<(std::ostream& os, GameState const & gameState)
{
	for (int_fast32_t y = 7; y >= 0; y--)
	{
		for (int_fast32_t x = 0; x < 8; x++)
		{
#ifdef PRINT_LAST_MOVE
			if ((x == gameState.lastMoveStartX && y == gameState.lastMoveStartY) || (x == gameState.lastMoveEndX && y == gameState.lastMoveEndY))
			{
				os << "\033[47;30m";
				// os << "\033[1m";
			}
#endif
			switch (gameState.GetTile(x, y))
			{
				case Tile::Empty:       os << "-- "; break;
				case Tile::WhitePawn:   os << "WP "; break;
				case Tile::WhiteRook:   os << "WR "; break;
				case Tile::WhiteKnight: os << "WN "; break;
				case Tile::WhiteBishop: os << "WB "; break;
				case Tile::WhiteQueen:  os << "WQ "; break;
				case Tile::WhiteKing:   os << "WK "; break;
				case Tile::BlackPawn:   os << "BP "; break;
				case Tile::BlackRook:   os << "BR "; break;
				case Tile::BlackKnight: os << "BN "; break;
				case Tile::BlackBishop: os << "BB "; break;
				case Tile::BlackQueen:  os << "BQ "; break;
				case Tile::BlackKing:   os << "BK "; break;
			}

#ifdef PRINT_LAST_MOVE
			if ((x == gameState.lastMoveStartX && y == gameState.lastMoveStartY) || (x == gameState.lastMoveEndX && y == gameState.lastMoveEndY))
			{
				os << "\033[0m";
			}
#endif
		}
		os << "\n";
	}
	os << "Castles Available: ";
	if ((gameState.castlesAvailable & CastleMove::WhiteKingSide) != CastleMove::None)
	{
		os << "WK ";
	}
	if ((gameState.castlesAvailable & CastleMove::WhiteQueenSide) != CastleMove::None)
	{
		os << "WQ ";
	}
	if ((gameState.castlesAvailable & CastleMove::BlackKingSide) != CastleMove::None)
	{
		os << "BK ";
	}
	if ((gameState.castlesAvailable & CastleMove::BlackQueenSide) != CastleMove::None)
	{
		os << "BQ ";
	}
	os << "\n";
	os << "En Passant File: " << static_cast<char>(gameState.enPassantPawn == 9 ? '~' : ('a' + gameState.enPassantPawn)) << "\n";
	return os;
}
