#include "pgn-loader.h"

using namespace std::string_literals;

PgnLoader::PgnLoader(std::string filename)
{
	file = new std::ifstream(filename);
}

PgnLoader::~PgnLoader()
{
	file->close();
	delete file;
}

bool PgnLoader::IsFinished()
{
	return file->eof();
}

bool PgnLoader::Next(GameState * output)
{
	ConsumeWhitespaceAndComments();
	if (file->eof())
	{
		return false;
	}

	// Process metadata
	while (file->peek() == '[')
	{
		//while (file->get() != ']');
		while (file->get() != ']');
		ConsumeWhitespaceAndComments();
	}
	if (file->eof())
	{
		return false;
	}

	int moveNumber = 0;
	GameState gameState;

	if (file->peek() == '*')
	{
		file->ignore();
		*output = gameState;
		return true;
	}

	if (!std::isdigit(file->peek()))
	{
		throw std::runtime_error("Invalid PGN; Expected at least one move"s);
	}
	// Process moves
	while (std::isdigit(file->peek()))
	{
		// Consume move number
		std::string moveNumberString = "";
		while (std::isdigit(file->peek()))
		{
			moveNumberString += file->get();
		}
		if (file->eof())
		{
			throw std::runtime_error("Invalid PGN; Unexpected eof after move number \""s + moveNumberString + "\""s);
		}
		if (file->peek() == '-' || file->peek() == '/')
		{
			ConsumeGameResult();
			break;
		}
		if (file->peek() != '.')
		{
			throw std::runtime_error("Invalid PGN; Expected number \""s + moveNumberString + "\" to be followed by '.' instead of '"s + static_cast<char>(file->peek()) + "'"s);
		}
		file->ignore();
		int nextMoveNumber = std::stoi(moveNumberString);
		if (nextMoveNumber != moveNumber + 1)
		{
			throw std::runtime_error("Invalid PGN; Expected move number \""s + moveNumberString + ".\" to be one plus the last move number \""s + std::to_string(moveNumber) +"\""s);
		}
		moveNumber = nextMoveNumber;

		ConsumeWhitespaceAndComments();
		// Some files have '*' at the end of games with result '*', even though many do not
		if (file->peek() == '*')
		{
			file->ignore();
			break;
		}
		if (std::isdigit(file->peek()))
		{
			ConsumeGameResult();
			break;
		}

		// White move
		std::string whiteMove = "";
		while (!std::isspace(file->peek()))
		{
			whiteMove += file->get();
		}
		gameState = MakePgnMove(gameState, whiteMove);

		ConsumeWhitespaceAndComments();
		
		// Optionally consume continuation move number
		if (std::isdigit(file->peek()))
		{
			moveNumberString = "";
			while (std::isdigit(file->peek()))
			{
				moveNumberString += file->get();
			}
			if (file->eof())
			{
				throw std::runtime_error("Invalid PGN; Unexpected eof after move number \""s + moveNumberString + "\""s);
			}
			if (file->peek() == '-' || file->peek() == '/')
			{
				ConsumeGameResult();
				break;
			}
			if (file->peek() != '.')
			{
				throw std::runtime_error("Invalid PGN; Expected number \""s + moveNumberString + "\" to be followed by '...' instead of '"s + static_cast<char>(file->peek()) + "'"s);
			}
			file->ignore();
			if (file->peek() != '.')
			{
				throw std::runtime_error("Invalid PGN; Expected number \""s + moveNumberString + "\" to be followed by '...' instead of '."s + static_cast<char>(file->peek()) + "'"s);
			}
			file->ignore();
			if (file->peek() != '.')
			{
				throw std::runtime_error("Invalid PGN; Expected number \""s + moveNumberString + "\" to be followed by '...' instead of '.."s + static_cast<char>(file->peek()) + "'"s);
			}
			file->ignore();

			int nextMoveNumber = std::stoi(moveNumberString);
			if (nextMoveNumber != moveNumber)
			{
				throw std::runtime_error("Invalid PGN; Expected move number \""s + moveNumberString + "...\" to be equal to the last move number \""s + std::to_string(moveNumber) + "\""s);
			}
			moveNumber = nextMoveNumber;
		}
		
		ConsumeWhitespaceAndComments();
		// Some files have '*' at the end of games with result '*', even though many do not
		if (file->peek() == '*')
		{
			file->ignore();
			break;
		}
		if (std::isdigit(file->peek()))
		{
			ConsumeGameResult();
			break;
		}

		// Black move
		std::string blackMove = "";
		while (!std::isspace(file->peek()))
		{
			blackMove += file->get();
		}
		gameState = MakePgnMove(gameState, blackMove);

		ConsumeWhitespaceAndComments();
	}

	*output = gameState;
	return true;
}

GameState PgnLoader::MakePgnMove(GameState& gameState, std::string pgnMove)
{
	std::vector<GameState> nextGameStates;
	gameState.NextGameStates(nextGameStates);

	if (pgnMove.rfind("O-O-O", 0) == 0) // starts with O-O
	{
		// Queenside castle
		if (
			(gameState.move == Tile::White && (gameState.castlesAvailable & CastleMove::WhiteQueenSide) != CastleMove::None) ||
			(gameState.move == Tile::Black && (gameState.castlesAvailable & CastleMove::BlackQueenSide) != CastleMove::None)
			)
		{
			for (GameState& nextGameState : nextGameStates)
			{
				if (gameState.move == Tile::White && nextGameState.whiteKingX == 2 || gameState.move == Tile::Black && nextGameState.blackKingX == 2)
				{
					return nextGameState;
				}
			}
		}
		throw std::runtime_error("Invalid PGN move \""s + pgnMove + "\"; Cannot queen side castle in current game state"s);
	}
	else if (pgnMove.rfind("O-O", 0) == 0) // starts with O-O
	{
		// Kingside castle
		if (
			(gameState.move == Tile::White && (gameState.castlesAvailable & CastleMove::WhiteKingSide) != CastleMove::None) ||
			(gameState.move == Tile::Black && (gameState.castlesAvailable & CastleMove::BlackKingSide) != CastleMove::None)
		)
		{
			for (GameState& nextGameState : nextGameStates)
			{
				if (gameState.move == Tile::White && nextGameState.whiteKingX == 6 || gameState.move == Tile::Black && nextGameState.blackKingX == 6)
				{
					return nextGameState;
				}
			}
		}
		throw std::runtime_error("Invalid PGN move \""s + pgnMove + "\"; Cannot king side castle in current game state"s);
	}
	
	// We match any trailing string, perhaps we can be more selective in the future with '?', '!', ' e.p.', etc.
	static std::regex const pgnMoveRegex("([RNBQKP]?)([a-h]?)([1-8]?)(x?)([a-h])([1-8])(=([RNBQ]))?([+|#]?).*");
	std::smatch matchResult;
	std::regex_match(pgnMove, matchResult, pgnMoveRegex);

	std::string pieceMatch = matchResult[1].str();
	std::string sourceFileMatch = matchResult[2].str();
	std::string sourceRankMatch = matchResult[3].str();
	std::string captureMatch = matchResult[4].str();
	std::string targetFileMatch = matchResult[5].str();
	std::string targetRankMatch = matchResult[6].str();
	std::string promotionMatch = matchResult[8].str();;
	std::string checkMatch = matchResult[9].str();

	Tile sourcePiece;
	if (pieceMatch == "" || pieceMatch == "P")
	{
		sourcePiece = gameState.move | Tile::Pawn;
	}
	else if (pieceMatch == "R")
	{
		sourcePiece = gameState.move | Tile::Rook;
	}
	else if (pieceMatch == "N")
	{
		sourcePiece = gameState.move | Tile::Knight;
	}
	else if (pieceMatch == "B")
	{
		sourcePiece = gameState.move | Tile::Bishop;
	}
	else if (pieceMatch == "Q")
	{
		sourcePiece = gameState.move | Tile::Queen;
	}
	else if (pieceMatch == "K")
	{
		sourcePiece = gameState.move | Tile::King;
	}

	int sourceFile = sourceFileMatch.length() == 0 ? -1 : sourceFileMatch[0] - 'a';
	int sourceRank = sourceRankMatch.length() == 0 ? -1 : sourceRankMatch[0] - '1';

	bool mustCapture = captureMatch.length() != 0;

	int targetFile = targetFileMatch[0] - 'a';
	int targetRank = targetRankMatch[0] - '1';

	Tile promotionPiece = Tile::Empty;
	if (promotionMatch == "R")
	{
		promotionPiece = gameState.move | Tile::Rook;
	}
	else if (promotionMatch == "N")
	{
		promotionPiece = gameState.move | Tile::Knight;
	}
	else if (promotionMatch == "B")
	{
		promotionPiece = gameState.move | Tile::Bishop;
	}
	else if (promotionMatch == "Q")
	{
		promotionPiece = gameState.move | Tile::Queen;
	}

	bool check = checkMatch == "+";
	bool checkmatch = checkMatch == "#";

	Tile targetPiece = promotionPiece == Tile::Empty ? sourcePiece : promotionPiece;

	// It actually is necessary to check the whole board for the source piece when no disambiguation is provided
	// Technically we could greatly decrease this check with more fine tuned logic but that seems unnecessary for this function
	int minSourceX = sourceFile == -1 ? 0 : sourceFile;
	int maxSourceX = sourceFile == -1 ? 8 : sourceFile + 1;
	int minSourceY = sourceRank == -1 ? 0 : sourceRank;
	int maxSourceY = sourceRank == -1 ? 8 : sourceRank + 1;

	GameState* result = nullptr;
	int foundSourceX, foundSourceY;

	for (GameState& nextGameState : nextGameStates)
	{
		if (gameState.GetTile(targetFile, targetRank) != targetPiece && nextGameState.GetTile(targetFile, targetRank) == targetPiece)
		{
			// If we are not moving the king the king can't move; this is needed to explicitly disambiguate castling and similar rook moves
			if ((targetPiece & Tile::King) == Tile::Empty && (gameState.blackKingX != nextGameState.blackKingX || gameState.whiteKingX != nextGameState.whiteKingX))
			{
				continue;
			}
			for (int_fast32_t x = minSourceX; x < maxSourceX; x++)
			{
				for (int_fast32_t y = minSourceY; y < maxSourceY; y++)
				{
					if (gameState.GetTile(x, y) == sourcePiece && nextGameState.GetTile(x, y) == Tile::Empty)
					{
						// We want to detect ambiguious moves
						if (result != nullptr)
						{
							throw std::runtime_error("Invalid PGN move \""s + pgnMove + "\"; Move is ambiguous, could originate from either "s + static_cast<char>('a' + foundSourceX) + static_cast<char>('1' + foundSourceY) + " or "s + static_cast<char>('a' + x) + static_cast<char>('1' + y));
						}
						result = &nextGameState;
						foundSourceX = x;
						foundSourceY = y;
					}
				}
			}
		}
	}

	if (result == nullptr)
	{
		throw std::runtime_error("Invalid PGN move \""s + pgnMove + "\"; Move is not possible in the current state"s);
	}
	return *result;
}

void PgnLoader::ConsumeWhitespaceAndComments()
{
	while (!file->eof())
	{
		char character = file->peek();
		if (std::isspace(character))
		{
			file->ignore();
		}
		else if (character == ';')
		{
			while (file->get() != '\n');
		}
		else if (character == '{')
		{
			while (file->get() != '}');
		}
		else
		{
			break;
		}
	}
}

void PgnLoader::ConsumeGameResult()
{
	while (std::isdigit(file->peek()) || file->peek() == '-' || file->peek() == '/')
	{
		file->ignore();
	}
}
