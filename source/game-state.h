#pragma once

#include <cstdint>
#include <vector>
#include <iostream>

enum class Tile : uint_fast32_t
{
	Empty = 0,

	Pawn = 1 << 0,
	Rook = 1 << 1,
	Knight = 1 << 2,
	Bishop = 1 << 3,
	Queen = 1 << 4,
	King = 1 << 6,

	Piece = Pawn | Rook | Knight | Bishop | Queen | King,

	White = 1 << 7,
	Black = 1 << 8,

	Color = White | Black,

	WhitePawn = White | Pawn,
	WhiteRook = White | Rook,
	WhiteKnight = White | Knight,
	WhiteBishop = White | Bishop,
	WhiteQueen = White | Queen,
	WhiteKing = White | King,

	BlackPawn = Black | Pawn,
	BlackRook = Black | Rook,
	BlackKnight = Black | Knight,
	BlackBishop = Black | Bishop,
	BlackQueen = Black | Queen,
	BlackKing = Black | King,
};

inline Tile operator &(Tile lhs, Tile rhs)
{
	return static_cast<Tile>(static_cast<uint_fast32_t>(lhs) & static_cast<uint_fast32_t>(rhs));
}

inline Tile operator |(Tile lhs, Tile rhs)
{
	return static_cast<Tile>(static_cast<uint_fast32_t>(lhs) | static_cast<uint_fast32_t>(rhs));
}

inline Tile operator ^(Tile lhs, Tile rhs)
{
	return static_cast<Tile>(static_cast<uint_fast32_t>(lhs) ^ static_cast<uint_fast32_t>(rhs));
}

inline Tile operator ~(Tile value)
{
	return static_cast<Tile>(~static_cast<uint_fast32_t>(value));
}

inline Tile& operator &=(Tile& lhs, Tile rhs)
{
	lhs = lhs & rhs;
	return lhs;
}

inline Tile& operator |=(Tile& lhs, Tile rhs)
{
	lhs = lhs | rhs;
	return lhs;
}

inline Tile& operator ^=(Tile& lhs, Tile rhs)
{
	lhs = lhs ^ rhs;
	return lhs;
}

inline bool IsWhite(Tile tile)
{
	return (tile & Tile::White) != Tile::Empty;
}

inline bool IsBlack(Tile tile)
{
	return (tile & Tile::Black) != Tile::Empty;
}


enum class CastleMove : uint_fast32_t
{
	None = 0,

	WhiteKingSide = 1 << 0,
	WhiteQueenSide = 1 << 1,
	BlackKingSide = 1 << 2,
	BlackQueenSide = 1 << 3,
};

inline CastleMove operator &(CastleMove lhs, CastleMove rhs)
{
	return static_cast<CastleMove>(static_cast<uint_fast32_t>(lhs) & static_cast<uint_fast32_t>(rhs));
}

inline CastleMove operator |(CastleMove lhs, CastleMove rhs)
{
	return static_cast<CastleMove>(static_cast<uint_fast32_t>(lhs) | static_cast<uint_fast32_t>(rhs));
}

inline CastleMove operator ^(CastleMove lhs, CastleMove rhs)
{
	return static_cast<CastleMove>(static_cast<uint_fast32_t>(lhs) ^ static_cast<uint_fast32_t>(rhs));
}

inline CastleMove operator ~(CastleMove value)
{
	return static_cast<CastleMove>(~static_cast<uint_fast32_t>(value));
}

inline CastleMove& operator &=(CastleMove& lhs, CastleMove rhs)
{
	lhs = lhs & rhs;
	return lhs;
}

inline CastleMove& operator |=(CastleMove& lhs, CastleMove rhs)
{
	lhs = lhs | rhs;
	return lhs;
}

inline CastleMove& operator ^=(CastleMove& lhs, CastleMove rhs)
{
	lhs = lhs ^ rhs;
	return lhs;
}

struct GameStateSignature
{
	uint8_t data[32];
};

inline bool operator ==(GameStateSignature const & lhs, GameStateSignature const & rhs)
{
	// compiler should very easily optimize this
	for (int_fast32_t index = 0; index < 32; index++)
	{
		if (lhs.data[index] != rhs.data[index])
		{
			return false;
		}
	}
	return true;
}

inline bool operator !=(GameStateSignature const & lhs, GameStateSignature const & rhs)
{
	// compiler should very easily optimize this
	for (int_fast32_t index = 0; index < 32; index++)
	{
		if (lhs.data[index] == rhs.data[index])
		{
			return false;
		}
	}
	return true;
}

// This is the same as boost's hash_combine implementation
template <class T>
inline void HashCombine(std::size_t& seed, const T& v)
{
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

namespace std
{
	template<>
	struct hash<GameStateSignature>
	{
		std::size_t operator()(GameStateSignature const & signature) const noexcept
		{
			static_assert(32 % sizeof(std::size_t) == 0);
			std::size_t result = 0;
			int_fast32_t offset = 0;
			while (offset < 32)
			{
				HashCombine(result, reinterpret_cast<std::size_t>(signature.data + offset));
				offset += sizeof(std::size_t);
			}
			return result;
		}
	};
}

// #define PRINT_LAST_MOVE

#define CloneForNextMove \
nextState = *this;\
nextState.move ^= Tile::Color;\
nextState.enPassantPawn = 9;

#ifdef PRINT_LAST_MOVE
#define SetLastMove(sx,sy,ex,ey) \
nextState.lastMoveStartX=sx;\
nextState.lastMoveStartY=sy;\
nextState.lastMoveEndX=ex;\
nextState.lastMoveEndY=ey;
#else
#define SetLastMove(sx,sy,ex,ey) do{}while(0)
#endif

// Note: For the purposes of completely unqiue hashing, a very simple method can do this in 31 bytes (241 bits) (11 states per square, king positions, en passant file, castles)
// the class itself is currently 284 bytes
// if we are to store this for caching results, we should try to keep the hashing function simple but with awareness that most pieces do not appear frequently on the board;
// most spaces will be empty, followed by pawns and so forth

class GameState
{
public:
	Tile board[64];
	CastleMove castlesAvailable;
	int_fast32_t enPassantPawn;
	Tile move; // Either Tile::White or Tile::Black

	int_fast32_t whiteKingX;
	int_fast32_t whiteKingY;
	int_fast32_t blackKingX;
	int_fast32_t blackKingY;

#ifdef PRINT_LAST_MOVE
	int_fast32_t lastMoveStartX;
	int_fast32_t lastMoveStartY;
	int_fast32_t lastMoveEndX;
	int_fast32_t lastMoveEndY;
#endif

private:
	bool computedSignature = false;
	GameStateSignature signature;

public:
	GameState();
	GameState(
		Tile* board,
		Tile move,
		CastleMove castlesAvailable,
		int_fast32_t enPassantPawn,
		int_fast32_t whiteKingX,
		int_fast32_t whiteKingY,
		int_fast32_t blackKingX,
		int_fast32_t blackKingY
	);

	inline Tile GetTile(int_fast32_t x, int_fast32_t y) const
	{
		return static_cast<Tile>(board[x * 8 + y]);
	}
	
	inline void SetTile(int_fast32_t x, int_fast32_t y, Tile tile)
	{
		board[x * 8 + y] = tile;
	}

	// Returns in centipawns
	int_fast32_t WhitePieceScore() const;
	// Returns in centipawns
	int_fast32_t BlackPieceScore() const;

	bool IsWhiteInCheck() const;
	bool IsBlackInCheck() const;

	void NextGameStates(std::vector<GameState>& output) const;

	GameStateSignature UniqueSignature();
};

std::ostream& operator <<(std::ostream& os, GameState const & gameState);
