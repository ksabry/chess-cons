#pragma once
#include <filesystem>
#include "cons-player.h"
#include "cons-player-manager.h"

class Tester
{
public:
	static constexpr char * saveLocation = "E:/top/development/cons-chess-players";
	ConsPlayerManager playerManager;

	Tester();
	~Tester();

	void RoundRobin(int_fast32_t whiteGeneration, int_fast32_t blackGeneration);
	float PlayGame(int_fast32_t whiteGeneration, int_fast32_t blackGeneration, int_fast32_t whitePlayerIndex, int_fast32_t blackPlayerIndex);

private:
	ConsPlayer * whitePlayer;
	ConsPlayer * blackPlayer;

	std::filesystem::path PlayerFilename(int_fast32_t generationNumber, int_fast32_t playerIndex);
};
