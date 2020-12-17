#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <string>
#include <regex>
#include "game-state.h"

class PgnLoader
{
	private:
		std::ifstream * file;
	
	public:
		PgnLoader(std::string filename);
		~PgnLoader();

		bool IsFinished();
		bool Next(GameState* output);
		GameState MakePgnMove(GameState& gameState, std::string pgnMove);

	private:
		void ConsumeWhitespaceAndComments();
		void ConsumeGameResult();
};
