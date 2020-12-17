#pragma once

#include <cstdint>
#include <string>
#include <filesystem>
#include <random>
#include "cons-player-constants.h"

class ConsPlayer
{
public:
	float weights[ConsPlayerConstants::WeightCount()];
	float biases[ConsPlayerConstants::BiasCount()];
	
	ConsPlayer();
	ConsPlayer(int_fast32_t seed);
	void Save(std::filesystem::path filename);
	static void Load(std::filesystem::path filename, ConsPlayer * player);
	ConsPlayer * Clone() const;
};
