#pragma once

#include <string>
#include <filesystem>
#include <vector>
#include <random>
#include <vector>
#include <set>
#include <array>
#include "cons-player.h"
#include "cons-player-manager.h"

class Trainer
{
public:
	static constexpr char * saveLocation = "E:/top/development/cons-chess-players";
	std::mt19937 randomEngine;
	ConsPlayer* generation[ConsPlayerConstants::generationSize];
	int_fast32_t generationNumber;
	ConsPlayerManager playerManager;

	Trainer();
	~Trainer();

	void Train();

private:
	void InitializeTraining();
	// Returns 0 if no generations are saved
	int_fast32_t FindLastSavedGeneration();
	
	void GenerateGenerationZero();
	void LoadGeneration();
	void SaveGeneration();
	void DeleteGeneration();
	std::filesystem::path::string_type PlayerFilename(int_fast32_t generationNumber, int_fast32_t playerIndex);

	void IterateGeneration();
	void PlayGeneration(std::array<ConsPlayer const *, ConsPlayerConstants::generationSize> & winners);
	void GenerateNextGeneration(std::array<ConsPlayer const *, ConsPlayerConstants::generationSize> & winners);
	ConsPlayer const * FindWinner(std::array<std::pair<float, ConsPlayer const *>, ConsPlayerConstants::generationTournamentSize> & scores);
	// tournaments is (ConsPlayer *)[generationSize * tournamentSize]
	void BuildTournaments(ConsPlayer ** tournaments);
	void MutatePlayer(ConsPlayer * player, int_fast32_t playerIndex);
};
