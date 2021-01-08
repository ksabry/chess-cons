#include <iostream>
#include <filesystem>
#include <random>
#include <string>
#include <regex>
#include <map>
#include <array>

#include "trainer.h"
#include "tournament-set.h"

using namespace std::string_literals;

constexpr uint_fast32_t INITIAL_GENERATION_SEED_OFFSET = 100000;
constexpr uint_fast32_t ITERATE_GENERATION_SEED_OFFSET = 200000;

Trainer::Trainer()
{
	for (int_fast32_t playerIndex = 0; playerIndex < ConsPlayerConstants::generationSize; playerIndex++)
	{
		generation[playerIndex] = nullptr;
	}
}

Trainer::~Trainer()
{
	DeleteGeneration();
}

void Trainer::Train()
{
	InitializeTraining();
	while (generationNumber < 10000)
	{
		IterateGeneration();
	}
}

void Trainer::InitializeTraining()
{
	generationNumber = FindLastSavedGeneration();
	if (generationNumber == 0)
	{
		GenerateGenerationZero();
	}
	else
	{
		LoadGeneration();
	}
}

int_fast32_t Trainer::FindLastSavedGeneration()
{
	static std::regex playerFilenameRegex("player_(\\d+)_(\\d+)_(\\d+)_(\\d+)"s);
	
	std::map<int_fast32_t, std::vector<bool>, std::greater<int_fast32_t>> savedGenerations;
	for (const auto & playerFile : std::filesystem::directory_iterator(saveLocation))
	{
		std::string playerFilename = playerFile.path().filename().string();
		std::smatch match;
		bool matchSuccess = std::regex_match(playerFilename, match, playerFilenameRegex);
		if (!matchSuccess)
		{
			std::cout << "Warning: Invalid filename encountered in saved player directory \""s + playerFilename + "\""s << std::endl;
			continue;
		}
		std::string versionString = match[1];
		std::string seedString = match[2];
		std::string generationNumberString = match[3];
		std::string playerIndexString = match[4];
		int version = std::stoi(versionString, nullptr, 10);
		int seed = std::stoi(seedString, nullptr, 10);
		int generationNumber = std::stoi(generationNumberString, nullptr, 10);
		int playerIndex = std::stoi(playerIndexString, nullptr, 10);

		if (version != ConsPlayerConstants::version || seed != ConsPlayerConstants::seed)
		{
			continue;
		}

		if (savedGenerations.count(generationNumber) == 0)
		{
			std::vector<bool> foundPlayers;
			foundPlayers.resize(ConsPlayerConstants::generationSize);
			savedGenerations[generationNumber] = foundPlayers;
		}
		if (playerIndex >= ConsPlayerConstants::generationSize)
		{
			std::cout << "Warning: Ignoring saved player with higher index than generation count \""s + playerFilename + "\""s << std::endl;
		}
		savedGenerations[generationNumber][playerIndex] = true;
	}

	for (const auto & savedGeneration: savedGenerations)
	{
		bool allPlayersFound = true;
		for (bool playerFound : savedGeneration.second)
		{
			if (!playerFound)
			{
				allPlayersFound = false;
				break;
			}
		}
		if (allPlayersFound)
		{
			return savedGeneration.first;
		}
		else
		{
			std::cout << "Warning: Found some but not all of generation "s << savedGeneration.first << "; loading previous generation" << std::endl;
		}
	}
	return 0;
}

void Trainer::GenerateGenerationZero()
{
	DeleteGeneration();
	for (int_fast32_t playerIndex = 0; playerIndex < ConsPlayerConstants::generationSize; playerIndex++)
	{
		std::cout << "Generation " << generationNumber << " random initialization: " << playerIndex << " / " << ConsPlayerConstants::generationSize << "\r";
		generation[playerIndex] = new ConsPlayer(ConsPlayerConstants::seed + INITIAL_GENERATION_SEED_OFFSET + playerIndex);
	}
	std::cout << "Generation " << generationNumber << " random initialization complete            " << std::endl;
}

void Trainer::LoadGeneration()
{
	DeleteGeneration();
	for (int playerIndex = 0; playerIndex < ConsPlayerConstants::generationSize; playerIndex++)
	{
		std::cout << "Generation " << generationNumber << " loading: " << playerIndex << " / " << ConsPlayerConstants::generationSize << "\r";
		generation[playerIndex] = new ConsPlayer();
		ConsPlayer::Load(PlayerFilename(generationNumber, playerIndex), generation[playerIndex]);
	}
	std::cout << "Generation " << generationNumber << " loading complete            " << std::endl;
}

void Trainer::SaveGeneration()
{
	for (int playerIndex = 0; playerIndex < ConsPlayerConstants::generationSize; playerIndex++)
	{
		std::cout << "Generation " << generationNumber << " saving: " << playerIndex << " / " << ConsPlayerConstants::generationSize << "\r";
		generation[playerIndex]->Save(PlayerFilename(generationNumber, playerIndex));
	}
	std::cout << "Generation " << generationNumber << " saving complete            " << std::endl;
}

std::filesystem::path::string_type Trainer::PlayerFilename(int_fast32_t generationNumber, int_fast32_t playerIndex)
{
	// <format> is in the C++20 spec but as of now is not actually supported by any compilers
	// auto filename = std::filesystem::path(saveLocation) / std::format("player_{:0>5}_{:0>10}_{:0>5}_{:0>5}", ConsPlayerConstants::version, ConsPlayerConstants::seed, generationNumber, playerIndex);
	std::string versionString = std::to_string(ConsPlayerConstants::version);
	std::string seedString = std::to_string(ConsPlayerConstants::seed);
	std::string generationNumberString = std::to_string(generationNumber);
	std::string playerIndexString = std::to_string(playerIndex);
	versionString = std::string(5 - versionString.length(), '0') + versionString;
	seedString = std::string(10 - seedString.length(), '0') + seedString;
	generationNumberString = std::string(5 - generationNumberString.length(), '0') + generationNumberString;
	playerIndexString = std::string(5 - playerIndexString.length(), '0') + playerIndexString;
	std::filesystem::path filename = std::filesystem::path(saveLocation) / ("player_"s + versionString + "_"s + seedString + "_"s + generationNumberString + "_"s + playerIndexString);
	return filename.native();
}

void Trainer::DeleteGeneration()
{
	for (int_fast32_t playerIndex = 0; playerIndex < ConsPlayerConstants::generationSize; playerIndex++)
	{
		if (generation[playerIndex] != nullptr)
		{
			delete generation[playerIndex];
			generation[playerIndex] = nullptr;
		}
	}
}

void Trainer::IterateGeneration()
{
	randomEngine.seed(ConsPlayerConstants::seed + ITERATE_GENERATION_SEED_OFFSET + generationNumber);
	std::array<ConsPlayer const *, ConsPlayerConstants::tournamentCount> winners;
	PlayGeneration(winners);
	GenerateNextGeneration(winners);
	SaveGeneration();
}

void Trainer::PlayGeneration(std::array<ConsPlayer const *, ConsPlayerConstants::tournamentCount> & winners)
{
	int_fast32_t gamesPlayed = 0;
	int_fast32_t cacheHits = 0;
	int_fast32_t totalGameCount = ConsPlayerConstants::tournamentCount * ConsPlayerConstants::generationTournamentSize * (ConsPlayerConstants::generationTournamentSize - 1);

	std::uniform_int_distribution<int_fast32_t> maxEvaluationCountDistribution(ConsPlayerConstants::maxMoveEvaluationCountLow, ConsPlayerConstants::maxMoveEvaluationCountHigh);
	// TODO: a conservative strategy to try if we have diffuculty getting results would be to halve the tournament count and have each winner be copied twice to the next generation;
	//       once mutated and once unchanged
	TournamentSet tournamentSet(randomEngine, generation);
	
	// std::cout << tournamentSet;
	// std::cin.ignore();

	std::array<EvaluationCache, ConsPlayerConstants::generationSize> playerEvaluationCaches;
	
	float cachedResults[ConsPlayerConstants::generationSize][ConsPlayerConstants::generationSize];
	for (int whitePlayerIndex = 0; whitePlayerIndex < ConsPlayerConstants::generationSize; whitePlayerIndex++)
	{
		for (int blackPlayerIndex = 0; blackPlayerIndex < ConsPlayerConstants::generationSize; blackPlayerIndex++)
		{
			cachedResults[whitePlayerIndex][blackPlayerIndex] = NAN;
		}
	}

	for (int_fast32_t tournamentIndex = 0; tournamentIndex < ConsPlayerConstants::tournamentCount; tournamentIndex++)
	{
		std::array<std::pair<float, ConsPlayer const *>, ConsPlayerConstants::generationTournamentSize> scores;
		for (int_fast32_t tournamentPlayerIndex = 0; tournamentPlayerIndex < ConsPlayerConstants::generationTournamentSize; tournamentPlayerIndex++)
		{
			scores[tournamentPlayerIndex] = { 0.0f, tournamentSet.tournaments[tournamentIndex][tournamentPlayerIndex] };
		}
		for (int_fast32_t whitePlayerIndex = 0; whitePlayerIndex < ConsPlayerConstants::generationTournamentSize; whitePlayerIndex++)
		{
			for (int_fast32_t blackPlayerIndex = 0; blackPlayerIndex < ConsPlayerConstants::generationTournamentSize; blackPlayerIndex++)
			{
				std::cout << "Generation " << generationNumber << " scoring: " << gamesPlayed << " / " << totalGameCount << " (" << cacheHits << " cache hits)\r";

				if (whitePlayerIndex == blackPlayerIndex)
				{
					continue;
				}
				int_fast32_t generationWhitePlayerIndex = tournamentSet.tournamentsByIndex[tournamentIndex][whitePlayerIndex];
				int_fast32_t generationBlackPlayerIndex = tournamentSet.tournamentsByIndex[tournamentIndex][blackPlayerIndex];
				
				float result = cachedResults[generationWhitePlayerIndex][generationBlackPlayerIndex];
				if (isnan(result))
				{
					result = playerManager.PlayGame(
						tournamentSet.tournaments[tournamentIndex][whitePlayerIndex],
						tournamentSet.tournaments[tournamentIndex][blackPlayerIndex],
						maxEvaluationCountDistribution(randomEngine) //,
						// &playerEvaluationCaches[generationWhitePlayerIndex],
						// &playerEvaluationCaches[generationBlackPlayerIndex]
					);
					cachedResults[generationWhitePlayerIndex][generationBlackPlayerIndex] = result;
				}
				else
				{
					cacheHits++;
				}
				scores[whitePlayerIndex].first += result;
				scores[blackPlayerIndex].first -= result;
				gamesPlayed++;
			}
		}
		winners[tournamentIndex] = FindWinner(scores);
	}
	std::cout << "Generation " << generationNumber << " scoring complete                      " << std::endl;
}

ConsPlayer const * Trainer::FindWinner(std::array<std::pair<float, ConsPlayer const *>, ConsPlayerConstants::generationTournamentSize> & scores)
{
	std::sort(scores.begin(), scores.end(), std::greater<std::pair<float, ConsPlayer const *>>());

	// If there are multiple best scores randomly select from them
	int_fast32_t equalScoreCount = 1;
	while (equalScoreCount < scores.size())
	{
		if (scores[equalScoreCount].first < scores[0].first)
		{
			break;
		}
		equalScoreCount++;
	}
	std::uniform_int_distribution<int_fast32_t> indexDistribution(0, equalScoreCount - 1);
	return scores[indexDistribution(randomEngine)].second;
}

void Trainer::GenerateNextGeneration(std::array<ConsPlayer const *, ConsPlayerConstants::tournamentCount> & winners)
{
	std::cout << "Tournament winners: ";
	for (int_fast32_t winnerIndex = 0; winnerIndex < ConsPlayerConstants::tournamentCount; winnerIndex++)
	{
		ConsPlayer const * winner = winners[winnerIndex];
		int_fast32_t index = -1;
		for (int_fast32_t playerIndex = 0; playerIndex < ConsPlayerConstants::generationSize; playerIndex++)
		{
			if (generation[playerIndex] == winner)
			{
				std::cout << playerIndex << " ";
				break;
			}
		}
		std::cout << std::endl;
	}

	ConsPlayer * nextGeneration[ConsPlayerConstants::generationSize];
	generationNumber++;

	for (int_fast32_t winnerIndex = 0; winnerIndex < ConsPlayerConstants::tournamentCount; winnerIndex++)
	{
		std::cout << "Generation " << generationNumber << " mutation: " << winnerIndex * 2 << " / " << ConsPlayerConstants::generationSize << "\r";
		nextGeneration[winnerIndex * 2] = winners[winnerIndex]->Clone();
		nextGeneration[winnerIndex * 2 + 1] = winners[winnerIndex]->Clone();
		MutatePlayer(nextGeneration[winnerIndex * 2 + 1]);
	}

	DeleteGeneration();
	for (int_fast32_t playerIndex = 0; playerIndex < ConsPlayerConstants::generationSize; playerIndex++)
	{
		generation[playerIndex] = nextGeneration[playerIndex];
	}
	std::cout << "Generation " << generationNumber << " mutation complete            " << std::endl;
}

void Trainer::MutatePlayer(ConsPlayer * player)
{
	std::uniform_real_distribution<float> mutationChanceDistribution(-1.0f, 1.0f);
	std::normal_distribution mutationAmountDistribution(0.0f, ConsPlayerConstants::mutationAmount);
	for (int_fast32_t weightIndex = 0; weightIndex < ConsPlayerConstants::WeightCount(); weightIndex++)
	{
		if (mutationChanceDistribution(randomEngine) < ConsPlayerConstants::mutationRate)
		{
			player->weights[weightIndex] += mutationAmountDistribution(randomEngine);
		}
	}
	for (int_fast32_t biasIndex = 0; biasIndex < ConsPlayerConstants::BiasCount(); biasIndex++)
	{
		if (mutationChanceDistribution(randomEngine) < ConsPlayerConstants::mutationRate)
		{
			player->biases[biasIndex] += mutationAmountDistribution(randomEngine);
		}
	}
}
