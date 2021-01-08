#include <string>
#include <array>
#include <algorithm>
#include "tester.h"

using namespace std::string_literals;

Tester::Tester()
{
	whitePlayer = nullptr;
	blackPlayer = nullptr;
}

Tester::~Tester()
{
	if (whitePlayer != nullptr)
	{
		delete whitePlayer;
	}
	if (blackPlayer != nullptr)
	{
		delete blackPlayer;
	}
}

void Tester::RoundRobin(int_fast32_t whiteGeneration, int_fast32_t blackGeneration)
{
	float totalResult = 0.0f;

	std::array<int_fast32_t, ConsPlayerConstants::generationSize> whitePlayerIndices;
	std::array<int_fast32_t, ConsPlayerConstants::generationSize> blackPlayerIndices;
	for (int_fast32_t index = 0; index < ConsPlayerConstants::generationSize; index++)
	{
		whitePlayerIndices[index] = index;
		blackPlayerIndices[index] = index;
	}

	std::mt19937 randomEngine(123);
	std::shuffle(whitePlayerIndices.begin(), whitePlayerIndices.end(), randomEngine);
	std::shuffle(blackPlayerIndices.begin(), blackPlayerIndices.end(), randomEngine);

	for (int_fast32_t whitePlayerIndex : whitePlayerIndices)
	{
		for (int_fast32_t blackPlayerIndex : blackPlayerIndices)
		{
			std::cout << "Playing white (gen=" << whiteGeneration << ", player=" << whitePlayerIndex << ") vs black (gen=" << blackGeneration << ", player=" << blackPlayerIndex << ")" << std::endl;
			float result = PlayGame(whiteGeneration, blackGeneration, whitePlayerIndex, blackPlayerIndex);
			totalResult += result;

			if (result == 1)
			{
				std::cout << "White wins" << std::endl;
			}
			else if (result == -1)
			{
				std::cout << "Black wins" << std::endl;
			}
			else
			{
				std::cout << "Tie with score " << result << std::endl;
			}
			
			std::cout << "Cumulative score " << totalResult << std::endl;
			// std::cin.ignore();
		}
	}
}

float Tester::PlayGame(int_fast32_t whiteGeneration, int_fast32_t blackGeneration, int_fast32_t whitePlayerIndex, int_fast32_t blackPlayerIndex)
{
	if (whitePlayer != nullptr)
	{
		delete whitePlayer;
	}
	if (blackPlayer != nullptr)
	{
		delete blackPlayer;
	}

	whitePlayer = new ConsPlayer();
	ConsPlayer::Load(PlayerFilename(whiteGeneration, whitePlayerIndex), whitePlayer);
	blackPlayer = new ConsPlayer();
	ConsPlayer::Load(PlayerFilename(blackGeneration, blackPlayerIndex), blackPlayer);

	return playerManager.PlayGame(whitePlayer, blackPlayer, 1000);
}

std::filesystem::path Tester::PlayerFilename(int_fast32_t generationNumber, int_fast32_t playerIndex)
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
	return filename;
}
