#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include "cons-player.h"

using namespace std::string_literals;

void PrintArray(std::ostream& out, int_fast32_t const * array, int_fast32_t array_length)
{
	out << "{ ";
	for (size_t index = 0; index < array_length; index++)
	{
		out << array[index];
		if (index != array_length - 1)
		{
			out << ", ";
		}
	}
	out << " }";
}

// Doesn't initialize weights or biases
ConsPlayer::ConsPlayer()
{
}

ConsPlayer::ConsPlayer(int_fast32_t seed)
{
	std::mt19937 generator(seed);
	std::uniform_real_distribution<float> distribution(-1.0, 1.0);
	double randomRealBetweenZeroAndOne = distribution(generator);
	for (int weightIndex = 0; weightIndex < ConsPlayerConstants::WeightCount(); weightIndex++)
	{
		weights[weightIndex] = distribution(generator);
	}
	for (int biasIndex = 0; biasIndex < ConsPlayerConstants::BiasCount(); biasIndex++)
	{
		biases[biasIndex] = distribution(generator);
	}
}

void ConsPlayer::Save(std::filesystem::path filename)
{
	std::ofstream file;

	if (ConsPlayerConstants::version == 1)
	{
		// heading is `{version} {tvInputSize} {cvInputSize} {tvLayerCount} {cvLayerCount} {maxGameMoveCount} {maxMoveEvaluationCount} {tieBreakWithPieceScore ? 1 : 0} {tvLayerSizes.join(' ')} {cvLayerSizes.join(' ')} |`
		// binary stream of all weights and biases
		file.open(filename, std::ios::out | std::ios::binary);
		
		file << ConsPlayerConstants::version << " ";
		file << ConsPlayerConstants::tvInputSize << " ";
		file << ConsPlayerConstants::cvInputSize << " ";
		file << ConsPlayerConstants::tvLayerCount << " ";
		file << ConsPlayerConstants::cvLayerCount << " ";
		file << ConsPlayerConstants::maxGameMoveCount << " ";
		file << ConsPlayerConstants::maxMoveEvaluationCount << " ";
		file << (ConsPlayerConstants::tieBreakWithPieceScore ? 1 : 0) << " ";
		file << ConsPlayerConstants::generationSize << " ";
		file << ConsPlayerConstants::generationTournamentSize << " ";
		file << ConsPlayerConstants::mutationRate << " ";
		file << ConsPlayerConstants::mutationAmount << " ";

		for (int_fast32_t tvLayerSize : ConsPlayerConstants::tvLayerSizes)
		{
			file << tvLayerSize << " ";
		}
		for (int_fast32_t cvLayerSize : ConsPlayerConstants::cvLayerSizes)
		{
			file << cvLayerSize << " ";
		}
		file << "|";

		// This code theoretically could produce different results on different machines, as neither the size of 'float' nor the endianness is guaranteed
		// I don't really care though

		for (int_fast32_t weight_index = 0; weight_index < ConsPlayerConstants::WeightCount(); weight_index++)
		{
			uint8_t * weight_bytes = reinterpret_cast<uint8_t *>(weights + weight_index);
			for (int_fast32_t weight_byte_index = 0; weight_byte_index < sizeof(float); weight_byte_index++)
			{
				file << weight_bytes[weight_byte_index];
			}
		}
		for (int_fast32_t bias_index = 0; bias_index < ConsPlayerConstants::BiasCount(); bias_index++)
		{
			uint8_t * bias_bytes = reinterpret_cast<uint8_t *>(biases + bias_index);
			for (int_fast32_t bias_byte_index = 0; bias_byte_index < sizeof(float); bias_byte_index++)
			{
				file << bias_bytes[bias_byte_index];
			}
		}
		file.close();
	}
	else
	{
		std::cerr << "Cannot save unrecognized version "s + std::to_string(ConsPlayerConstants::version) << std::endl;
		throw std::runtime_error("UnrecognizedVersion");
	}
}

void ConsPlayer::Load(std::filesystem::path filename, ConsPlayer * player)
{
	std::ifstream file;
	file.open(filename, std::ios::in | std::ios::binary);

	int_fast32_t version;
	file >> version;

	if (version != ConsPlayerConstants::version)
	{
		file.close();
		std::cerr << "Cannot load version "s + std::to_string(version) + "; currently compiled for version " + std::to_string(ConsPlayerConstants::version) << std::endl;
		throw std::runtime_error("IncorrectVersion");
	}
	if (version == 1)
	{
		int_fast32_t
			tvInputSize,
			cvInputSize,
			tvLayerCount,
			cvLayerCount,
			maxGameMoveCount,
			maxMoveEvaluationCount,
			tieBreakWithPieceScore,
			generationSize,
			generationTournamentSize
		;
		float mutationRate, mutationAmount;

		file >> tvInputSize;
		file >> cvInputSize;
		file >> tvLayerCount;
		file >> cvLayerCount;
		file >> maxGameMoveCount;
		file >> maxMoveEvaluationCount;
		file >> tieBreakWithPieceScore;
		file >> generationSize;
		file >> generationTournamentSize;
		file >> mutationRate;
		file >> mutationAmount;

		bool constants_correct = true;
		if (tvInputSize != ConsPlayerConstants::tvInputSize)
		{
			constants_correct = false;
			std::cerr << "Failed to load "s + filename.string() + "; Expected tvInputSize = " << ConsPlayerConstants::tvInputSize << " but file has " << tvInputSize << std::endl;
		}
		if (cvInputSize != ConsPlayerConstants::cvInputSize)
		{
			constants_correct = false;
			std::cerr << "Failed to load "s + filename.string() + "; Expected cvInputSize = " << ConsPlayerConstants::cvInputSize << " but file has " << cvInputSize << std::endl;
		}
		if (tvLayerCount != ConsPlayerConstants::tvLayerCount)
		{
			constants_correct = false;
			std::cerr << "Failed to load "s + filename.string() + "; Expected tvLayerCount = " << ConsPlayerConstants::tvLayerCount << " but file has " << tvLayerCount << std::endl;
		}
		if (cvLayerCount != ConsPlayerConstants::cvLayerCount)
		{
			constants_correct = false;
			std::cerr << "Failed to load "s + filename.string() + "; Expected cvLayerCount = " << ConsPlayerConstants::cvLayerCount << " but file has " << cvLayerCount << std::endl;
		}
		if (maxGameMoveCount != ConsPlayerConstants::maxGameMoveCount)
		{
			constants_correct = false;
			std::cerr << "Failed to load "s + filename.string() + "; Expected maxGameMoveCount = " << ConsPlayerConstants::maxGameMoveCount << " but file has " << maxGameMoveCount << std::endl;
		}
		if (maxMoveEvaluationCount != ConsPlayerConstants::maxMoveEvaluationCount)
		{
			constants_correct = false;
			std::cerr << "Failed to load "s + filename.string() + "; Expected maxMoveEvaluationCount = " << ConsPlayerConstants::maxMoveEvaluationCount << " but file has " << maxMoveEvaluationCount << std::endl;
		}
		if (static_cast<bool>(tieBreakWithPieceScore) != ConsPlayerConstants::tieBreakWithPieceScore)
		{
			constants_correct = false;
			std::cerr << "Failed to load "s + filename.string() + "; Expected tieBreakWithPieceScore = " << ConsPlayerConstants::tieBreakWithPieceScore << " but file has " << static_cast<bool>(tieBreakWithPieceScore) << std::endl;
		}
		if (generationSize != ConsPlayerConstants::generationSize)
		{
			constants_correct = false;
			std::cerr << "Failed to load "s + filename.string() + "; Expected generationSize = " << ConsPlayerConstants::generationSize << " but file has " << generationSize << std::endl;
		}
		if (generationTournamentSize != ConsPlayerConstants::generationTournamentSize)
		{
			constants_correct = false;
			std::cerr << "Failed to load "s + filename.string() + "; Expected generationTournamentSize = " << ConsPlayerConstants::generationTournamentSize << " but file has " << generationTournamentSize << std::endl;
		}
		if (mutationRate != ConsPlayerConstants::mutationRate)
		{
			constants_correct = false;
			std::cerr << "Failed to load "s + filename.string() + "; Expected mutationRate = " << ConsPlayerConstants::mutationRate << " but file has " << mutationRate << std::endl;
		}
		if (mutationAmount != ConsPlayerConstants::mutationAmount)
		{
			constants_correct = false;
			std::cerr << "Failed to load "s + filename.string() + "; Expected mutationAmount = " << ConsPlayerConstants::mutationAmount << " but file has " << mutationAmount << std::endl;
		}
		if (!constants_correct)
		{
			throw std::runtime_error("IncorrectConstants");
		}

		int_fast32_t tvLayerSizes[ConsPlayerConstants::tvLayerCount];
		int_fast32_t cvLayerSizes[ConsPlayerConstants::cvLayerCount];
		bool tv_layers_correct = true;
		bool cv_layers_correct = true;

		for (int_fast32_t layer_index = 0; layer_index < tvLayerCount; layer_index++)
		{
			file >> tvLayerSizes[layer_index];
			if (tvLayerSizes[layer_index] != ConsPlayerConstants::tvLayerSizes[layer_index])
			{
				tv_layers_correct = false;
			}
		}
		for (int_fast32_t layer_index = 0; layer_index < cvLayerCount; layer_index++)
		{
			file >> cvLayerSizes[layer_index];
			if (cvLayerSizes[layer_index] != ConsPlayerConstants::cvLayerSizes[layer_index])
			{
				cv_layers_correct = false;
			}
		}

		if (!tv_layers_correct)
		{
			std::cerr << "Failed to load "s + filename.string() + "; Expected tvLayerSizes = ";
			PrintArray(std::cerr, ConsPlayerConstants::tvLayerSizes, ConsPlayerConstants::tvLayerCount);
			std::cerr << " but file has ";
			PrintArray(std::cerr, tvLayerSizes, tvLayerCount);
			std::cerr << std::endl;
		}
		if (!cv_layers_correct)
		{
			std::cerr << "Failed to load "s + filename.string() + "; Expected cvLayerSizes = ";
			PrintArray(std::cerr, ConsPlayerConstants::cvLayerSizes, ConsPlayerConstants::cvLayerCount);
			std::cerr << " but file has ";
			PrintArray(std::cerr, cvLayerSizes, cvLayerCount);
			std::cerr << std::endl;
		}
		if (!tv_layers_correct || !cv_layers_correct)
		{
			throw std::runtime_error("IncorrectConstants");
		}

		char seperator;
		file >> seperator;
		_ASSERT(seperator == '|');

		for (int_fast32_t weight_index = 0; weight_index < ConsPlayerConstants::WeightCount(); weight_index++)
		{
			uint8_t * weight_bytes = reinterpret_cast<uint8_t *>(player->weights + weight_index);
			for (int_fast32_t weight_byte_index = 0; weight_byte_index < sizeof(float); weight_byte_index++)
			{
				int byte = file.get();
				if (byte == EOF)
				{
					throw std::runtime_error("Failed to load "s + filename.string() + "; Unexpected EOF"s);
				}
				weight_bytes[weight_byte_index] = (uint8_t)byte;
			}
		}
		for (int_fast32_t bias_index = 0; bias_index < ConsPlayerConstants::BiasCount(); bias_index++)
		{
			uint8_t * bias_bytes = reinterpret_cast<uint8_t *>(player->biases + bias_index);
			for (int_fast32_t bias_byte_index = 0; bias_byte_index < sizeof(float); bias_byte_index++)
			{
				int byte = file.get();
				if (byte == EOF)
				{
					throw std::runtime_error("Failed to load "s + filename.string() + "; Unexpected EOF"s);
				}
				bias_bytes[bias_byte_index] = (uint8_t)byte;
			}
		}

		_ASSERT(file.peek() == EOF);
		file.close();
	}
	else
	{
		file.close();
		std::cerr << "Cannot load unrecognized version "s + std::to_string(version) << std::endl;
		throw std::runtime_error("UnrecognizedVersion");
	}
}

ConsPlayer * ConsPlayer::Clone() const
{
	ConsPlayer * result = new ConsPlayer();
	memcpy(result->weights, weights, ConsPlayerConstants::WeightCount() * sizeof(float));
	memcpy(result->biases, biases, ConsPlayerConstants::BiasCount() * sizeof(float));
	return result;
}
